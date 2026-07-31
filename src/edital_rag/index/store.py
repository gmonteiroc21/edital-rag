"""Persistência do índice: SQLite + sqlite-vec + FTS5.

Por que SQLite e não Postgres/pgvector, Chroma ou Pinecone: o índice inteiro é um
arquivo. Não há serviço para subir, credencial para configurar, nem container
extra no compose. `docker compose up` e funciona.

Para o volume de dados aqui — dezenas de editais, alguns milhares de chunks — a
busca é instantânea. Trocar por um vetorial dedicado seria otimização para um
problema que este projeto não tem.

Três tabelas, um único arquivo:
  chunks      — os dados e metadados
  vec_chunks  — índice vetorial (sqlite-vec), busca semântica
  fts_chunks  — índice invertido (FTS5), busca lexical com BM25

`rowid` é compartilhado entre as três, o que dispensa junções por chave externa.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from edital_rag.config import get_config
from edital_rag.models import Chunk

logger = logging.getLogger(__name__)

ESQUEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    documento     TEXT    NOT NULL,
    secao         TEXT    NOT NULL DEFAULT '',
    titulo        TEXT    NOT NULL DEFAULT '',
    hierarquia    TEXT    NOT NULL DEFAULT '[]',
    texto         TEXT    NOT NULL,
    pagina_inicio INTEGER NOT NULL,
    pagina_fim    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_documento ON chunks(documento);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    texto,
    tokenize = "unicode61 remove_diacritics 2"
);
"""


@contextmanager
def conectar(caminho: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Abre conexão com a extensão sqlite-vec carregada."""
    config = get_config()
    destino = caminho or config.db_path
    destino.parent.mkdir(parents=True, exist_ok=True)

    conexao = sqlite3.connect(destino)
    conexao.row_factory = sqlite3.Row
    try:
        conexao.enable_load_extension(True)
        sqlite_vec.load(conexao)
        conexao.enable_load_extension(False)
        yield conexao
    finally:
        conexao.close()


def inicializar(conexao: sqlite3.Connection) -> None:
    """Cria as tabelas. A dimensão do vetor é fixa no esquema da tabela virtual."""
    config = get_config()
    conexao.executescript(ESQUEMA)
    conexao.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding FLOAT[{config.embedding_dim}]
        )
        """
    )
    conexao.commit()


def remover_documento(conexao: sqlite3.Connection, documento: str) -> int:
    """Apaga um documento das três tabelas. Torna a ingestão idempotente."""
    ids = [linha["id"] for linha in conexao.execute(
        "SELECT id FROM chunks WHERE documento = ?", (documento,)
    )]
    if not ids:
        return 0

    marcadores = ",".join("?" * len(ids))
    conexao.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({marcadores})", ids)
    conexao.execute(f"DELETE FROM fts_chunks WHERE rowid IN ({marcadores})", ids)
    conexao.execute("DELETE FROM chunks WHERE documento = ?", (documento,))
    conexao.commit()
    logger.info("Removidos %d chunks anteriores de %s", len(ids), documento)
    return len(ids)


def inserir(
    conexao: sqlite3.Connection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Grava chunks e embeddings, mantendo o rowid alinhado entre as três tabelas."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Descompasso: {len(chunks)} chunks e {len(embeddings)} embeddings. "
            "Os índices precisam corresponder um-a-um."
        )

    for chunk, vetor in zip(chunks, embeddings, strict=True):
        cursor = conexao.execute(
            """
            INSERT INTO chunks
                (documento, secao, titulo, hierarquia, texto, pagina_inicio, pagina_fim)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.documento,
                chunk.secao,
                chunk.titulo,
                json.dumps(chunk.hierarquia, ensure_ascii=False),
                chunk.texto,
                chunk.pagina_inicio,
                chunk.pagina_fim,
            ),
        )
        chunk_id = cursor.lastrowid

        conexao.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(vetor)),
        )
        # Mesma representação usada para gerar o embedding — ver
        # Chunk.para_indexacao(). Os dois índices precisam enxergar o mesmo texto.
        conexao.execute(
            "INSERT INTO fts_chunks(rowid, texto) VALUES (?, ?)",
            (chunk_id, chunk.para_indexacao()),
        )

    conexao.commit()
    return len(chunks)


def _linha_para_chunk(linha: sqlite3.Row) -> Chunk:
    return Chunk(
        documento=linha["documento"],
        secao=linha["secao"],
        titulo=linha["titulo"],
        hierarquia=json.loads(linha["hierarquia"]),
        texto=linha["texto"],
        pagina_inicio=linha["pagina_inicio"],
        pagina_fim=linha["pagina_fim"],
    )


def buscar_vetorial(
    conexao: sqlite3.Connection,
    vetor: list[float],
    k: int,
    documento: str | None = None,
) -> list[tuple[Chunk, float]]:
    """k-NN por distância. Converte distância em score onde maior é melhor.

    `documento` restringe a busca a um único arquivo indexado.

    O filtro entra como `rowid IN (...)` dentro da cláusula do MATCH, e não como
    um `WHERE c.documento = ?` depois do JOIN. A diferença não é estilística: o
    sqlite-vec resolve a restrição de rowid **antes** de escolher os k vizinhos,
    então a busca devolve os k melhores *do documento*. Filtrar depois do JOIN
    escolheria os k melhores do índice inteiro e jogaria fora o que não fosse do
    documento — devolvendo menos de k resultados, ou nenhum quando o edital
    filtrado é o menor dos indexados.
    """
    filtro = "AND v.rowid IN (SELECT id FROM chunks WHERE documento = :documento)"
    linhas = conexao.execute(
        f"""
        SELECT c.*, v.distance AS distancia
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH :vetor AND k = :k
              {filtro if documento else ""}
        ORDER BY v.distance
        """,
        {
            "vetor": sqlite_vec.serialize_float32(vetor),
            "k": k,
            "documento": documento,
        },
    ).fetchall()

    return [(_linha_para_chunk(linha), 1.0 / (1.0 + linha["distancia"])) for linha in linhas]


def _consulta_fts(pergunta: str) -> str:
    """Converte texto livre em consulta FTS5 válida.

    Necessário porque FTS5 tem sintaxe própria: aspas, hífens, asteriscos e as
    palavras AND/OR/NOT/NEAR são operadores. Passar a pergunta do usuário direto
    gera `sqlite3.OperationalError` no primeiro acento ou hífen.

    Cada termo é isolado e citado, e os termos são unidos por OR — recall alto na
    etapa lexical, deixando a precisão para a fusão com a busca vetorial.
    """
    termos = re.findall(r"\w{3,}", pergunta.lower(), flags=re.UNICODE)
    if not termos:
        return '""'
    return " OR ".join(f'"{termo}"' for termo in termos)


def buscar_lexical(
    conexao: sqlite3.Connection,
    pergunta: str,
    k: int,
    documento: str | None = None,
) -> list[tuple[Chunk, float]]:
    """BM25 via FTS5. Captura o que o vetorial erra: códigos, siglas, números de item.

    Aqui o filtro é um predicado comum: o `LIMIT` só é aplicado depois do WHERE,
    então restringir por documento não custa recall.
    """
    consulta = _consulta_fts(pergunta)
    try:
        linhas = conexao.execute(
            f"""
            SELECT c.*, bm25(fts_chunks) AS rank
            FROM fts_chunks
            JOIN chunks c ON c.id = fts_chunks.rowid
            WHERE fts_chunks MATCH :consulta
                  {"AND c.documento = :documento" if documento else ""}
            ORDER BY rank
            LIMIT :k
            """,
            {"consulta": consulta, "k": k, "documento": documento},
        ).fetchall()
    except sqlite3.OperationalError as erro:
        logger.warning("Busca lexical falhou para %r: %s", consulta, erro)
        return []

    # bm25() do SQLite retorna valores negativos, quanto menor melhor. Invertemos
    # para a convenção "maior é melhor" usada na fusão.
    return [(_linha_para_chunk(linha), -linha["rank"]) for linha in linhas]


def contar(conexao: sqlite3.Connection) -> dict[str, int]:
    """Estatísticas do índice, para o endpoint /health e para debug."""
    total = conexao.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    documentos = conexao.execute(
        "SELECT COUNT(DISTINCT documento) AS n FROM chunks"
    ).fetchone()["n"]
    return {"chunks": total, "documentos": documentos}


def nomes_documentos(conexao: sqlite3.Connection) -> list[str]:
    """Nomes dos documentos indexados. Usado para validar o filtro de consulta."""
    return [
        linha["documento"]
        for linha in conexao.execute(
            "SELECT DISTINCT documento FROM chunks ORDER BY documento"
        )
    ]


def listar_documentos(conexao: sqlite3.Connection) -> list[dict[str, object]]:
    linhas = conexao.execute(
        """
        SELECT documento, COUNT(*) AS chunks, MAX(pagina_fim) AS paginas
        FROM chunks GROUP BY documento ORDER BY documento
        """
    ).fetchall()
    return [dict(linha) for linha in linhas]
