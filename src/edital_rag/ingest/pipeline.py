"""Orquestração da ingestão: PDF → páginas → chunks → embeddings → índice."""

from __future__ import annotations

import logging
from pathlib import Path

from edital_rag.index import embed, store
from edital_rag.ingest.chunk import dividir_em_secoes
from edital_rag.ingest.extract import extrair_paginas

logger = logging.getLogger(__name__)


def ingerir_pdf(caminho: str | Path, nome: str | None = None) -> dict[str, object]:
    """Indexa um PDF. Idempotente: reingerir o mesmo nome substitui o anterior."""
    caminho = Path(caminho)
    documento = nome or caminho.name

    paginas = extrair_paginas(caminho)
    chunks = dividir_em_secoes(paginas, documento)

    if not chunks:
        raise ValueError(f"Nenhum chunk gerado a partir de {documento}")

    # `para_indexacao()` e não `texto`: o vetor precisa enxergar a mesma coisa
    # que o índice lexical, hierarquia inclusa.
    embeddings = embed.embutir_passagens([c.para_indexacao() for c in chunks])

    with store.conectar() as conexao:
        store.inicializar(conexao)
        substituidos = store.remover_documento(conexao, documento)
        inseridos = store.inserir(conexao, chunks, embeddings)

    logger.info("Ingerido %s: %d chunks, %d páginas", documento, inseridos, len(paginas))

    return {
        "documento": documento,
        "paginas": len(paginas),
        "chunks": inseridos,
        "chunks_substituidos": substituidos,
        "secoes": sorted({c.secao for c in chunks if c.secao}),
    }
