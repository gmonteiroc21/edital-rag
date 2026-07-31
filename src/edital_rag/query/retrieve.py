"""Recuperação híbrida: busca vetorial + lexical, fundidas por RRF.

## Por que híbrida

As duas buscas erram de formas complementares:

- **Vetorial** entende paráfrase ("quanto paga?" → "Remuneração Mensal Bruta"),
  mas é fraca em identificadores exatos. Perguntar "o que diz o item 5.1.6" pode
  não trazer o item 5.1.6, porque a numeração carrega pouco sinal semântico.
- **Lexical (BM25)** acerta identificadores, siglas, códigos e valores em cheio,
  mas falha em qualquer pergunta que não repita as palavras do documento.

Consultas a editais têm as duas naturezas na mesma frase — "qual a pontuação do
item de LLMs" mistura conceito e referência estrutural. Rodar as duas e fundir é
mais robusto que escolher uma.

## Por que RRF e não soma ponderada de scores

Os scores não são comparáveis: a distância vetorial vive em [0, 2] e o BM25 do
SQLite é uma magnitude negativa sem limite superior. Normalizar por min-max é
instável quando uma das listas tem poucos resultados — dois itens fazem o melhor
virar 1.0 e o pior 0.0, independentemente da qualidade real.

Reciprocal Rank Fusion usa apenas a *posição*, não o valor. É invariante à escala
e ao esquema de pontuação de cada buscador.
"""

from __future__ import annotations

import logging

from edital_rag.config import get_config
from edital_rag.index import embed, store
from edital_rag.models import Chunk, ChunkRecuperado

logger = logging.getLogger(__name__)

# Constante de amortecimento do RRF. 60 é o valor da publicação original
# (Cormack et al., 2009) e o padrão de facto: alto o suficiente para que as
# primeiras posições não dominem completamente o ranking final.
K_RRF = 60


def _chave(chunk: Chunk) -> tuple[str, str, int]:
    """Identidade de um chunk para deduplicação entre as duas listas."""
    return (chunk.documento, chunk.secao, chunk.pagina_inicio)


def recuperar(pergunta: str, top_k: int | None = None) -> list[ChunkRecuperado]:
    """Recupera os chunks mais relevantes para a pergunta.

    Cada buscador contribui com uma lista ranqueada; a fusão soma as
    contribuições recíprocas ponderadas por `peso_vetorial`.
    """
    config = get_config()
    k = top_k or config.top_k
    # Sobre-recupera em cada buscador para dar material à fusão.
    k_bruto = max(k * 3, 20)

    with store.conectar() as conexao:
        store.inicializar(conexao)

        vetorial = store.buscar_vetorial(conexao, embed.embutir_pergunta(pergunta), k_bruto)
        lexical = store.buscar_lexical(conexao, pergunta, k_bruto)

    logger.info(
        "Recuperação para %r: %d vetoriais, %d lexicais",
        pergunta[:60], len(vetorial), len(lexical)
    )

    pontos: dict[tuple[str, str, int], float] = {}
    objetos: dict[tuple[str, str, int], Chunk] = {}
    origens: dict[tuple[str, str, int], set[str]] = {}

    for lista, peso, nome in (
        (vetorial, config.peso_vetorial, "vetorial"),
        (lexical, 1.0 - config.peso_vetorial, "lexical"),
    ):
        for posicao, (chunk, _score) in enumerate(lista, start=1):
            chave = _chave(chunk)
            pontos[chave] = pontos.get(chave, 0.0) + peso / (K_RRF + posicao)
            objetos.setdefault(chave, chunk)
            origens.setdefault(chave, set()).add(nome)

    ordenados = sorted(pontos.items(), key=lambda par: par[1], reverse=True)[:k]

    return [
        ChunkRecuperado(
            chunk=objetos[chave],
            score=score,
            origem="hibrido" if len(origens[chave]) > 1 else next(iter(origens[chave])),
        )
        for chave, score in ordenados
    ]
