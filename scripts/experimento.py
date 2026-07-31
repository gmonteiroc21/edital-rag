#!/usr/bin/env python3
"""Compara as quatro configurações de indexação, reindexando a cada uma.

Contexto: duas mudanças foram introduzidas juntas — incluir a hierarquia no texto
indexado e verbalizar tabelas — e o MRR caiu de 0,513 para 0,436. Com duas
variáveis alteradas ao mesmo tempo, não dá para saber qual causou o quê.

Este script isola cada uma. Reindexa o documento em cada configuração e roda o
mesmo gabarito, imprimindo MRR e Recall lado a lado.

Não gasta tokens: indexação e recuperação são inteiramente locais.

Uso (dentro do container):
    python scripts/experimento.py data/edital.pdf
    make experimento PDF=data/edital.pdf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from avaliar_retrieval import GABARITO, posicao_do_acerto  # noqa: E402

import edital_rag.config as cfg  # noqa: E402
from edital_rag.ingest.pipeline import ingerir_pdf  # noqa: E402

CONFIGURACOES = [
    ("nenhuma (linha de base)", False, False),
    ("só hierarquia", True, False),
    ("só verbalização", False, True),
    ("ambas", True, True),
]


def _aplicar(hierarquia: bool, verbalizar: bool) -> None:
    """Reconfigura o processo. `get_config` é cacheado — precisa ser invalidado."""
    os.environ["EDITAL_RAG_INDEXAR_HIERARQUIA"] = str(hierarquia)
    os.environ["EDITAL_RAG_VERBALIZAR_TABELAS"] = str(verbalizar)
    cfg.get_config.cache_clear()


def medir(k: int) -> tuple[float, float, dict[str, int | None]]:
    posicoes = {p: posicao_do_acerto(p, e, k) for p, e in GABARITO}
    reciprocos = [1.0 / p if p else 0.0 for p in posicoes.values()]
    acertos = sum(1 for p in posicoes.values() if p)
    return sum(reciprocos) / len(reciprocos), acertos / len(posicoes), posicoes


def main(caminho_pdf: str, k: int = 6) -> int:
    resultados = []

    for rotulo, hierarquia, verbalizar in CONFIGURACOES:
        _aplicar(hierarquia, verbalizar)
        ingerir_pdf(caminho_pdf)
        mrr, recall, posicoes = medir(k)
        resultados.append((rotulo, mrr, recall, posicoes))
        print(f"  {rotulo:<24} MRR {mrr:.3f}   Recall@{k} {recall:.3f}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'configuração':<24} {'MRR':>7} {'Recall':>8}")
    print("-" * 78)
    melhor = max(resultados, key=lambda r: (r[1], r[2]))
    for rotulo, mrr, recall, _ in resultados:
        marca = "  ←" if rotulo == melhor[0] else ""
        print(f"{rotulo:<24} {mrr:>7.3f} {recall:>8.3f}{marca}")

    print("\n" + "=" * 78)
    print("Onde cada configuração diverge (posição por pergunta):\n")
    print(f"{'pergunta':<44} " + " ".join(f"{r[0][:8]:>9}" for r in resultados))
    print("-" * 78)
    for pergunta, _ in GABARITO:
        linha = [f"{r[3][pergunta] or '-':>9}" for r in resultados]
        if len(set(linha)) > 1:  # só mostra o que mudou entre configurações
            print(f"{pergunta[:43]:<44} " + " ".join(linha))

    print(f"\nMelhor: {melhor[0]} (MRR {melhor[1]:.3f})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python scripts/experimento.py <caminho-do-pdf>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
