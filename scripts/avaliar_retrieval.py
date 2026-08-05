#!/usr/bin/env python3
"""Avaliação quantitativa da recuperação.

Mede se o chunk que responde a pergunta aparece — e em que posição — entre os
recuperados. Sem isso, "melhorei o retrieval" é opinião.

Não gasta tokens: `recuperar()` usa embeddings locais e SQLite, sem chamar a API.
Isso torna barato rodar a cada mudança no chunking, no modelo de embedding ou na
fusão, que é justamente quando dá para regredir sem perceber.

Duas métricas:

  MRR (Mean Reciprocal Rank) — média de 1/posição do primeiro acerto. Recompensa
    trazer a resposta em primeiro, não só trazê-la. 1,00 é o teto.
  Recall@k — fração de perguntas cuja resposta apareceu entre os k primeiros.
    É o que determina se o modelo *pode* acertar; o que ele faz com o contexto
    é outro problema.

Uso:
    python scripts/avaliar_retrieval.py            # com o índice já populado
    make eval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edital_rag.query.retrieve import recuperar  # noqa: E402

# O gabarito é de um documento específico, e a busca precisa ser restrita a ele.
# Sem o filtro, a medição mente: a numeração de seção se repete entre editais, e
# um chunk "8" de outro documento indexado conta como acerto da pergunta sobre o
# cronograma deste. Com três editais no índice, a métrica vira ruído.
DOCUMENTO = "Edital-081_2026-assinado.pdf"

# Gabarito para o Edital FADE 081/2026. Cada entrada aceita mais de uma seção
# quando a resposta legitimamente vive em mais de um lugar do documento.
GABARITO: list[tuple[str, list[str]]] = [
    ("qual o prazo final de inscrição?", ["8"]),
    ("até quando posso me inscrever?", ["8"]),
    ("quando são as entrevistas?", ["8"]),
    ("qual a remuneração mensal?", ["3"]),
    ("qual a carga horária semanal?", ["3"]),
    ("quantos pontos vale o mestrado?", ["6", "6.1"]),
    ("quantos pontos vale experiência com LLMs?", ["6", "6.1"]),
    ("quais documentos preciso anexar na inscrição?", ["5.1.6"]),
    ("quem participa da banca da entrevista?", ["5.2.7"]),
    ("como é pontuada a entrevista?", ["5.2.2"]),
    ("qual o nível de inglês exigido?", ["2.1.2"]),
    ("o trabalho é presencial ou remoto?", ["9.4", "9.3"]),
    ("quanto tempo vale o processo seletivo?", ["1.1"]),
    ("como faço para interpor recurso?", ["7.4", "7.2", "7.3"]),
    ("quantos candidatos vão para a entrevista?", ["4.2"]),
]


def posicao_do_acerto(pergunta: str, esperadas: list[str], k: int) -> int | None:
    """Posição (1-indexada) do primeiro chunk esperado, ou None se não veio."""
    recuperados = recuperar(pergunta, top_k=k, documento=DOCUMENTO)
    for posicao, recuperado in enumerate(recuperados, start=1):
        if recuperado.chunk.secao in esperadas:
            return posicao
    return None


def main(k: int = 6) -> int:
    print(f"Avaliando {len(GABARITO)} perguntas com top_k={k} em {DOCUMENTO}\n")
    print(f"{'pos':<5} {'esperado':<16} pergunta")
    print("-" * 78)

    reciprocos: list[float] = []
    acertos = 0

    for pergunta, esperadas in GABARITO:
        posicao = posicao_do_acerto(pergunta, esperadas, k)

        if posicao is None:
            marca, reciproco = "  ✗", 0.0
        else:
            marca = f"  {posicao}" if posicao > 1 else "  1"
            reciproco = 1.0 / posicao
            acertos += 1

        reciprocos.append(reciproco)
        print(f"{marca:<5} {','.join(esperadas):<16} {pergunta}")

    mrr = sum(reciprocos) / len(reciprocos)
    recall = acertos / len(GABARITO)

    print("-" * 78)
    print(f"MRR       {mrr:.3f}")
    print(f"Recall@{k}  {recall:.3f}  ({acertos}/{len(GABARITO)})")

    # Sinaliza regressão para uso em CI. O limiar é deliberadamente baixo: serve
    # para pegar quebra estrutural, não para travar experimentação.
    return 0 if recall >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 6))
