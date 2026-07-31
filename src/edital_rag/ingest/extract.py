"""Extração de texto de PDF preservando o número de página.

O número de página não é um detalhe: é o que torna a citação verificável. Uma
resposta que diz "segundo o item 5.1.6 (p. 3)" pode ser conferida em cinco
segundos. Uma que diz só "segundo o edital" não pode ser conferida de forma
alguma — e é indistinguível de uma alucinação.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pdfplumber

from edital_rag.models import Pagina

logger = logging.getLogger(__name__)

# pdfplumber insere um espaço quando o vão horizontal entre dois caracteres passa
# de `x_tolerance`. O padrão (3) é alto demais para cabeçalhos em negrito, cujo
# kerning é mais apertado: o edital da FADE extrai
# "1.DAS DISPOSIÇÕESPRELIMINARES" e "2.1.2Requisitos:" com o valor padrão.
#
# O efeito é pior do que estético. Sem o espaço:
#   - o detector de seção não reconhece o cabeçalho e a hierarquia se perde;
#   - "DOSCRITÉRIOS" nunca é encontrado por uma busca lexical por "critérios".
#
# 2 é o maior valor que corrige o caso real — portanto o menos agressivo.
# Reduzir mais aumentaria o risco oposto: quebrar palavras ao meio em fontes
# com espaçamento naturalmente largo.
X_TOLERANCE = 2.0

# Delimitadores das tabelas serializadas. O chunker os usa para suspender a
# detecção de seção lá dentro — números de célula não são numeração de seção.
INICIO_TABELA = "[TABELA]"
FIM_TABELA = "[/TABELA]"


def _parece_cabecalho(celulas: list[str]) -> bool:
    """Heurística: a primeira linha é cabeçalho se for majoritariamente texto.

    Tabelas de layout (usadas só para posicionar conteúdo) e tabelas cuja
    primeira linha já é dado abriríam a verbalização com rótulos errados —
    "27/07/2026: 11/08/2026" é pior que não verbalizar.
    """
    preenchidas = [c for c in celulas if c]
    if len(preenchidas) < 2:
        return False
    textuais = [c for c in preenchidas if not re.fullmatch(r"[\d\s.,/%R$-]+", c)]
    return len(textuais) >= len(preenchidas) / 2


def _verbalizar(cabecalho: list[str], linha: list[str]) -> str:
    """Reconstrói o par cabeçalho→valor que a serialização em pipe descarta."""
    pares = [
        f"{titulo}: {valor}"
        for titulo, valor in zip(cabecalho, linha, strict=False)
        if titulo and valor
    ]
    return "; ".join(pares) + "." if pares else ""


def _tabela_para_texto(tabela: list[list[str | None]]) -> str:
    """Serializa uma tabela em duas representações complementares.

    Editais concentram informação crítica em tabelas — remuneração, cronograma,
    quadro de pontuação. O extrator de texto puro embaralha as células, então
    extraímos tabelas à parte.

    Mas a serialização em pipe sozinha recupera mal:

        1ª Etapa - Inscrições | 27/07/2026 | 07/08/2026

    Para o BM25 quase não há palavra de conteúdo; para o modelo vetorial é uma
    string sem sintaxe. A informação está presente e a *relação* entre célula e
    cabeçalho não — ela estava na diagramação, que a serialização jogou fora.

    A hipótese era que verbalizar cada linha resolveria isso:

        Atividade: 1ª Etapa - Inscrições; Data Inicial: 27/07/2026;
        Data Final: 07/08/2026.

    Com "Data Final" perto de "07/08/2026", uma pergunta sobre prazo final teria
    em que ancorar.

    **A medição não confirmou.** No Edital FADE 081/2026 o MRR caiu de 0,558 para
    0,544, e o Recall@6 não mudou. A verbalização resgatou uma pergunta ("como
    interpor recurso", 5º → 1º) e degradou duas. Fica atrás da flag
    `EDITAL_RAG_VERBALIZAR_TABELAS`, desligada por padrão — o custo é dobrar o
    tamanho de chunks de tabela, o que ainda pode estourar `MAX_CARACTERES` e
    dividir um quadro em dois. Ver `scripts/experimento.py`.
    """
    linhas = []
    for linha in tabela:
        celulas = [(c or "").replace("\n", " ").strip() for c in linha]
        if any(celulas):
            linhas.append(celulas)

    if not linhas:
        return ""

    from edital_rag.config import get_config

    partes = ["\n".join(" | ".join(linha) for linha in linhas)]

    if get_config().verbalizar_tabelas and len(linhas) > 1 and _parece_cabecalho(linhas[0]):
        frases = [f for linha in linhas[1:] if (f := _verbalizar(linhas[0], linha))]
        if frases:
            partes.append("\n".join(frases))

    return "\n\n".join(partes)


def extrair_paginas(caminho_pdf: str | Path) -> list[Pagina]:
    """Extrai texto e tabelas de cada página, mantendo a numeração 1-indexada."""
    caminho = Path(caminho_pdf)
    if not caminho.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {caminho}")

    paginas: list[Pagina] = []

    with pdfplumber.open(caminho) as pdf:
        for indice, pagina_pdf in enumerate(pdf.pages, start=1):
            partes = []

            texto = pagina_pdf.extract_text(x_tolerance=X_TOLERANCE) or ""
            if texto.strip():
                partes.append(texto)

            for tabela in pagina_pdf.extract_tables() or []:
                serializada = _tabela_para_texto(tabela)
                if serializada:
                    partes.append(f"\n{INICIO_TABELA}\n{serializada}\n{FIM_TABELA}")

            conteudo = "\n".join(partes).strip()
            if not conteudo:
                logger.warning("Página %d de %s veio vazia (PDF escaneado?)", indice, caminho.name)
                continue

            paginas.append(Pagina(numero=indice, texto=conteudo))

    if not paginas:
        raise ValueError(
            f"Nenhum texto extraído de {caminho.name}. "
            "Se for um PDF escaneado, precisa de OCR antes — ver README › Limitações."
        )

    logger.info("Extraídas %d páginas de %s", len(paginas), caminho.name)
    return paginas
