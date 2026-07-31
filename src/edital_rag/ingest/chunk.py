"""Chunking semântico por hierarquia de seção.

## O problema com chunking por janela fixa

A abordagem padrão — quebrar o documento a cada N caracteres com sobreposição —
funciona razoavelmente em prosa corrida e mal em documentos normativos. Editais,
contratos e regulamentos têm uma propriedade que a janela fixa ignora: a unidade
de sentido é a *cláusula*, e ela é delimitada explicitamente pela numeração.

Cortar no meio do item 5.1.6 produz dois chunks que individualmente não
respondem a nada. Pior: o recuperador devolve o pedaço com maior similaridade
lexical, que é frequentemente a metade que não contém a informação decisiva.

## A abordagem daqui

Quebrar exatamente onde o documento se quebra — na numeração — e carregar a
hierarquia dos ancestrais no metadado de cada chunk. O chunk do item "5.1.6"
sabe que vive dentro de "5.1 Primeira Etapa", que vive dentro de "5. DA SELEÇÃO".

Isso importa porque o texto de um item frequentemente só faz sentido no contexto
do pai. "5.1.6 Para comprovação documental, os candidatos deverão anexar..."
depende de saber que 5.1 é a *primeira* etapa — sem isso, a resposta pode ser
atribuída à etapa errada.
"""

from __future__ import annotations

import logging
import re

from edital_rag.ingest.extract import FIM_TABELA, INICIO_TABELA
from edital_rag.models import Chunk, Pagina

logger = logging.getLogger(__name__)

# Numeração de seção no início da linha: "1", "2.1", "5.1.6", opcionalmente com
# ponto final, seguida do conteúdo.
#
# Duas restrições fazem o trabalho pesado, e nenhuma é opcional:
#
# 1. Cada componente tem no máximo 2 dígitos. Sem isso, números do corpo do texto
#    viram seções fantasma, que contaminam o índice em silêncio.
# 2. O conteúdo precisa começar com **letra**. É o que permite tornar o espaço
#    opcional (`\s*`) sem passar a casar decimais e valores monetários.
#
# O espaço é opcional porque PDFs reais frequentemente não têm um: o edital da
# FADE extrai "2.1.2Requisitos:" mesmo com a tolerância de extração calibrada.
# Exigir `\s+` fazia o parser perder justamente os cabeçalhos de seção — os
# nós que sustentam toda a hierarquia.
#
#   "5.1.6 Para comprovação"  -> 5 / 1 / 6, com espaço      -> aceito   ✓
#   "2.1.2Requisitos:"        -> 2 / 1 / 2, sem espaço      -> aceito   ✓
#   "15.193 registros"        -> "193" tem 3 dígitos        -> rejeitado ✓
#   "2024. A permanência"     -> "2024" tem 4 dígitos       -> rejeitado ✓
#   "8,7 milhões"             -> vírgula não é letra        -> rejeitado ✓
#   "1.234.567 habitantes"    -> componente de 3 dígitos    -> rejeitado ✓
#
# Zeros à esquerda são rejeitados: editais numeram seções "1", "2", nunca "01".
# Linhas de tabela, por outro lado, usam "01"/"02" o tempo todo.
PADRAO_SECAO = re.compile(r"^([1-9]\d?(?:\.\d{1,2})*)\.?\s*([^\W\d_].*)$", re.UNICODE)


def casar_secao(linha: str) -> tuple[str, str] | None:
    """Retorna (numeração, conteúdo) se a linha for um cabeçalho de seção.

    Além do padrão, aplica a guarda que o regex sozinho não expressa bem:
    **o conteúdo precisa começar com letra maiúscula.**

    Em texto normativo isso é sempre verdade — títulos e cláusulas abrem com
    maiúscula. E é justamente o que separa uma seção de um falso positivo que
    sobreviveu a todas as outras restrições:

        "9.4 As atividades serão executadas"  -> "As"     -> seção   ✓
        "2.1.2Requisitos:"                    -> "Requi"  -> seção   ✓
        "40 horas (Trabalho híbrido)"         -> "horas"  -> tabela  ✗
        "13h00min às 17h00min"                -> "h00min" -> horário ✗

    Os dois últimos são casos reais do edital FADE 081/2026 que criavam as
    seções fantasma "40" e "13".
    """
    match = PADRAO_SECAO.match(linha)
    if not match:
        return None

    conteudo = match.group(2)
    if not conteudo[:1].isupper():
        return None

    return match.group(1), conteudo

# Seções acima deste tamanho são subdivididas. Raro em edital, mas um anexo longo
# sem numeração interna pode estourar o contexto útil de um único chunk.
MAX_CARACTERES = 3000

# Abaixo disso o chunk é ruído (linha órfã, cabeçalho repetido) e é descartado —
# exceto se tiver numeração de seção, caso em que um item curto é legítimo
# ("7.2 Não serão analisados os recursos interpostos fora do prazo.").
MIN_CARACTERES = 40


def _profundidade(secao: str) -> int:
    return len(secao.split("."))


def _ancestrais(secao: str) -> list[str]:
    """["5", "5.1"] para "5.1.6"."""
    partes = secao.split(".")
    return [".".join(partes[: i + 1]) for i in range(len(partes) - 1)]


def _primeira_linha(texto: str, limite: int = 120) -> str:
    linha = texto.strip().split("\n", 1)[0].strip()
    return linha[:limite]


def _subdividir(texto: str, limite: int = MAX_CARACTERES) -> list[str]:
    """Divide um texto longo em parágrafos, sem cortar no meio de uma frase."""
    if len(texto) <= limite:
        return [texto]

    partes: list[str] = []
    atual: list[str] = []
    tamanho = 0

    for paragrafo in texto.split("\n"):
        if tamanho + len(paragrafo) > limite and atual:
            partes.append("\n".join(atual))
            atual, tamanho = [], 0
        atual.append(paragrafo)
        tamanho += len(paragrafo) + 1

    if atual:
        partes.append("\n".join(atual))
    return partes


def _montar(
    documento: str,
    secao: str,
    corpo: list[str],
    pagina_inicio: int,
    pagina_fim: int,
    titulos: dict[str, str],
) -> list[Chunk]:
    """Constrói os chunks de uma seção fechada, subdividindo se necessário."""
    texto = "\n".join(corpo).strip()
    if not texto:
        return []
    if len(texto) < MIN_CARACTERES and not secao:
        return []

    hierarquia = [
        f"{pai} {titulos[pai]}".strip() for pai in _ancestrais(secao) if pai in titulos
    ]

    return [
        Chunk(
            documento=documento,
            secao=secao,
            titulo=titulos.get(secao, ""),
            hierarquia=hierarquia,
            texto=parte,
            pagina_inicio=pagina_inicio,
            pagina_fim=pagina_fim,
        )
        for parte in _subdividir(texto)
    ]


def dividir_em_secoes(paginas: list[Pagina], documento: str) -> list[Chunk]:
    """Converte páginas extraídas em chunks alinhados à estrutura do documento.

    Percorre as linhas na ordem, abrindo um chunk novo a cada numeração de seção
    encontrada e fechando o anterior. O número da página em que a seção *começa*
    é o que vai para a citação.
    """
    chunks: list[Chunk] = []
    titulos: dict[str, str] = {}

    secao_atual = ""
    corpo: list[str] = []
    pagina_inicio = paginas[0].numero if paginas else 1
    pagina_fim = pagina_inicio

    dentro_de_tabela = False

    for pagina in paginas:
        for linha in pagina.texto.split("\n"):
            despida = linha.strip()

            # Tabelas ficam anexadas à seção corrente. Sem isso, "40 horas" do
            # Quadro I e as linhas "01"/"02" do Quadro II viram seções de topo,
            # e a hierarquia do documento passa a conter nós que não existem.
            if despida == INICIO_TABELA:
                dentro_de_tabela = True
            elif despida == FIM_TABELA:
                dentro_de_tabela = False

            cabecalho = None if dentro_de_tabela else casar_secao(despida)

            if cabecalho:
                # Fecha a seção anterior antes de abrir a nova.
                chunks.extend(
                    _montar(documento, secao_atual, corpo, pagina_inicio, pagina_fim, titulos)
                )

                secao_atual, resto = cabecalho
                resto = resto.strip()
                titulos[secao_atual] = _primeira_linha(resto)
                corpo = [resto]
                pagina_inicio = pagina.numero
                pagina_fim = pagina.numero
            else:
                corpo.append(linha)
                pagina_fim = pagina.numero

    chunks.extend(_montar(documento, secao_atual, corpo, pagina_inicio, pagina_fim, titulos))

    logger.info(
        "%s: %d chunks a partir de %d páginas (profundidade máxima: %d)",
        documento,
        len(chunks),
        len(paginas),
        max((_profundidade(c.secao) for c in chunks if c.secao), default=0),
    )
    return chunks
