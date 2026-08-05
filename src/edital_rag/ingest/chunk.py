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

## Anexos

Numeração decimal cobre o corpo do edital e para exatamente onde os anexos
começam. `ANEXO V / CRONOGRAMA` não casa com nenhum padrão numérico, então antes
disso todo o bloco de anexos era absorvido como continuação da última seção
numerada — no Edital UFPE 12/2026, 31 páginas coladas dentro da seção "17.12".

O estrago é duplo e silencioso. O embedding de um chunk que mistura lista de
classificação, descrição de cargos e cronograma não aponta para nada, e a busca
vetorial deixa de alcançá-lo; e a citação sai com a seção e a página erradas,
que é pior do que não responder. Por isso `ANEXO <numeração>` também abre seção.
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


# Cabeçalho de anexo: "ANEXO V", "ANEXO I - REQUISITOS", "APÊNDICE II".
PADRAO_ANEXO = re.compile(
    r"^(ANEXO|AP[ÊE]NDICE)\s+([IVXLCDM]{1,6}|\d{1,2}|[ÚU]NICO)\b[\s\-–—:.]*(.*)$",
    re.UNICODE,
)


def casar_anexo(linha: str) -> tuple[str, str] | None:
    """Retorna (identificação, resto) se a linha abrir um anexo.

    **A linha precisa estar toda em maiúsculas**, e essa guarda não é cosmética:
    é o que separa o cabeçalho do anexo da *lista* de anexos que o próprio edital
    traz no corpo. O UFPE 12/2026 enumera os seus na seção 1.10 —

        "Anexo I - REQUISITOS E DESCRIÇÃO SUMÁRIA DOS CARGOS"   -> referência ✗
        "ANEXO I"                                               -> cabeçalho  ✓
        "ANEXO V"                                               -> cabeçalho  ✓
        "anexo para a FADE/UFPE, através do e-mail"             -> corpo      ✗

    — e sem a guarda a seção 1.10 seria estilhaçada em cinco anexos fantasma,
    exatamente o tipo de contaminação silenciosa que o chunker já evita nas
    seções numeradas.
    """
    if linha != linha.upper() or len(linha) > 120:
        return None

    match = PADRAO_ANEXO.match(linha)
    if not match:
        return None

    return f"{match.group(1)} {match.group(2)}", match.group(3).strip()


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


# Uma linha do documento com a página em que ela apareceu. As duas informações
# andam juntas até o fim porque a página é o que torna a citação conferível.
Linha = tuple[str, int]


def _subdividir(linhas: list[Linha], limite: int = MAX_CARACTERES) -> list[list[Linha]]:
    """Divide uma seção longa em partes, sem cortar no meio de uma frase."""
    if sum(len(texto) + 1 for texto, _ in linhas) <= limite:
        return [linhas]

    partes: list[list[Linha]] = []
    atual: list[Linha] = []
    tamanho = 0

    for linha in linhas:
        if tamanho + len(linha[0]) > limite and atual:
            partes.append(atual)
            atual, tamanho = [], 0
        atual.append(linha)
        tamanho += len(linha[0]) + 1

    if atual:
        partes.append(atual)
    return partes


def _montar(
    documento: str,
    secao: str,
    corpo: list[Linha],
    titulos: dict[str, str],
) -> list[Chunk]:
    """Constrói os chunks de uma seção fechada, subdividindo se necessário."""
    completo = "\n".join(texto for texto, _ in corpo).strip()
    if not completo:
        return []
    if len(completo) < MIN_CARACTERES and not secao:
        return []

    hierarquia = [
        f"{pai} {titulos[pai]}".strip() for pai in _ancestrais(secao) if pai in titulos
    ]
    # Anexos abrem com a identificação sozinha na linha ("ANEXO V"), então o
    # título só aparece na linha seguinte — que já é corpo.
    titulo = titulos.get(secao) or _primeira_linha(completo)

    chunks: list[Chunk] = []
    for parte in _subdividir(corpo):
        texto = "\n".join(linha for linha, _ in parte).strip()
        if not texto:
            continue

        # Cada parte carrega a própria faixa de páginas. Um anexo de 15 páginas
        # subdividido em cinco chunks citaria a página de abertura nos cinco —
        # cinco citações verificáveis, quatro delas erradas.
        paginas = [pagina for _, pagina in parte]

        chunks.append(
            Chunk(
                documento=documento,
                secao=secao,
                titulo=titulo,
                hierarquia=hierarquia,
                texto=texto,
                pagina_inicio=min(paginas),
                pagina_fim=max(paginas),
            )
        )
    return chunks


def dividir_em_secoes(paginas: list[Pagina], documento: str) -> list[Chunk]:
    """Converte páginas extraídas em chunks alinhados à estrutura do documento.

    Percorre as linhas na ordem, abrindo um chunk novo a cada numeração de seção
    encontrada e fechando o anterior. O número da página em que a seção *começa*
    é o que vai para a citação.
    """
    chunks: list[Chunk] = []
    titulos: dict[str, str] = {}

    secao_atual = ""
    corpo: list[Linha] = []

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

            cabecalho = (
                None if dentro_de_tabela else casar_secao(despida) or casar_anexo(despida)
            )

            if cabecalho:
                # Fecha a seção anterior antes de abrir a nova.
                chunks.extend(_montar(documento, secao_atual, corpo, titulos))

                secao_atual, resto = cabecalho
                resto = resto.strip()
                titulos[secao_atual] = _primeira_linha(resto)
                corpo = [(resto, pagina.numero)]
            else:
                corpo.append((linha, pagina.numero))

    chunks.extend(_montar(documento, secao_atual, corpo, titulos))

    logger.info(
        "%s: %d chunks a partir de %d páginas (profundidade máxima: %d)",
        documento,
        len(chunks),
        len(paginas),
        max((_profundidade(c.secao) for c in chunks if c.secao), default=0),
    )
    return chunks
