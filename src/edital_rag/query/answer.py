"""Geração da resposta com citação obrigatória.

## O requisito central do projeto

Toda resposta cita seção e página, ou declara que não encontrou. Não é enfeite: é
a diferença entre uma ferramenta utilizável e um brinquedo.

Alucinação em consulta a edital tem consequência real — alguém perde um prazo,
deixa de anexar um documento obrigatório, ou é eliminado por não cumprir um
requisito que o sistema informou errado. Uma resposta sem procedência verificável
é indistinguível de uma inventada, então o sistema é construído para não produzir
esse tipo de resposta.

Três mecanismos garantem isso:

1. **Structured outputs** (`output_config.format`) — a resposta é validada contra
   um JSON Schema. O campo `citacoes` existe por construção; não depende de o
   modelo lembrar de citar.
2. **`encontrado: false`** — caminho explícito para "o edital não responde isso".
   Sem essa saída, um modelo pressionado a responder tende a preencher a lacuna.
3. **`trecho` literal** — cada citação carrega o texto copiado do edital, o que
   torna a verificação uma comparação de strings em vez de um ato de fé.

## Por que `effort: low`

A tarefa aqui não é raciocinar do zero: é sintetizar e citar um contexto que já
foi recuperado. `low` responde bem, mais rápido e mais barato. Se as respostas
saírem rasas em perguntas que cruzam várias seções, subir para `medium` é o
primeiro ajuste — não trocar o modelo.
"""

from __future__ import annotations

import json
import logging

import anthropic

from edital_rag.config import get_config
from edital_rag.models import ChunkRecuperado, Resposta

logger = logging.getLogger(__name__)

INSTRUCOES = """\
Você responde perguntas sobre editais e documentos normativos, usando exclusivamente \
os trechos fornecidos.

Regras:

1. Use apenas o que está nos trechos. Não complete com conhecimento geral sobre \
editais, nem com o que "normalmente" esses documentos dizem.
2. Toda afirmação precisa de citação: número da seção, página e o trecho literal \
que a sustenta.
3. Se os trechos não responderem à pergunta, defina `encontrado` como false e \
explique em `resposta` o que especificamente falta. Não tente uma resposta \
aproximada.
4. Quando o documento for contraditório, diga isso e cite os dois lados. Editais \
frequentemente se contradizem, e apontar a contradição é mais útil do que \
escolher um lado em silêncio.
5. Cite prazos, valores e quantidades exatamente como aparecem no documento. Não \
converta, arredonde nem reformate.
6. Responda em português do Brasil, de forma direta. Sem preâmbulo.\
"""

# Schema escrito à mão em vez de derivado do modelo Pydantic: structured outputs
# exige `additionalProperties: false` em todo objeto e não aceita `$defs`, que é o
# que `model_json_schema()` gera para tipos aninhados.
ESQUEMA_RESPOSTA = {
    "type": "object",
    "properties": {
        "encontrado": {
            "type": "boolean",
            "description": "true se os trechos respondem à pergunta; false caso contrário",
        },
        "resposta": {
            "type": "string",
            "description": "A resposta em PT-BR. Se encontrado=false, explique o que falta.",
        },
        "citacoes": {
            "type": "array",
            "description": "Vazio somente quando encontrado=false",
            "items": {
                "type": "object",
                "properties": {
                    "secao": {"type": "string", "description": 'Ex. "5.1.6"'},
                    "pagina": {"type": "integer"},
                    "trecho": {
                        "type": "string",
                        "description": "Texto literal copiado do edital",
                    },
                },
                "required": ["secao", "pagina", "trecho"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["encontrado", "resposta", "citacoes"],
    "additionalProperties": False,
}


def _montar_contexto(recuperados: list[ChunkRecuperado]) -> str:
    return "\n\n---\n\n".join(r.chunk.para_contexto() for r in recuperados)


def responder(pergunta: str, recuperados: list[ChunkRecuperado]) -> Resposta:
    """Gera a resposta ancorada nos chunks recuperados."""
    if not recuperados:
        return Resposta(
            encontrado=False,
            resposta=(
                "Nenhum trecho relevante foi encontrado no índice. "
                "Verifique se o documento foi ingerido em /ingest."
            ),
        )

    config = get_config()
    cliente = anthropic.Anthropic(api_key=config.anthropic_api_key or None)

    mensagem = (
        f"Trechos do documento:\n\n{_montar_contexto(recuperados)}\n\n"
        f"---\n\nPergunta: {pergunta}"
    )

    resposta_api = cliente.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        output_config={
            "effort": config.effort,
            "format": {"type": "json_schema", "schema": ESQUEMA_RESPOSTA},
        },
        system=INSTRUCOES,
        messages=[{"role": "user", "content": mensagem}],
    )

    # Classificadores de segurança podem recusar a requisição: HTTP 200 com
    # stop_reason "refusal" e `content` vazio. Ler content[0] direto quebraria.
    if resposta_api.stop_reason == "refusal":
        logger.warning("Requisição recusada pelos classificadores: %s", resposta_api.stop_details)
        return Resposta(
            encontrado=False,
            resposta="A requisição foi recusada pelos filtros de segurança do modelo.",
        )

    if resposta_api.stop_reason == "max_tokens":
        logger.warning("Resposta truncada em max_tokens=%d", config.max_tokens)

    texto = next((b.text for b in resposta_api.content if b.type == "text"), None)
    if not texto:
        return Resposta(encontrado=False, resposta="O modelo não retornou conteúdo textual.")

    try:
        return Resposta.model_validate(json.loads(texto))
    except (json.JSONDecodeError, ValueError) as erro:
        # Com structured outputs isso é improvável — mas truncamento por
        # max_tokens produz JSON incompleto, e falhar em silêncio seria pior.
        logger.error("JSON inválido do modelo: %s", erro)
        return Resposta(
            encontrado=False,
            resposta=f"Não foi possível interpretar a resposta do modelo: {erro}",
        )
