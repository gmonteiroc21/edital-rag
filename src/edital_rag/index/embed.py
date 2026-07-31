"""Geração de embeddings.

Usa `fastembed` (ONNX Runtime) em vez de `sentence-transformers` (PyTorch). Mesma
qualidade para este modelo, mas a imagem Docker fica em ~600MB em vez de ~2.5GB.
Num projeto cujo critério de sucesso inclui "o avaliador consegue rodar com um
comando", o tamanho da imagem é requisito, não detalhe.

Modelo padrão: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(384 dimensões, ~220MB). Escolhido por ser multilíngue com bom desempenho em
português — a maioria dos modelos pequenos é treinada só em inglês e degrada
bastante em PT-BR.

⚠️ **Prefixos são específicos da família do modelo.** Modelos E5 exigem prefixos
assimétricos (`query:` na pergunta, `passage:` nos documentos) e degradam sem
eles. Modelos `paraphrase-*` não usam prefixo nenhum — aplicá-los faz o texto
literal "query: " entrar no embedding e piorar a similaridade.

Nos dois casos o erro é silencioso: o sistema continua respondendo, só que pior.
Por isso a escolha é derivada do nome do modelo em vez de ficar hardcoded — trocar
`EDITAL_RAG_EMBEDDING_MODEL` não pode exigir que alguém lembre de ajustar isto
junto.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastembed import TextEmbedding

from edital_rag.config import get_config

logger = logging.getLogger(__name__)


def _usa_prefixos_e5(nome_modelo: str) -> bool:
    """Famílias E5 (e5-small/base/large, multilingual-e5-*) exigem os prefixos."""
    return "e5" in nome_modelo.lower()


@lru_cache
def _modelo() -> TextEmbedding:
    config = get_config()
    logger.info(
        "Carregando embedding: %s (prefixos E5: %s)",
        config.embedding_model,
        _usa_prefixos_e5(config.embedding_model),
    )
    return TextEmbedding(model_name=config.embedding_model)


def embutir_passagens(textos: list[str]) -> list[list[float]]:
    """Embeddings para os chunks do documento."""
    if not textos:
        return []
    if _usa_prefixos_e5(get_config().embedding_model):
        textos = [f"passage: {t}" for t in textos]
    return [vetor.tolist() for vetor in _modelo().embed(textos)]


def embutir_pergunta(pergunta: str) -> list[float]:
    """Embedding para a pergunta do usuário."""
    if _usa_prefixos_e5(get_config().embedding_model):
        pergunta = f"query: {pergunta}"
    return next(iter(_modelo().embed([pergunta]))).tolist()
