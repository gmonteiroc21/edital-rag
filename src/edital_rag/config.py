"""Configuração via variáveis de ambiente."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EDITAL_RAG_",
        extra="ignore",
    )

    # A chave não usa o prefixo — é o nome que o SDK da Anthropic já espera.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    model: str = "claude-opus-5"

    # low é adequado: a tarefa é sintetizar contexto já recuperado, não raciocinar
    # do zero. Ver README › Decisões de arquitetura.
    effort: str = "low"

    max_tokens: int = 2048

    db_path: Path = Path("data/index.db")
    # Precisa estar em TextEmbedding.list_supported_models() do fastembed
    # instalado. `embedding_dim` tem de casar com a dimensão do modelo — é o
    # esquema da tabela virtual do sqlite-vec, não um número livre.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    top_k: int = 6
    # Peso da busca vetorial na fusão com a lexical (0 = só BM25, 1 = só vetorial).
    peso_vetorial: float = 0.6

    # --- Variáveis do experimento de indexação (ver scripts/experimento.py) ---
    # Ambas nasceram como hipóteses plausíveis e ambas foram medidas e rejeitadas
    # no Edital FADE 081/2026 (MRR: 0,558 sem nenhuma; 0,517 só hierarquia;
    # 0,544 só verbalização; 0,436 com as duas).
    #
    # Ficam como flags, e não removidas, porque o efeito depende do documento:
    # a hierarquia é a única coisa que torna a seção 9.4 recuperável, e num corpus
    # de documentos mais curtos ou com seções mais ambíguas pode compensar.
    # O padrão é o que venceu a medição, não o que parecia certo.
    indexar_hierarquia: bool = False
    verbalizar_tabelas: bool = False


@lru_cache
def get_config() -> Config:
    return Config()
