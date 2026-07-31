"""API HTTP.

Três endpoints: ingerir um PDF, perguntar, verificar saúde. A superfície é
pequena de propósito — o valor do projeto está no chunking e no retrieval, não
em variedade de rotas.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from edital_rag.config import get_config
from edital_rag.index import store
from edital_rag.ingest.pipeline import ingerir_pdf
from edital_rag.models import Resposta
from edital_rag.query.answer import responder
from edital_rag.query.retrieve import recuperar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Assistente de consulta a editais",
    description=(
        "RAG sobre editais em PDF com citação obrigatória de seção e página. "
        "Chunking semântico por hierarquia de seção e busca híbrida (vetorial + BM25)."
    ),
    version="0.1.0",
)


class PerguntaRequest(BaseModel):
    pergunta: str = Field(min_length=3, examples=["Qual o prazo final de inscrição?"])
    top_k: int | None = Field(default=None, ge=1, le=20)
    documento: str | None = Field(
        default=None,
        description="Restringe a busca a um documento indexado (o nome que /health lista). "
        "Omitir busca em todos — o que, com mais de um edital no índice, mistura "
        "documentos no mesmo contexto.",
        examples=["Edital-081_2026-assinado.pdf"],
    )


class TrechoUsado(BaseModel):
    documento: str
    secao: str
    caminho: str
    pagina: int
    score: float
    origem: str


class PerguntaResponse(BaseModel):
    pergunta: str
    resposta: Resposta
    trechos_usados: list[TrechoUsado] = Field(
        description="O que a busca recuperou. Exposto para tornar o retrieval auditável — "
        "quando a resposta sai errada, é aqui que se vê se o erro foi da busca ou do modelo."
    )


# Página única, sem build. A tese do projeto é "docker compose up e funciona";
# um passo de `npm install` antes de ver a tela contradiria isso.
INDEX_HTML = Path(__file__).resolve().parent / "estatico" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Interface web: upload do PDF e consulta."""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health", tags=["sistema"])
def health() -> dict[str, object]:
    """Estado do serviço e do índice."""
    config = get_config()
    try:
        with store.conectar() as conexao:
            store.inicializar(conexao)
            estatisticas = store.contar(conexao)
            documentos = store.listar_documentos(conexao)
    except Exception as erro:  # noqa: BLE001 — health nunca deve derrubar o processo
        logger.exception("Falha ao consultar o índice")
        return {"status": "degradado", "erro": str(erro)}

    return {
        "status": "ok",
        "modelo": config.model,
        "effort": config.effort,
        "chave_configurada": bool(config.anthropic_api_key),
        "indice": estatisticas,
        "documentos": documentos,
    }


# Deliberadamente `def`, não `async def`: o corpo é todo bloqueante (pdfplumber,
# embeddings) e num handler assíncrono travaria o event loop pela duração inteira
# da indexação — o servidor pararia de responder e o navegador abortaria o upload.
# Como `def`, o FastAPI executa isto no threadpool e o loop segue livre.
@app.post("/ingest", tags=["ingestão"])
def ingest(arquivo: UploadFile = File(...)) -> dict[str, object]:
    """Indexa um PDF. Reenviar o mesmo nome de arquivo substitui a versão anterior."""
    if not (arquivo.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .pdf")

    # Grava em disco porque pdfplumber precisa de um caminho seekable.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporario:
        shutil.copyfileobj(arquivo.file, temporario)
        caminho = Path(temporario.name)

    try:
        return ingerir_pdf(caminho, nome=arquivo.filename)
    except (ValueError, FileNotFoundError) as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
    finally:
        caminho.unlink(missing_ok=True)


@app.post("/ask", response_model=PerguntaResponse, tags=["consulta"])
def ask(requisicao: PerguntaRequest) -> PerguntaResponse:
    """Responde uma pergunta sobre os documentos indexados, com citações."""
    try:
        recuperados = recuperar(
            requisicao.pergunta, top_k=requisicao.top_k, documento=requisicao.documento
        )
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro

    resposta = responder(requisicao.pergunta, recuperados)

    return PerguntaResponse(
        pergunta=requisicao.pergunta,
        resposta=resposta,
        trechos_usados=[
            TrechoUsado(
                documento=r.chunk.documento,
                secao=r.chunk.secao or "preâmbulo",
                caminho=r.chunk.caminho,
                pagina=r.chunk.pagina_inicio,
                score=round(r.score, 5),
                origem=r.origem,
            )
            for r in recuperados
        ],
    )
