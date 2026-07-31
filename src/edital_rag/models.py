"""Estruturas de dados compartilhadas entre ingestão, índice e consulta."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Pagina(BaseModel):
    """Uma página extraída do PDF, com o número preservado.

    O número de página é o que permite citar a fonte de forma verificável.
    Sem ele, a citação não vale nada — o usuário não consegue conferir.
    """

    numero: int = Field(description="1-indexado, como o leitor de PDF mostra")
    texto: str


class Chunk(BaseModel):
    """Uma seção do edital, com a hierarquia do documento preservada.

    A decisão central do projeto: o chunk é uma *seção numerada*, não uma janela
    de N caracteres. Editais têm estrutura hierárquica explícita (1., 2.1, 5.1.6)
    e cortar no meio de uma cláusula destrói a unidade de sentido — que é
    justamente o que faz um RAG responder mal.

    `hierarquia` carrega o caminho dos ancestrais, então o chunk "5.1.6" sabe que
    pertence a "5.1 Primeira Etapa" e a "5. DA SELEÇÃO". Esse contexto vai junto
    para o modelo.
    """

    documento: str = Field(description="Nome do arquivo de origem")
    secao: str = Field(description='Numeração da seção, ex. "5.1.6". Vazio para preâmbulo.')
    titulo: str = Field(default="", description="Primeira linha da seção, quando parece um título")
    hierarquia: list[str] = Field(
        default_factory=list,
        description='Caminho dos ancestrais, ex. ["5. DA SELEÇÃO", "5.1 Primeira Etapa"]',
    )
    texto: str
    pagina_inicio: int
    pagina_fim: int

    @property
    def caminho(self) -> str:
        """Representação legível da hierarquia, usada no prompt e na citação."""
        partes = [*self.hierarquia, self.secao or "preâmbulo"]
        return " › ".join(p for p in partes if p)

    def para_contexto(self) -> str:
        """Como o chunk é apresentado ao modelo — com procedência explícita."""
        cabecalho = f"[seção {self.secao or 'preâmbulo'} | página {self.pagina_inicio}]"
        if self.hierarquia:
            cabecalho += f"\n(em: {' › '.join(self.hierarquia)})"
        return f"{cabecalho}\n{self.texto}"

    def para_indexacao(self) -> str:
        """Como o chunk é representado nos índices — vetorial **e** lexical.

        Existe como método único, chamado pelos dois índices, por um motivo
        concreto: numa versão anterior o índice lexical recebia a hierarquia e o
        vetorial não. A busca lexical enxergava a estrutura do documento e a
        semântica não, e nada acusava a diferença.

        Corrigir a assimetria melhorou o MRR de 0,513 para 0,558 — mas removendo
        a hierarquia dos dois lados, não adicionando aos dois. Prefixar o caminho
        dos ancestrais **dilui** o vetor de chunks que já são específicos: o
        embedding passa a apontar para o significado do cabeçalho da seção em vez
        do conteúdo do item. Ver `scripts/experimento.py`.
        """
        from edital_rag.config import get_config

        if not get_config().indexar_hierarquia:
            return self.texto

        partes = [*self.hierarquia, self.secao, self.titulo, self.texto]
        return "\n".join(p for p in partes if p)


class ChunkRecuperado(BaseModel):
    """Chunk com o score da busca, para inspeção e debug do retrieval."""

    chunk: Chunk
    score: float
    origem: str = Field(description='"vetorial", "lexical" ou "hibrido"')


class Citacao(BaseModel):
    secao: str
    pagina: int
    trecho: str = Field(description="Citação literal do edital que sustenta a resposta")


class Resposta(BaseModel):
    encontrado: bool = Field(
        description="False quando o edital não responde a pergunta. Nesse caso, `resposta` "
        "explica o que falta em vez de inventar."
    )
    resposta: str
    citacoes: list[Citacao] = Field(default_factory=list)
