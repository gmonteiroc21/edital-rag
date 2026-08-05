"""Testes do chunking — a decisão de design central do projeto.

Os casos de falso positivo são os mais importantes. O padrão de numeração de
seção é ambíguo com números que aparecem no corpo do texto, e uma seção fantasma
criada a partir de "15.193 registros" contamina o índice de forma silenciosa: a
busca continua funcionando e devolvendo resultados piores.
"""

from __future__ import annotations

import pytest

from edital_rag.ingest.chunk import casar_anexo, casar_secao, dividir_em_secoes
from edital_rag.models import Pagina


def _paginas(*textos: str) -> list[Pagina]:
    return [Pagina(numero=i, texto=t) for i, t in enumerate(textos, start=1)]


class TestPadraoSecao:
    @pytest.mark.parametrize(
        "linha,esperado",
        [
            ("1. DAS DISPOSIÇÕES PRELIMINARES", "1"),
            ("2.1 Áreas e Perfil de Formação:", "2.1"),
            ("5.1.6 Para comprovação documental, os candidatos", "5.1.6"),
            ("9.4 As atividades serão executadas presencialmente", "9.4"),
            ("10.1 É de responsabilidade do candidato", "10.1"),
        ],
    )
    def test_reconhece_numeracao_de_secao(self, linha: str, esperado: str) -> None:
        match = casar_secao(linha)
        assert match is not None
        assert match[0] == esperado

    @pytest.mark.parametrize(
        "linha,esperado",
        [
            # PDFs reais frequentemente perdem o espaço após a numeração em
            # cabeçalhos com kerning apertado. Regressão observada no edital
            # FADE 081/2026 — ver extract.X_TOLERANCE.
            ("1.DAS DISPOSIÇÕES PRELIMINARES", "1"),
            ("2.1Áreas e Perfil de Formação:", "2.1"),
            ("2.1.2Requisitos:", "2.1.2"),
            ("5.1Primeira Etapa - Inscrições", "5.1"),
        ],
    )
    def test_reconhece_secao_sem_espaco_apos_numeracao(self, linha: str, esperado: str) -> None:
        match = casar_secao(linha)
        assert match is not None, f"cabeçalho perdido: {linha!r}"
        assert match[0] == esperado

    @pytest.mark.parametrize(
        "linha",
        [
            "15.193 registros e 73 colunas",  # número do corpo do texto
            "2024. A permanência escolar é uma condição",  # ano seguido de ponto
            "8,7 milhões de jovens de 14 a 29 anos",  # decimal com vírgula
            "R$ 5.904,23 bruto mensal",  # valor monetário
            "1.234.567 habitantes",  # milhar
            "5.904,23",  # valor isolado
            "12.5% dos candidatos",  # percentual
            "",  # linha vazia
            "   ",  # só espaço
            "Texto normal sem numeração",
            # Regressões reais do edital FADE 081/2026: criavam as seções
            # fantasma "40" e "13". Ambas são rejeitadas pela guarda de
            # maiúscula — conteúdo de norma nunca abre em minúscula.
            "40 horas (Trabalho híbrido)",  # célula do Quadro I
            "13h00min às 17h00min, de segunda a sexta.",  # horário
            "08h00min às 12h00min",
        ],
    )
    def test_ignora_numeros_do_corpo_do_texto(self, linha: str) -> None:
        assert casar_secao(linha.strip()) is None

    @pytest.mark.parametrize("linha", ["01 Inscrições + Análise", "02 Entrevista"])
    def test_ignora_numeracao_de_linha_de_tabela(self, linha: str) -> None:
        """Zero à esquerda indica linha de tabela, não seção de norma."""
        assert casar_secao(linha) is None


class TestHierarquia:
    def test_chunk_carrega_o_caminho_dos_ancestrais(self) -> None:
        paginas = _paginas(
            "5. DA SELEÇÃO\n"
            "Texto introdutório da seção de seleção do processo.\n"
            "5.1 Primeira Etapa - Inscrições e Análise Documental\n"
            "As inscrições serão realizadas de forma online conforme cronograma.\n"
            "5.1.6 Para comprovação documental, os candidatos deverão anexar "
            "obrigatoriamente ao formulário, em documento único, formato PDF."
        )

        chunks = dividir_em_secoes(paginas, "edital.pdf")
        por_secao = {c.secao: c for c in chunks}

        assert "5.1.6" in por_secao
        filho = por_secao["5.1.6"]
        assert len(filho.hierarquia) == 2
        assert filho.hierarquia[0].startswith("5 DA SELEÇÃO")
        assert filho.hierarquia[1].startswith("5.1 Primeira Etapa")

    def test_raiz_nao_tem_ancestrais(self) -> None:
        paginas = _paginas("1. DAS DISPOSIÇÕES PRELIMINARES\nO prazo de validade é de 4 meses.")
        chunks = dividir_em_secoes(paginas, "edital.pdf")
        assert chunks[0].hierarquia == []

    def test_caminho_legivel_inclui_a_secao(self) -> None:
        paginas = _paginas(
            "7. DOS CRITÉRIOS DE DESEMPATE\n"
            "Conteúdo da seção sobre critérios de desempate e recursos.\n"
            "7.2 Não serão analisados os recursos interpostos fora do prazo."
        )
        chunks = dividir_em_secoes(paginas, "edital.pdf")
        alvo = next(c for c in chunks if c.secao == "7.2")
        assert "7.2" in alvo.caminho
        assert "›" in alvo.caminho


class TestPaginacao:
    def test_pagina_registrada_e_a_do_inicio_da_secao(self) -> None:
        paginas = _paginas(
            "1. PRIMEIRA SEÇÃO\nConteúdo suficientemente longo da primeira seção do edital.",
            "2. SEGUNDA SEÇÃO\nConteúdo suficientemente longo da segunda seção do edital.",
            "3. TERCEIRA SEÇÃO\nConteúdo suficientemente longo da terceira seção do edital.",
        )
        chunks = dividir_em_secoes(paginas, "edital.pdf")
        assert {c.secao: c.pagina_inicio for c in chunks} == {"1": 1, "2": 2, "3": 3}

    def test_secao_que_atravessa_paginas_mantem_inicio_e_fim(self) -> None:
        paginas = _paginas(
            "4. SEÇÃO LONGA\nPrimeira parte do conteúdo, ainda na página um do documento.",
            "Continuação do mesmo item, agora já na página dois, sem nova numeração.",
        )
        chunks = dividir_em_secoes(paginas, "edital.pdf")
        alvo = next(c for c in chunks if c.secao == "4")
        assert alvo.pagina_inicio == 1
        assert alvo.pagina_fim == 2
        assert "página dois" in alvo.texto


class TestAnexos:
    """Anexos: onde a numeração decimal do edital acaba.

    Nenhum edital numera `ANEXO V` como `18`, então sem um padrão próprio todo o
    bloco de anexos — inclusive o cronograma, que é onde moram as datas que mais
    se pergunta — vira continuação da última seção numerada.
    """

    @pytest.mark.parametrize(
        "linha,esperado",
        [
            ("ANEXO V", "ANEXO V"),
            ("ANEXO I", "ANEXO I"),
            ("ANEXO 1", "ANEXO 1"),
            ("APÊNDICE II", "APÊNDICE II"),
            ("ANEXO ÚNICO", "ANEXO ÚNICO"),
            ("ANEXO I - REQUISITOS E DESCRIÇÃO SUMÁRIA DOS CARGOS", "ANEXO I"),
        ],
    )
    def test_reconhece_cabecalho_de_anexo(self, linha: str, esperado: str) -> None:
        casado = casar_anexo(linha)
        assert casado is not None and casado[0] == esperado

    @pytest.mark.parametrize(
        "linha",
        [
            # A lista de anexos que o próprio edital traz no corpo (seção 1.10 do
            # UFPE 12/2026) — referência, não cabeçalho.
            "Anexo I - REQUISITOS E DESCRIÇÃO SUMÁRIA DOS CARGOS DE NÍVEL D",
            "Os seguintes anexos integram o presente Edital:",
            "anexo para a FADE/UFPE, através do e-mail ufpe2026@fadeconcursos.org.br",
            "ANEXOS DESTE EDITAL",
        ],
    )
    def test_ignora_referencia_a_anexo_no_corpo(self, linha: str) -> None:
        assert casar_anexo(linha) is None

    def test_anexo_abre_secao_propria(self) -> None:
        paginas = _paginas(
            "17.12 O prazo de impugnação será de 05 dias corridos da publicação no DOU.",
            "ANEXO V\nCRONOGRAMA\nINSCRIÇÃO VIA INTERNET 12/08 a 08/09/2026 no site do concurso.",
        )
        chunks = dividir_em_secoes(paginas, "edital.pdf")

        anexo = [c for c in chunks if c.secao == "ANEXO V"]
        assert len(anexo) == 1
        assert anexo[0].pagina_inicio == 2
        assert anexo[0].titulo == "CRONOGRAMA"
        assert "12/08 a 08/09/2026" in anexo[0].texto
        # E o cronograma não contamina a seção numerada anterior.
        assert "12/08" not in next(c.texto for c in chunks if c.secao == "17.12")

    def test_lista_de_anexos_no_corpo_nao_estilhaca_a_secao(self) -> None:
        paginas = _paginas(
            "1.10 Os seguintes anexos integram o presente Edital:\n"
            "Anexo I - REQUISITOS E DESCRIÇÃO SUMÁRIA DOS CARGOS\n"
            "Anexo II - CONTEÚDO PROGRAMÁTICO DAS PROVAS\n"
            "Anexo V - CRONOGRAMA DO CERTAME"
        )
        chunks = dividir_em_secoes(paginas, "edital.pdf")
        assert [c.secao for c in chunks] == ["1.10"]


class TestIntegridade:
    def test_nenhum_conteudo_e_perdido(self) -> None:
        """Toda linha significativa do documento aparece em algum chunk.

        Perda silenciosa de conteúdo é o modo de falha mais perigoso da ingestão:
        o sistema responde "não encontrei" para algo que está no documento.
        """
        paginas = _paginas(
            "1. PRIMEIRA\nAlfa bravo charlie delta echo foxtrot golf hotel.\n"
            "1.1 SUBSEÇÃO\nÍndia julliet kilo lima mike november oscar papa.\n"
            "2. SEGUNDA\nQuebec romeu sierra tango uniform victor whiskey xray."
        )
        chunks = dividir_em_secoes(paginas, "edital.pdf")
        tudo = " ".join(c.texto for c in chunks)

        for palavra in ("alfa", "índia", "quebec", "xray"):
            assert palavra in tudo.lower(), f"conteúdo perdido: {palavra}"

    def test_documento_sem_numeracao_produz_chunk_unico(self) -> None:
        paginas = _paginas(
            "Este é um documento em prosa corrida, sem qualquer numeração de "
            "seção, que mesmo assim precisa ser indexável pelo sistema."
        )
        chunks = dividir_em_secoes(paginas, "carta.pdf")
        assert len(chunks) == 1
        assert chunks[0].secao == ""

    def test_secao_muito_longa_e_subdividida(self) -> None:
        corpo = "\n".join(f"Linha {i} com conteúdo de preenchimento." * 3 for i in range(200))
        chunks = dividir_em_secoes(_paginas(f"1. SEÇÃO ENORME\n{corpo}"), "edital.pdf")

        assert len(chunks) > 1
        # Os fragmentos preservam a identidade da seção de origem.
        assert all(c.secao == "1" for c in chunks)

    def test_cada_fragmento_cita_a_propria_pagina(self) -> None:
        """Um anexo de várias páginas não pode citar a página de abertura em todos.

        A citação é a promessa central do projeto: cinco fragmentos apontando
        para a mesma página são quatro citações erradas, e erradas de um jeito
        que só quem abre o PDF descobre.
        """
        corpo = "Conteúdo de preenchimento do anexo, com texto suficiente. " * 40
        paginas = _paginas(f"ANEXO I\n{corpo}", corpo, corpo)
        chunks = dividir_em_secoes(paginas, "edital.pdf")

        assert len(chunks) > 1
        assert all(c.secao == "ANEXO I" for c in chunks)
        assert chunks[0].pagina_inicio == 1
        assert chunks[-1].pagina_inicio > 1
