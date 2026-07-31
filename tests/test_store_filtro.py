"""Testes do filtro por documento nas duas buscas.

O caso que importa é o desequilíbrio: um edital pequeno indexado ao lado de um
grande. É onde a implementação ingênua quebra — filtrar por documento *depois*
do k-NN devolve menos de k resultados, ou nenhum, porque o sqlite-vec já
escolheu os vizinhos olhando o índice inteiro.

A falha seria silenciosa: a busca continua respondendo, só que com material pior
ou vazio, e a resposta vira "o edital não responde isso" para uma pergunta que o
edital responde.
"""

from __future__ import annotations

import pytest

from edital_rag.config import get_config
from edital_rag.index import store
from edital_rag.models import Chunk

DIM = get_config().embedding_dim


def _vetor(eixo: int) -> list[float]:
    """Vetor unitário num eixo. Eixos diferentes ficam maximamente distantes."""
    vetor = [0.0] * DIM
    vetor[eixo] = 1.0
    return vetor


def _chunk(documento: str, secao: str, texto: str) -> Chunk:
    return Chunk(
        documento=documento, secao=secao, texto=texto, pagina_inicio=1, pagina_fim=1
    )


@pytest.fixture
def conexao(tmp_path):
    with store.conectar(tmp_path / "teste.db") as conexao:
        store.inicializar(conexao)

        # 50 chunks de um edital, 3 de outro — e os do menor deliberadamente
        # longe da consulta, para que nenhum deles entre no k-NN global.
        grande = [
            _chunk("grande.pdf", f"1.{i}", f"prazo de inscrição item {i}") for i in range(50)
        ]
        pequeno = [
            _chunk("pequeno.pdf", f"2.{i}", f"prazo de inscrição anexo {i}") for i in range(3)
        ]

        store.inserir(conexao, grande, [_vetor(0)] * len(grande))
        store.inserir(conexao, pequeno, [_vetor(1)] * len(pequeno))
        yield conexao


class TestBuscaVetorial:
    def test_filtro_nao_perde_recall_no_documento_pequeno(self, conexao):
        # k=10 pedidos, e o documento só tem 3 chunks: devolve os 3, e não zero.
        resultado = store.buscar_vetorial(conexao, _vetor(0), k=10, documento="pequeno.pdf")

        assert [chunk.documento for chunk, _ in resultado] == ["pequeno.pdf"] * 3

    def test_sem_filtro_a_consulta_cai_no_documento_dominante(self, conexao):
        resultado = store.buscar_vetorial(conexao, _vetor(0), k=10)

        assert {chunk.documento for chunk, _ in resultado} == {"grande.pdf"}


class TestBuscaLexical:
    def test_filtro_restringe_ao_documento(self, conexao):
        # O termo aparece nos dois editais; só um deve voltar.
        resultado = store.buscar_lexical(
            conexao, "prazo de inscrição", k=20, documento="pequeno.pdf"
        )

        assert resultado
        assert {chunk.documento for chunk, _ in resultado} == {"pequeno.pdf"}

    def test_sem_filtro_mistura_os_dois_editais(self, conexao):
        resultado = store.buscar_lexical(conexao, "prazo de inscrição", k=53)

        assert {chunk.documento for chunk, _ in resultado} == {"grande.pdf", "pequeno.pdf"}


def test_nomes_documentos(conexao):
    assert store.nomes_documentos(conexao) == ["grande.pdf", "pequeno.pdf"]
