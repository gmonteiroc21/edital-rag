#!/usr/bin/env python3
"""Captura as telas da demo do README, sem clicar em nada.

    docker compose up -d
    python3 scripts/capturar_demo.py

Depende do deep link da interface (`/?q=...&doc=...`): a página abre com a
consulta já feita, então o Chrome headless consegue fotografar o resultado
renderizado. As imagens vão para docs/.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image, ImageChops

BASE = "http://localhost:8000/"
DOC = "Edital-081_2026-assinado.pdf"
RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "docs"

# A ordem responde a três objeções diferentes, nesta sequência: funciona e a
# citação confere; acerta dentro de tabela, onde RAG de PDF costuma falhar;
# e recusa em vez de inventar.
TELAS = [
    ("01-citacao.png", "quais documentos preciso anexar na inscrição?"),
    ("02-tabela.png", "quantos pontos vale experiência com LLMs?"),
    ("03-recusa.png", "o edital oferece plano de saúde ou vale-alimentação?"),
]


def capturar(chrome, destino, pergunta, altura=1600):
    url = BASE + "?" + urlencode({"doc": DOC, "q": pergunta})
    with tempfile.TemporaryDirectory() as perfil:
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--user-data-dir=%s" % perfil,
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                "--window-size=1100,%d" % altura,
                # A resposta depende de uma chamada à API da Anthropic: o
                # orçamento de tempo virtual segura o screenshot até ela voltar.
                "--virtual-time-budget=90000",
                "--screenshot=%s" % destino,
                url,
            ],
            check=True,
            capture_output=True,
        )
    aparar(destino)


def aparar(caminho, margem=24):
    """Corta o vazio abaixo do conteúdo — a altura da janela é fixa, a da
    página não."""
    img = Image.open(caminho).convert("RGB")
    fundo = Image.new("RGB", img.size, img.getpixel((img.width - 2, img.height - 2)))
    caixa = ImageChops.difference(img, fundo).getbbox()
    if not caixa:
        return
    baixo = min(img.height, caixa[3] + margem)
    img.crop((0, 0, img.width, baixo)).save(caminho, optimize=True)


def main():
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        sys.exit("Chrome/Chromium não encontrado")

    SAIDA.mkdir(exist_ok=True)
    for nome, pergunta in TELAS:
        destino = SAIDA / nome
        print("capturando %s — %s" % (nome, pergunta))
        capturar(chrome, destino, pergunta)
        print("  %s (%d KB)" % (destino, destino.stat().st_size // 1024))


if __name__ == "__main__":
    main()
