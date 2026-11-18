"""Testa a integração com um classificador visual falso (rápido, sem pesos)."""
import numpy as np
from PIL import Image

from src.pipeline import AssistenteResiduos
from src.pln.intencao import ReconhecedorIntencao
from src.visao_computacional.classificador import PrevisaoVisual
from src.visao_computacional.qualidade import analisar_qualidade


class CNNFalsa:
    def __init__(self, classe="metal", conf=0.95):
        self.classe, self.conf = classe, conf

    def prever(self, imagem):
        return PrevisaoVisual(self.classe, self.conf, "plastic", (1 - self.conf) / 2,
                              {self.classe: self.conf})


def foto_boa(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(60, 200, (240, 320, 3), dtype=np.uint8))


def test_fluxo_integrado_imagem_mais_pergunta():
    a = AssistenteResiduos(classificador=CNNFalsa("metal"), intencao=ReconhecedorIntencao.treinar())
    r = a.responder(foto_boa(), "qual a cor da lixeira pra isso?")
    assert r.decisao.classe_final == "metal"
    assert "amarela" in r.decisao.resposta


def test_texto_muda_a_resposta_do_agente():
    a = AssistenteResiduos(classificador=CNNFalsa("paper", 0.95),
                           intencao=ReconhecedorIntencao.treinar())
    limpo = a.responder(foto_boa(), "esse papel pode ser reciclado?").decisao.resposta
    sujo = a.responder(foto_boa(), "esse papel engordurado pode ser reciclado?").decisao.resposta
    assert limpo.startswith("Sim") and sujo.startswith("Não")


def test_qualidade_detecta_foto_escura_e_desfocada():
    escura = Image.fromarray(np.full((200, 200, 3), 10, dtype=np.uint8))
    q = analisar_qualidade(escura)
    assert "foto muito escura" in q.problemas and "foto desfocada" in q.problemas
    assert analisar_qualidade(foto_boa()).boa
