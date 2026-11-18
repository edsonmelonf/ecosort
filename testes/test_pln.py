from src.pln.dados_intencoes import TESTE
from src.pln.entidades import extrair_entidades
from src.pln.intencao import ReconhecedorIntencao, textos_e_rotulos
from src.pln.preprocessamento import normalizar


def test_normalizar_remove_acentos_e_pontuacao():
    assert normalizar("  Posso RECICLÁR isso?! ") == "posso reciclar isso"


def test_entidades_material_e_condicao():
    e = extrair_entidades("Essa caixa de pizza está engordurada, pode reciclar?")
    assert "cardboard" in e["materiais"]
    assert "gordura" in e["condicoes"]


def test_entidades_sem_falso_positivo():
    e = extrair_entidades("Qual a cor da lixeira?")
    assert e == {"materiais": [], "condicoes": []}


def test_palavra_inteira_nao_casa_parte_de_palavra():
    # "pet" não deve casar dentro de "carpete"
    assert extrair_entidades("jogo o carpete fora")["materiais"] == []


def test_intencao_acuracia_minima_no_teste():
    modelo = ReconhecedorIntencao.treinar()
    x, y = textos_e_rotulos(TESTE)
    acc = sum(p == r for p, r in zip(modelo.pipeline.predict(x), y)) / len(y)
    assert acc >= 0.85


def test_intencao_exemplos_claros():
    modelo = ReconhecedorIntencao.treinar()
    assert modelo.prever("qual a cor da lixeira pra isso?")[0] == "cor_lixeira"
    assert modelo.prever("isso pode ser reciclado?")[0] == "pode_reciclar"
