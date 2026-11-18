"""Pipeline integrado: Visão Computacional + PLN + Decisão no mesmo fluxo.

    imagem ──► qualidade + CNN ──┐
                                 ├─► Agente de decisão ─► resposta contextualizada
    pergunta ─► intenção + entidades ┘
"""
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from src.decisao.agente import AgenteDescarte, Decisao, Percepcao
from src.pln.entidades import extrair_entidades
from src.pln.intencao import ReconhecedorIntencao
from src.visao_computacional.classificador import ClassificadorResiduos
from src.visao_computacional.qualidade import analisar_qualidade


@dataclass
class Resultado:
    percepcao: Percepcao
    decisao: Decisao
    previsao_visual: object
    qualidade: object
    entidades: dict
    distribuicao_intencao: dict


class AssistenteResiduos:
    def __init__(self, classificador: Optional[ClassificadorResiduos] = None,
                 intencao: Optional[ReconhecedorIntencao] = None,
                 agente: Optional[AgenteDescarte] = None):
        self.classificador = classificador or ClassificadorResiduos()
        self.intencao = intencao or ReconhecedorIntencao.carregar()
        self.agente = agente or AgenteDescarte()

    def responder(self, imagem: Image.Image, pergunta: str) -> Resultado:
        # 1) Visão computacional
        previsao = self.classificador.prever(imagem)
        qualidade = analisar_qualidade(imagem)
        # 2) PLN
        intencao, conf_i, dist = self.intencao.prever(pergunta)
        entidades = extrair_entidades(pergunta)
        # 3) Decisão (combina os dois resultados)
        percepcao = Percepcao(
            classe_visual=previsao.classe, confianca_visual=previsao.confianca,
            segunda_classe=previsao.segunda_classe, segunda_confianca=previsao.segunda_confianca,
            intencao=intencao, confianca_intencao=conf_i,
            materiais_texto=entidades["materiais"], condicoes=entidades["condicoes"],
            problemas_imagem=qualidade.problemas)
        decisao = self.agente.decidir(percepcao)
        return Resultado(percepcao, decisao, previsao, qualidade, entidades, dist)
