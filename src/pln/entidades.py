"""Extração de palavras-chave: materiais e condições citados na pergunta.

Exemplos:
  "essa caixa de pizza suja pode reciclar?" -> materiais: papelão | condições: gordura
  "essa garrafa pet tá cheia de refrigerante" -> materiais: plástico | condições: resto de líquido/comida
A extração é feita por dicionário sobre o texto normalizado (sem acentos), com casamento
de palavra inteira. O resultado alimenta o módulo de decisão.
"""
import re

from src.pln.preprocessamento import normalizar

# material (classe TrashNet) -> termos que o usuário costuma usar
MATERIAIS = {
    "cardboard": ["papelao", "caixa de papelao", "caixa de pizza", "caixa de sapato",
                  "caixa de leite", "embalagem de papelao"],
    "glass": ["vidro", "garrafa de vidro", "pote de vidro", "frasco de vidro",
              "copo de vidro", "espelho", "cacos de vidro"],
    "metal": ["metal", "lata", "latinha", "aluminio", "ferro", "aco", "lata de conserva",
              "lata de refrigerante"],
    "paper": ["papel", "jornal", "revista", "folha", "caderno", "papel de escritorio"],
    "plastic": ["plastico", "pet", "garrafa pet", "sacola", "sacola plastica", "pote de plastico",
                "embalagem plastica", "copo descartavel", "canudo"],
}

# condição -> termos
CONDICOES = {
    "gordura": ["gordura", "gordurosa", "gorduroso", "engordurada", "engordurado", "oleo",
                "oleosa", "oleoso", "pizza suja"],
    "sujo": ["sujo", "suja", "sujos", "sujas", "resto de comida", "restos de comida",
             "com comida", "cheio de comida", "cheia de comida", "melado", "melada",
             "cheio de refrigerante", "cheia de refrigerante"],
    "molhado": ["molhado", "molhada", "encharcado", "encharcada", "umido", "umida"],
    "quebrado": ["quebrado", "quebrada", "cacos", "estilhacado", "estilhacada", "trincado",
                 "trincada", "cortante"],
    "limpo": ["limpo", "limpa", "lavado", "lavada", "higienizado", "higienizada"],
}


def _contem(texto_norm: str, termo: str) -> bool:
    return re.search(rf"\b{re.escape(termo)}\b", texto_norm) is not None


def extrair_entidades(texto: str) -> dict:
    t = normalizar(texto)
    materiais = [m for m, termos in MATERIAIS.items() if any(_contem(t, x) for x in termos)]
    condicoes = [c for c, termos in CONDICOES.items() if any(_contem(t, x) for x in termos)]
    # "limpo" não coexiste com "sujo" (a negação "não está limpo" não é tratada: limitação)
    if "limpo" in condicoes and ("sujo" in condicoes or "gordura" in condicoes):
        condicoes.remove("limpo")
    return {"materiais": materiais, "condicoes": condicoes}
