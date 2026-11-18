"""Base de conhecimento de descarte, por material.

Fonte das cores: Resolução CONAMA nº 275/2001 (código de cores da coleta seletiva):
azul = papel/papelão, vermelho = plástico, verde = vidro, amarelo = metal, cinza = resíduo
geral não reciclável, misto ou contaminado. As orientações de preparo seguem as práticas
usuais de programas de coleta seletiva; REGRAS LOCAIS PODEM DIFERIR. A equipe deve validar
o conteúdo com o órgão municipal de limpeza urbana e preencher os pontos de coleta locais em
`dados/pontos_coleta.json` (o sistema usa esse arquivo quando houver itens).
"""
import json
from pathlib import Path

from src.config import DIR_DADOS

BASE = {
    "cardboard": {
        "nome": "papelão",
        "reciclavel": True,
        "lixeira": "azul",
        "preparo": ["desmonte e achate a caixa para ocupar menos espaço",
                    "mantenha o papelão seco e sem restos de comida",
                    "remova fitas adesivas grandes, se possível"],
        "nao_reciclar_quando": ["gordura", "molhado"],
        "observacao": "Papelão engordurado ou molhado perde as fibras úteis e contamina o lote.",
    },
    "glass": {
        "nome": "vidro",
        "reciclavel": True,
        "lixeira": "verde",
        "preparo": ["esvazie e enxágue a embalagem",
                    "retire tampas e rolhas (elas têm outro destino)",
                    "não precisa remover o rótulo de papel"],
        "nao_reciclar_quando": [],
        "observacao": ("Espelhos, vidros de janela, cerâmica e louça costumam NÃO ser aceitos "
                       "junto com embalagens de vidro."),
    },
    "metal": {
        "nome": "metal",
        "reciclavel": True,
        "lixeira": "amarela",
        "preparo": ["esvazie a lata e enxágue rapidamente",
                    "amasse, se possível, para economizar espaço",
                    "latas de aerossol só devem ir vazias e sem pressão"],
        "nao_reciclar_quando": [],
        "observacao": "Latas de alumínio e aço são muito valorizadas pelas cooperativas.",
    },
    "paper": {
        "nome": "papel",
        "reciclavel": True,
        "lixeira": "azul",
        "preparo": ["mantenha o papel seco e limpo",
                    "não precisa remover grampos pequenos",
                    "evite amassar em bolinhas muito compactas"],
        "nao_reciclar_quando": ["gordura", "molhado", "sujo"],
        "observacao": ("Papel higiênico, guardanapo usado, papel engordurado, fita adesiva, "
                       "papel carbono e fotografias não são recicláveis."),
    },
    "plastic": {
        "nome": "plástico",
        "reciclavel": True,
        "lixeira": "vermelha",
        "preparo": ["esvazie e enxágue a embalagem",
                    "amasse garrafas e potes para economizar espaço",
                    "pode manter a tampa fechada na garrafa"],
        "nao_reciclar_quando": [],
        "observacao": ("Nem todo plástico é reciclável: embalagens metalizadas (salgadinho), "
                       "isopor sujo e plásticos muito pequenos costumam ser rejeitados."),
    },
    "trash": {
        "nome": "rejeito (lixo comum)",
        "reciclavel": False,
        "lixeira": "cinza",
        "preparo": ["embale bem para evitar vazamentos",
                    "não misture com recicláveis limpos"],
        "nao_reciclar_quando": [],
        "observacao": "Rejeito é o que não tem reciclagem viável hoje e vai para a coleta comum.",
    },
}

DESTINO_GERAL = ("Leve ao ponto de entrega voluntária (ecoponto), a uma cooperativa de catadores "
                 "ou use a coleta seletiva da sua rua. Consulte o órgão municipal de limpeza "
                 "urbana para confirmar dias e locais na sua cidade.")


def carregar_pontos_locais() -> list:
    """Lê dados/pontos_coleta.json (lista de {"nome","endereco","materiais"})."""
    arq = Path(DIR_DADOS) / "pontos_coleta.json"
    if not arq.exists():
        return []
    with open(arq, encoding="utf-8") as f:
        return json.load(f)
