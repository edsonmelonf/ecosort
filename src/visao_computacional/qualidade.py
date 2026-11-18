"""Extração de características simples de qualidade da imagem (brilho, nitidez, resolução).

Essas características são usadas pelo módulo de decisão: uma foto escura ou desfocada
torna a previsão da CNN menos confiável, e o agente pode pedir uma nova foto.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

LIMIAR_ESCURA = 50.0      # brilho médio (0-255) abaixo disso => foto escura
LIMIAR_CLARA = 220.0      # acima disso => estourada
LIMIAR_NITIDEZ = 20.0     # variância do laplaciano abaixo disso => desfocada
# (calibrado no conjunto de VALIDAÇÃO do TrashNet: abaixo de 20, entre as fotos em que a CNN
#  também está incerta, a acurácia cai para ~53%, contra ~87% no geral)
RESOLUCAO_MINIMA = 100    # lado menor em pixels


@dataclass
class QualidadeImagem:
    brilho: float
    nitidez: float
    largura: int
    altura: int
    problemas: list

    @property
    def boa(self) -> bool:
        return not self.problemas


def _variancia_laplaciano(cinza: np.ndarray) -> float:
    """Variância do operador laplaciano (medida clássica de foco), só com NumPy."""
    c = cinza.astype(np.float32)
    lap = (-4 * c[1:-1, 1:-1] + c[:-2, 1:-1] + c[2:, 1:-1] + c[1:-1, :-2] + c[1:-1, 2:])
    return float(lap.var())


def analisar_qualidade(imagem: Image.Image) -> QualidadeImagem:
    cinza = np.asarray(imagem.convert("L"))
    brilho = float(cinza.mean())
    nitidez = _variancia_laplaciano(cinza)
    larg, alt = imagem.size
    problemas = []
    if brilho < LIMIAR_ESCURA:
        problemas.append("foto muito escura")
    elif brilho > LIMIAR_CLARA:
        problemas.append("foto muito clara (estourada)")
    if nitidez < LIMIAR_NITIDEZ:
        problemas.append("foto desfocada")
    if min(larg, alt) < RESOLUCAO_MINIMA:
        problemas.append("resolução muito baixa")
    return QualidadeImagem(brilho, nitidez, larg, alt, problemas)
