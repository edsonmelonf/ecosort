"""Classificador visual de resíduos: MobileNetV3 (timm) com fine-tuning no TrashNet."""
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from src.config import (ARQ_CNN, CLASSES, DESVIO_IMAGENET, MEDIA_IMAGENET,
                        MODELO_TIMM, TAMANHO_IMG)


def transformacao_avaliacao():
    return transforms.Compose([
        transforms.Resize((TAMANHO_IMG, TAMANHO_IMG)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


def transformacao_treino():
    return transforms.Compose([
        transforms.RandomResizedCrop(TAMANHO_IMG, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


def criar_modelo(pretreinado: bool):
    """Cria a MobileNetV3. Com pretreinado=True baixa os pesos ImageNet do Hugging Face."""
    import timm
    if pretreinado:
        from src.config import PESOS_HF
        modelo = timm.create_model(PESOS_HF, pretrained=True, num_classes=len(CLASSES))
    else:
        modelo = timm.create_model(MODELO_TIMM, pretrained=False, num_classes=len(CLASSES))
    return modelo


@dataclass
class PrevisaoVisual:
    classe: str
    confianca: float
    segunda_classe: str
    segunda_confianca: float
    probabilidades: dict

    @property
    def margem(self) -> float:
        return self.confianca - self.segunda_confianca


class ClassificadorResiduos:
    def __init__(self, caminho_pesos=ARQ_CNN, dispositivo: Optional[str] = None):
        self.dispositivo = dispositivo or "cpu"
        self.modelo = criar_modelo(pretreinado=False)
        estado = torch.load(caminho_pesos, map_location=self.dispositivo)
        self.modelo.load_state_dict(estado)
        self.modelo.to(self.dispositivo).eval()
        self.transformacao = transformacao_avaliacao()

    @torch.no_grad()
    def probabilidades_lote(self, tensores: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.modelo(tensores.to(self.dispositivo)), dim=1).cpu()

    def prever(self, imagem: Image.Image) -> PrevisaoVisual:
        x = self.transformacao(imagem.convert("RGB")).unsqueeze(0)
        probs = self.probabilidades_lote(x)[0]
        ordem = torch.argsort(probs, descending=True)
        i1, i2 = int(ordem[0]), int(ordem[1])
        return PrevisaoVisual(
            classe=CLASSES[i1], confianca=float(probs[i1]),
            segunda_classe=CLASSES[i2], segunda_confianca=float(probs[i2]),
            probabilidades={c: float(p) for c, p in zip(CLASSES, probs)})
