"""Configurações centrais do EcoSort (caminhos, classes e hiperparâmetros)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "dados"
DIR_IMAGENS = DIR_DADOS / "trashnet"          # dataset-resized extraído aqui
ARQ_SPLITS = DIR_DADOS / "splits.csv"
DIR_MODELOS = RAIZ / "modelos"
ARQ_CNN = DIR_MODELOS / "cnn_residuos.pt"
ARQ_INTENCAO = DIR_MODELOS / "intencao.joblib"
ARQ_BASELINE = DIR_MODELOS / "baseline_classico.joblib"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_GRAFICOS = DIR_RESULTADOS / "graficos"
DIR_EXEMPLOS = DIR_RESULTADOS / "exemplos"

SEED = 42

# Classes originais do TrashNet (nomes das pastas) e rótulos em português.
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NOMES_PT = {
    "cardboard": "papelão",
    "glass": "vidro",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plástico",
    "trash": "rejeito (lixo comum)",
}

# Visão computacional / treino
TAMANHO_IMG = 224
MODELO_TIMM = "mobilenetv3_large_100"
PESOS_HF = "hf_hub:timm/mobilenetv3_large_100.ra_in1k"
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)
FRACOES_SPLIT = (0.70, 0.15, 0.15)  # treino / validação / teste

# Decisão (heurísticas de confiança) — justificadas e avaliadas em resultados/
LIMIAR_REJEICAO = 0.50   # abaixo disso o agente pede nova foto
LIMIAR_ALTA = 0.80       # acima disso o agente confia plenamente na imagem
MARGEM_AMBIGUA = 0.20    # diferença top1-top2 abaixo disso => ambíguo
