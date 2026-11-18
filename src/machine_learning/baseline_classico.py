"""Baseline de ML clássico (sem deep learning) para comparar com a CNN.

Características extraídas à mão: histogramas de cor HSV (3 x 16 bins), histograma de
orientação de gradientes (9 bins, em grade 2x2) e estatísticas de textura (média/desvio
do gradiente). Classificador: SVM com kernel RBF (padronizado). Serve como referência
experimental: o ganho da CNN sobre ele é um resultado do artigo.

Uso:  python -m src.machine_learning.baseline_classico
"""
import json

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (ARQ_BASELINE, ARQ_SPLITS, CLASSES, DIR_IMAGENS,
                        DIR_MODELOS, DIR_RESULTADOS, SEED)


def extrair_caracteristicas(imagem: Image.Image) -> np.ndarray:
    img = imagem.convert("RGB").resize((128, 128))
    hsv = np.asarray(img.convert("HSV")).astype(np.float32)
    feats = []
    for canal in range(3):
        h, _ = np.histogram(hsv[..., canal], bins=16, range=(0, 255), density=True)
        feats.append(h)
    cinza = np.asarray(img.convert("L")).astype(np.float32)
    gy, gx = np.gradient(cinza)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    for i in range(2):
        for j in range(2):
            sl = (slice(i * 64, (i + 1) * 64), slice(j * 64, (j + 1) * 64))
            h, _ = np.histogram(ang[sl], bins=9, range=(0, np.pi), weights=mag[sl])
            feats.append(h / (h.sum() + 1e-8))
    feats.append(np.array([mag.mean(), mag.std(), cinza.mean(), cinza.std()]))
    return np.concatenate(feats)


def matriz(df: pd.DataFrame):
    X = np.stack([extrair_caracteristicas(Image.open(DIR_IMAGENS / a)) for a in df["arquivo"]])
    y = np.array([CLASSES.index(c) for c in df["classe"]])
    return X, y


def main():
    df = pd.read_csv(ARQ_SPLITS)
    X_tr, y_tr = matriz(df[df.split == "treino"])
    X_te, y_te = matriz(df[df.split == "teste"])
    modelo = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=SEED))
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    res = {"acuracia_teste": float(accuracy_score(y_te, pred)),
           "f1_macro_teste": float(f1_score(y_te, pred, average="macro")),
           "n_caracteristicas": int(X_tr.shape[1])}
    DIR_MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo, ARQ_BASELINE)
    with open(DIR_RESULTADOS / "metricas_baseline.json", "w") as f:
        json.dump(res, f, indent=2)
    print(res)


if __name__ == "__main__":
    main()
