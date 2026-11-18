"""Treinamento da CNN por transfer learning (aprendizado supervisionado).

Estratégia em duas fases:
  1) só a cabeça (classificador) treina, com o backbone congelado;
  2) fine-tuning de toda a rede com taxa de aprendizado menor.
Seleciona o melhor modelo pela acurácia de validação (o teste nunca é usado aqui).

Uso:
    python -m src.machine_learning.treino_cnn [--epocas-cabeca 3 --epocas-total 8]
"""
import argparse
import json
import time

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import (ARQ_CNN, ARQ_SPLITS, CLASSES, DIR_IMAGENS, DIR_MODELOS,
                        DIR_RESULTADOS, SEED)
from src.visao_computacional.classificador import (criar_modelo,
                                                   transformacao_avaliacao,
                                                   transformacao_treino)


class DatasetResiduos(Dataset):
    def __init__(self, df: pd.DataFrame, transformacao):
        self.arquivos = df["arquivo"].tolist()
        self.rotulos = [CLASSES.index(c) for c in df["classe"]]
        self.transformacao = transformacao

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, i):
        img = Image.open(DIR_IMAGENS / self.arquivos[i]).convert("RGB")
        return self.transformacao(img), self.rotulos[i]


def _avaliar(modelo, loader, criterio, disp):
    modelo.eval()
    perda, acertos, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(disp), y.to(disp)
            saida = modelo(x)
            perda += criterio(saida, y).item() * len(y)
            acertos += (saida.argmax(1) == y).sum().item()
            n += len(y)
    return perda / n, acertos / n


def _epoca(modelo, loader, criterio, otimizador, disp):
    modelo.train()
    perda, acertos, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(disp), y.to(disp)
        otimizador.zero_grad()
        saida = modelo(x)
        p = criterio(saida, y)
        p.backward()
        otimizador.step()
        perda += p.item() * len(y)
        acertos += (saida.argmax(1) == y).sum().item()
        n += len(y)
    return perda / n, acertos / n


def treinar(epocas_cabeca=3, epocas_total=8, lote=32, lr_cabeca=1e-3, lr_total=2e-4):
    torch.manual_seed(SEED)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(ARQ_SPLITS)
    dl_tr = DataLoader(DatasetResiduos(df[df.split == "treino"], transformacao_treino()),
                       batch_size=lote, shuffle=True, num_workers=2)
    dl_va = DataLoader(DatasetResiduos(df[df.split == "val"], transformacao_avaliacao()),
                       batch_size=lote, num_workers=2)

    modelo = criar_modelo(pretreinado=True).to(disp)
    criterio = nn.CrossEntropyLoss(label_smoothing=0.05)
    historico, melhor_acc = [], 0.0
    DIR_MODELOS.mkdir(exist_ok=True)

    fases = [("cabeca", epocas_cabeca, lr_cabeca), ("total", epocas_total, lr_total)]
    for nome, n_ep, lr in fases:
        for nome_p, p in modelo.named_parameters():
            p.requires_grad = (nome == "total") or nome_p.startswith(("classifier", "conv_head"))
        params = [p for p in modelo.parameters() if p.requires_grad]
        otimizador = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=max(n_ep, 1))
        for ep in range(n_ep):
            t0 = time.time()
            p_tr, a_tr = _epoca(modelo, dl_tr, criterio, otimizador, disp)
            p_va, a_va = _avaliar(modelo, dl_va, criterio, disp)
            sched.step()
            historico.append({"fase": nome, "epoca": len(historico) + 1,
                              "perda_treino": p_tr, "acc_treino": a_tr,
                              "perda_val": p_va, "acc_val": a_va})
            print(f"[{nome}] ep {len(historico):02d}  treino {a_tr:.3f}  val {a_va:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if a_va > melhor_acc:
                melhor_acc = a_va
                torch.save(modelo.state_dict(), ARQ_CNN)

    DIR_RESULTADOS.mkdir(exist_ok=True)
    with open(DIR_RESULTADOS / "historico_treino.json", "w") as f:
        json.dump(historico, f, indent=2)
    print(f"Melhor acurácia de validação: {melhor_acc:.4f}  -> {ARQ_CNN}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas-cabeca", type=int, default=3)
    ap.add_argument("--epocas-total", type=int, default=8)
    a = ap.parse_args()
    treinar(a.epocas_cabeca, a.epocas_total)
