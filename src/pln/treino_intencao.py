"""Treina o reconhecedor de intenção, compara modelos e avalia no corpus de teste.

Uso:
    python -m src.pln.treino_intencao
Saídas: modelos/intencao.joblib e resultados/metricas_pln.json
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import DIR_MODELOS, DIR_RESULTADOS, SEED
from src.pln.dados_intencoes import INTENCOES, TESTE, TREINO
from src.pln.intencao import (ReconhecedorIntencao, construir_pipeline,
                              textos_e_rotulos)


def main():
    DIR_MODELOS.mkdir(exist_ok=True)
    DIR_RESULTADOS.mkdir(exist_ok=True)
    x_tr, y_tr = textos_e_rotulos(TREINO)
    x_te, y_te = textos_e_rotulos(TESTE)

    candidatos = {
        "LogReg (escolhido)": LogisticRegression(C=10, max_iter=2000, random_state=SEED),
        "LinearSVC": None,
        "NaiveBayes": MultinomialNB(),
    }
    resultados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for nome in candidatos:
        if nome == "LinearSVC":
            clf = LinearSVC(C=1.0, random_state=SEED)
        else:
            clf = candidatos[nome]
        pipe = construir_pipeline(clf)
        cv_acc = cross_val_score(pipe, x_tr, y_tr, cv=cv, scoring="accuracy")
        pipe.fit(x_tr, y_tr)
        pred = pipe.predict(x_te)
        resultados[nome] = {
            "cv5_acc_media": float(cv_acc.mean()), "cv5_acc_desvio": float(cv_acc.std()),
            "teste_acc": float(accuracy_score(y_te, pred)),
            "teste_f1_macro": float(f1_score(y_te, pred, average="macro")),
        }
        print(f"{nome:20s} CV5 {cv_acc.mean():.3f}±{cv_acc.std():.3f}  "
              f"teste acc {resultados[nome]['teste_acc']:.3f}  F1 {resultados[nome]['teste_f1_macro']:.3f}")

    modelo = ReconhecedorIntencao.treinar()
    modelo.salvar()
    pred = modelo.pipeline.predict(x_te)
    relatorio = classification_report(y_te, pred, labels=INTENCOES, output_dict=True, zero_division=0)
    erros = [{"frase": f, "real": r, "previsto": p}
             for f, r, p in zip(x_te, y_te, pred) if r != p]
    saida = {"n_treino": len(x_tr), "n_teste": len(x_te), "comparacao": resultados,
             "relatorio_teste": relatorio, "erros_teste": erros}
    with open(DIR_RESULTADOS / "metricas_pln.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"\nModelo salvo. Erros no teste: {len(erros)}/{len(x_te)}")
    for e in erros:
        print("  ", e)


if __name__ == "__main__":
    main()
