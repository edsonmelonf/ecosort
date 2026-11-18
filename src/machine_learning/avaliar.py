"""Avaliação final no conjunto de TESTE: CNN, baseline, PLN, agente e fluxo integrado.

Uso:  python -m src.machine_learning.avaliar
Gera resultados/metricas_cnn.json, metricas_integracao.json e figuras em resultados/graficos/.
Todos os números do artigo vêm daqui (nada é digitado à mão).
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import (ARQ_BASELINE, ARQ_SPLITS, CLASSES, DIR_EXEMPLOS,
                        DIR_GRAFICOS, DIR_IMAGENS, DIR_RESULTADOS, LIMIAR_ALTA,
                        LIMIAR_REJEICAO, NOMES_PT)
from src.decisao.agente import AgenteDescarte, Percepcao
from src.machine_learning.baseline_classico import extrair_caracteristicas
from src.pipeline import AssistenteResiduos
from src.pln.dados_intencoes import INTENCOES, TESTE
from src.pln.intencao import ReconhecedorIntencao, textos_e_rotulos
from src.visao_computacional.classificador import ClassificadorResiduos

NOMES = [NOMES_PT[c].split(" ")[0] for c in CLASSES]


def matriz_confusao(cm, rotulos, titulo, arq):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(rotulos)), rotulos, rotation=45, ha="right")
    ax.set_yticks(range(len(rotulos)), rotulos)
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(DIR_GRAFICOS / arq, dpi=200)
    plt.close(fig)


def prever_teste(clf: ClassificadorResiduos, df_te: pd.DataFrame):
    probs = []
    for a in df_te["arquivo"]:
        img = Image.open(DIR_IMAGENS / a).convert("RGB")
        x = clf.transformacao(img).unsqueeze(0)
        probs.append(clf.probabilidades_lote(x)[0].numpy())
    return np.stack(probs)


def main():
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    DIR_EXEMPLOS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ARQ_SPLITS)
    df_te = df[df.split == "teste"].reset_index(drop=True)
    y = np.array([CLASSES.index(c) for c in df_te["classe"]])

    # ---------------- CNN ----------------
    clf = ClassificadorResiduos()
    probs = prever_teste(clf, df_te)
    pred = probs.argmax(1)
    conf = probs.max(1)
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    rel = classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred)
    matriz_confusao(cm, NOMES, f"CNN (teste): acc = {acc:.3f}", "matriz_confusao_cnn.png")

    # curvas de treino
    with open(DIR_RESULTADOS / "historico_treino.json") as f:
        hist = pd.DataFrame(json.load(f))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(hist.epoca, hist.perda_treino, label="treino")
    ax[0].plot(hist.epoca, hist.perda_val, label="validação")
    ax[0].set_xlabel("época"); ax[0].set_ylabel("perda"); ax[0].legend(); ax[0].set_title("Perda")
    ax[1].plot(hist.epoca, hist.acc_treino, label="treino")
    ax[1].plot(hist.epoca, hist.acc_val, label="validação")
    n_cabeca = int((hist.fase == "cabeca").sum())
    for a in ax:
        a.axvline(n_cabeca + 0.5, color="gray", ls="--", lw=0.8)
    ax[1].set_xlabel("época"); ax[1].set_ylabel("acurácia"); ax[1].legend(); ax[1].set_title("Acurácia")
    fig.tight_layout(); fig.savefig(DIR_GRAFICOS / "curvas_treino.png", dpi=200); plt.close(fig)

    # baseline clássico
    base = joblib.load(ARQ_BASELINE)
    Xb = np.stack([extrair_caracteristicas(Image.open(DIR_IMAGENS / a)) for a in df_te["arquivo"]])
    pred_b = base.predict(Xb)
    acc_b = accuracy_score(y, pred_b)
    f1_b = f1_score(y, pred_b, average="macro")

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    barras = ax.bar(["SVM + características\nmanuais", "CNN (MobileNetV3)\nfine-tuning"],
                    [acc_b, acc], color=["#999999", "#2a7"])
    for b_ in barras:
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.01, f"{b_.get_height():.3f}",
                ha="center")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Acurácia no teste")
    fig.tight_layout(); fig.savefig(DIR_GRAFICOS / "comparacao_modelos.png", dpi=200); plt.close(fig)

    # cobertura x acurácia (justifica os limiares do agente)
    limiares = np.linspace(0.3, 0.99, 50)
    cob = [(conf >= t).mean() for t in limiares]
    acc_sel = [accuracy_score(y[conf >= t], pred[conf >= t]) if (conf >= t).any() else np.nan
               for t in limiares]
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.plot(limiares, acc_sel, label="acurácia nas respostas dadas")
    ax.plot(limiares, cob, label="cobertura (fração respondida)")
    for t, ls in ((LIMIAR_REJEICAO, ":"), (LIMIAR_ALTA, "--")):
        ax.axvline(t, color="gray", ls=ls, lw=1)
    ax.set_xlabel("limiar de confiança"); ax.legend(); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(DIR_GRAFICOS / "cobertura_acuracia.png", dpi=200); plt.close(fig)

    erradas_alta = int(((pred != y) & (conf >= LIMIAR_ALTA)).sum())
    erradas_baixa = int(((pred != y) & (conf < LIMIAR_ALTA)).sum())
    metricas_cnn = {
        "n_teste": int(len(y)), "acuracia": float(acc), "f1_macro": float(f1m),
        "por_classe": {c: {k: float(rel[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                       for c in CLASSES},
        "baseline_svm": {"acuracia": float(acc_b), "f1_macro": float(f1_b)},
        "confianca_media_acertos": float(conf[pred == y].mean()),
        "confianca_media_erros": float(conf[pred != y].mean()),
        "erros_com_confianca_alta": erradas_alta, "erros_com_confianca_baixa": erradas_baixa,
        "acuracia_conf_ge_alta": float(accuracy_score(y[conf >= LIMIAR_ALTA], pred[conf >= LIMIAR_ALTA])),
        "cobertura_conf_ge_alta": float((conf >= LIMIAR_ALTA).mean()),
    }
    with open(DIR_RESULTADOS / "metricas_cnn.json", "w", encoding="utf-8") as f:
        json.dump(metricas_cnn, f, indent=2, ensure_ascii=False)

    # ---------------- PLN ----------------
    intencao = ReconhecedorIntencao.carregar()
    x_te, y_te = textos_e_rotulos(TESTE)
    pred_i = intencao.pipeline.predict(x_te)
    cm_i = confusion_matrix(y_te, pred_i, labels=INTENCOES)
    rot_i = [i.replace("_", "\n") for i in INTENCOES]
    matriz_confusao(cm_i, rot_i, f"Intenção (teste): acc = {accuracy_score(y_te, pred_i):.3f}",
                    "matriz_confusao_intencao.png")

    # ---------------- Fluxo integrado ----------------
    assistente = AssistenteResiduos(classificador=clf, intencao=intencao)
    oraculo = AgenteDescarte()
    frases = [(f, i) for i, fs in TESTE.items() for f in fs]
    rng = np.random.default_rng(7)
    linhas = []
    for k, row in df_te.iterrows():
        pergunta, intencao_real = frases[int(rng.integers(len(frases)))]
        img = Image.open(DIR_IMAGENS / row["arquivo"]).convert("RGB")
        r = assistente.responder(img, pergunta)
        esperado = oraculo.decidir(Percepcao(
            classe_visual=row["classe"], confianca_visual=1.0, segunda_classe=row["classe"],
            segunda_confianca=0.0, intencao=intencao_real, confianca_intencao=1.0,
            materiais_texto=[], condicoes=[], problemas_imagem=[])).resposta
        d = r.decisao
        respondeu = d.acao != "pedir_nova_foto"
        # Correta = agente respondeu com a classe certa e a intenção certa
        correta = respondeu and d.classe_final == row["classe"] and r.percepcao.intencao == intencao_real
        linhas.append({"arquivo": row["arquivo"], "classe_real": row["classe"],
                       "classe_prev": r.previsao_visual.classe, "pergunta": pergunta,
                       "intencao_real": intencao_real, "intencao_prev": r.percepcao.intencao,
                       "acao": d.acao, "correta": bool(correta), "respondeu": bool(respondeu),
                       "resposta": d.resposta, "resposta_esperada": esperado})
    ie = pd.DataFrame(linhas)
    ie.to_csv(DIR_RESULTADOS / "avaliacao_integrada.csv", index=False)
    resp = ie[ie.respondeu]
    sem_agente = float(((ie.classe_prev == ie.classe_real) & (ie.intencao_prev == ie.intencao_real)).mean())
    integ = {
        "n": int(len(ie)),
        "fluxo_correto_sobre_total": float(ie.correta.mean()),
        "taxa_pedido_nova_foto": float(1 - ie.respondeu.mean()),
        "fluxo_correto_sobre_respondidas": float(resp.correta.mean()),
        "sem_agente_visao_e_intencao_corretas": sem_agente,
        # mesmo denominador do agente (só os casos em que ele respondeu)
        "sem_agente_sobre_respondidas": float(((resp.classe_prev == resp.classe_real)
                                               & (resp.intencao_prev == resp.intencao_real)).mean()),
        "acoes": ie.acao.value_counts().to_dict(),
        "resposta_igual_a_esperada_sobre_respondidas": float((resp.resposta == resp.resposta_esperada).mean()),
        "acuracia_intencao_teste": float(accuracy_score(y_te, pred_i)),
    }
    with open(DIR_RESULTADOS / "metricas_integracao.json", "w", encoding="utf-8") as f:
        json.dump(integ, f, indent=2, ensure_ascii=False)

    # ---------------- Exemplos entrada -> saída ----------------
    exemplos = []
    escolha = (ie[ie.correta].groupby("classe_real").head(1).head(4).index.tolist()
               + ie[~ie.correta].head(2).index.tolist())
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.6))
    for ax, idx in zip(axes.ravel(), escolha):
        row = ie.loc[idx]
        ax.imshow(Image.open(DIR_IMAGENS / row.arquivo)); ax.axis("off")
        ax.set_title(f"real: {NOMES_PT[row.classe_real].split(' ')[0]} | "
                     f"prev: {NOMES_PT[row.classe_prev].split(' ')[0]}", fontsize=9)
        ax.text(0.5, -0.03, f"\"{row.pergunta}\"\n→ {row.acao}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5)
        exemplos.append({k: (v if not isinstance(v, (np.bool_,)) else bool(v)) for k, v in row.items()})
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.subplots_adjust(hspace=0.35)
    fig.savefig(DIR_EXEMPLOS / "exemplos_entrada_saida.png", dpi=170); plt.close(fig)
    with open(DIR_EXEMPLOS / "exemplos_entrada_saida.json", "w", encoding="utf-8") as f:
        json.dump(exemplos, f, indent=2, ensure_ascii=False)

    print(json.dumps({"cnn": {k: metricas_cnn[k] for k in
                              ("acuracia", "f1_macro", "erros_com_confianca_alta",
                               "acuracia_conf_ge_alta", "cobertura_conf_ge_alta")},
                      "baseline": metricas_cnn["baseline_svm"], "integracao": integ},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
