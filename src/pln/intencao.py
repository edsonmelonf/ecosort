"""Reconhecimento de intenção da pergunta (classificação de textos supervisionada).

Modelo: TF-IDF (palavras + n-gramas de caracteres, robusto a erros de digitação)
seguido de Regressão Logística. Escolhido por funcionar bem com poucos exemplos,
rodar em CPU e ser totalmente explicável.
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import ARQ_INTENCAO, SEED
from src.pln.dados_intencoes import TREINO
from src.pln.preprocessamento import normalizar


def construir_pipeline(classificador=None) -> Pipeline:
    atributos = FeatureUnion([
        ("palavras", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                       ngram_range=(2, 5), sublinear_tf=True)),
    ])
    clf = classificador or LogisticRegression(C=10, max_iter=2000, random_state=SEED)
    return Pipeline([("atributos", atributos), ("clf", clf)])


def textos_e_rotulos(corpus: dict):
    textos, rotulos = [], []
    for intencao, frases in corpus.items():
        textos += frases
        rotulos += [intencao] * len(frases)
    return textos, rotulos


class ReconhecedorIntencao:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def treinar(cls, corpus=None):
        textos, rotulos = textos_e_rotulos(corpus or TREINO)
        pipe = construir_pipeline().fit(textos, rotulos)
        return cls(pipe)

    @classmethod
    def carregar(cls, caminho=ARQ_INTENCAO):
        return cls(joblib.load(caminho))

    def salvar(self, caminho=ARQ_INTENCAO):
        joblib.dump(self.pipeline, caminho)

    def prever(self, texto: str):
        """Retorna (intenção, confiança, distribuição completa)."""
        probs = self.pipeline.predict_proba([texto])[0]
        classes = self.pipeline.classes_
        i = int(np.argmax(probs))
        return str(classes[i]), float(probs[i]), {str(c): float(p) for c, p in zip(classes, probs)}
