"""Baixa o TrashNet (Hugging Face), extrai as imagens e cria a divisão treino/val/teste.

Uso:
    python -m src.dados.preparar_dados

Fonte do dataset: Yang & Thung (2016), TrashNet, https://github.com/garythung/trashnet
(licença MIT). Espelho no Hugging Face: garythung/trashnet.
A divisão é estratificada por classe, com semente fixa, e salva em dados/splits.csv
para que todos os experimentos usem exatamente os mesmos conjuntos.
"""
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (ARQ_SPLITS, CLASSES, DIR_DADOS, DIR_IMAGENS,
                        FRACOES_SPLIT, SEED)

EXTENSOES = {".jpg", ".jpeg", ".png"}


def baixar_e_extrair() -> None:
    if DIR_IMAGENS.exists() and any(DIR_IMAGENS.rglob("*.jpg")):
        print(f"Imagens já presentes em {DIR_IMAGENS}")
        return
    from huggingface_hub import hf_hub_download

    print("Baixando dataset-resized.zip do Hugging Face...")
    zip_path = hf_hub_download(
        repo_id="garythung/trashnet", repo_type="dataset",
        filename="dataset-resized.zip", local_dir=str(DIR_DADOS / "_download"))
    tmp = DIR_DADOS / "_extraido"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    origem = tmp / "dataset-resized"
    DIR_IMAGENS.mkdir(parents=True, exist_ok=True)
    for pasta in origem.iterdir():
        if pasta.is_dir() and pasta.name in CLASSES:
            destino = DIR_IMAGENS / pasta.name
            if not destino.exists():
                pasta.rename(destino)
    print(f"Imagens extraídas em {DIR_IMAGENS}")


def criar_splits() -> pd.DataFrame:
    linhas = []
    for classe in CLASSES:
        for arq in sorted((DIR_IMAGENS / classe).iterdir()):
            if arq.suffix.lower() in EXTENSOES and not arq.name.startswith("."):
                linhas.append({"arquivo": f"{classe}/{arq.name}", "classe": classe})
    df = pd.DataFrame(linhas)
    f_tr, f_va, f_te = FRACOES_SPLIT
    treino, resto = train_test_split(
        df, test_size=1 - f_tr, stratify=df["classe"], random_state=SEED)
    val, teste = train_test_split(
        resto, test_size=f_te / (f_va + f_te), stratify=resto["classe"], random_state=SEED)
    df["split"] = ""
    df.loc[treino.index, "split"] = "treino"
    df.loc[val.index, "split"] = "val"
    df.loc[teste.index, "split"] = "teste"
    df.to_csv(ARQ_SPLITS, index=False)
    print(df.groupby(["split", "classe"]).size().unstack(fill_value=0))
    return df


if __name__ == "__main__":
    baixar_e_extrair()
    criar_splits()
