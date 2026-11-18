"""Interface web do EcoSort.  Execute:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image

from src.config import CLASSES, NOMES_PT
from src.pipeline import AssistenteResiduos

st.set_page_config(page_title="EcoSort", page_icon="♻️", layout="wide")


@st.cache_resource
def carregar_assistente():
    return AssistenteResiduos()


st.title("♻️ EcoSort — assistente de descarte de resíduos")
st.caption("Envie uma foto do resíduo e pergunte como descartá-lo. "
           "Visão computacional + PLN + agente de decisão.")

col_a, col_b = st.columns([1, 1])
with col_a:
    arquivo = st.file_uploader("Foto do resíduo", type=["jpg", "jpeg", "png"])
    pergunta = st.text_input("Sua pergunta", value="Posso reciclar isso?",
                             help="Ex.: qual a cor da lixeira? preciso lavar antes? onde descartar?")
    if arquivo:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, use_container_width=True)

with col_b:
    if arquivo and pergunta.strip():
        r = carregar_assistente().responder(imagem, pergunta)
        d = r.decisao
        if d.acao == "pedir_nova_foto":
            st.warning(d.resposta)
        elif d.acao in ("confirmar_material", "responder_com_ressalva", "esclarecer_pergunta"):
            st.info(d.resposta)
        else:
            st.success(d.resposta)

        with st.expander("Como o sistema chegou nessa resposta", expanded=True):
            st.markdown(f"**Ação do agente:** `{d.acao}` · **Fonte do material:** `{d.fonte_material}`")
            st.markdown(f"**Visão:** {NOMES_PT[r.previsao_visual.classe]} "
                        f"({r.previsao_visual.confianca:.0%}); 2ª hipótese: "
                        f"{NOMES_PT[r.previsao_visual.segunda_classe]} "
                        f"({r.previsao_visual.segunda_confianca:.0%})")
            st.markdown(f"**PLN:** intenção `{r.percepcao.intencao}` "
                        f"({r.percepcao.confianca_intencao:.0%}); entidades: `{r.entidades}`")
            st.markdown(f"**Qualidade da foto:** brilho {r.qualidade.brilho:.0f}, nitidez "
                        f"{r.qualidade.nitidez:.0f}"
                        + (f" — problemas: {', '.join(r.qualidade.problemas)}" if r.qualidade.problemas else " — ok"))
            st.markdown("**Regras acionadas:**")
            for passo in d.rastro:
                st.markdown(f"- {passo}")
            st.bar_chart({NOMES_PT[c]: r.previsao_visual.probabilidades[c] for c in CLASSES})
    else:
        st.info("Envie uma imagem e escreva uma pergunta para começar.")
