# EcoSort — assistente de descarte de resíduos

Projeto Integrador de Inteligência Artificial — AFYA | UNIMA Maceió (Prof. Icaro Ferreira).

Você envia a **foto de um resíduo** e faz uma **pergunta em português** ("posso reciclar isso?", "qual a cor da lixeira?", "preciso lavar antes?"). O sistema identifica o material, entende a pergunta e responde com a orientação de descarte, explicando como chegou à resposta.

## Integrantes e responsabilidades

| Nome completo | Responsabilidade |
|---|---|
| Edson de Melo Neto Filho | [preencher] |
| Renato Berto dos Santos | [preencher] |
| Nathan Silva Melo | [preencher] |
| Matheus Mendes | [preencher] |

Artigo (Overleaf): [colar o link aqui]

## Arquitetura

```
 Foto ──► qualidade da imagem + CNN (MobileNetV3) ──► classe + confiança ──┐
                                                                            ├─► Agente de decisão ─► resposta + rastro das regras
 Pergunta ─► intenção (TF-IDF + LogReg) + entidades (material, condição) ──┘
```

| Pasta | Conteúdo |
|---|---|
| `src/visao_computacional/` | Classificador (CNN) e análise de qualidade da foto (brilho, nitidez) |
| `src/pln/` | Normalização, reconhecimento de intenção, extração de palavras-chave, corpus |
| `src/machine_learning/` | Treino da CNN, baseline clássico (SVM) e avaliação final |
| `src/decisao/` | Base de conhecimento de descarte e agente baseado em regras |
| `src/pipeline.py` | Fluxo integrado imagem + pergunta → resposta |
| `app/streamlit_app.py` | Interface web |
| `testes/` | Testes automatizados (pytest) |
| `resultados/` | Gráficos, métricas (JSON) e exemplos de entrada/saída, todos gerados pelo código |
| `artigo/` | Fonte LaTeX (IEEE) do artigo |
| `notebooks/` | Notebook para rodar tudo no Google Colab |

## Tecnologias

Python, PyTorch, timm (MobileNetV3), scikit-learn, pandas, matplotlib, Streamlit, pytest. Roda em CPU; não precisa de GPU nem de serviço pago.

## Como instalar e executar

```bash
git clone https://github.com/edsonmelonf/ecosort.git && cd ecosort
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.dados.preparar_dados          # baixa o TrashNet e cria a divisão treino/val/teste
python -m src.machine_learning.treino_cnn   # treina a CNN (~20 min em CPU de 2 núcleos)
python -m src.pln.treino_intencao           # treina e avalia o PLN
python -m src.machine_learning.baseline_classico
python -m src.machine_learning.avaliar      # gera métricas, gráficos e exemplos em resultados/
python -m pytest testes -q                  # testes

streamlit run app/streamlit_app.py          # interface
```

O repositório já inclui `modelos/cnn_residuos.pt` e `modelos/intencao.joblib`, então dá para pular os treinos e rodar só o app. No Google Colab, abra `notebooks/ecosort_colab.ipynb`.

## Modelos

- **Visão:** MobileNetV3-Large pré-treinada no ImageNet, com a cabeça trocada por 6 saídas. Treino em 2 fases (3 épocas só a cabeça, 8 de ajuste fino), AdamW, aumento de dados e *label smoothing*. Modelo escolhido pela acurácia de validação.
- **Baseline:** SVM RBF sobre 88 características manuais (histogramas HSV, gradientes e textura).
- **PLN:** TF-IDF (palavras e n-gramas de caracteres) + Regressão Logística, 6 intenções. Extração de materiais e condições por dicionário.
- **Decisão:** agente reativo baseado em regras com limiares de confiança (regras R1 a R9, descritas em `src/decisao/agente.py`). Cada resposta traz o rastro das regras acionadas.

## Resultados (conjunto de teste, gerados por `avaliar.py`)

| Métrica | Valor |
|---|---|
| CNN: acurácia / F1 macro (380 imagens) | 87,4% / 0,850 |
| Baseline SVM: acurácia / F1 macro | 72,4% / 0,704 |
| CNN com confiança ≥ 0,80: acurácia e cobertura | 95,7% e 73,4% |
| PLN: acurácia em 48 frases inéditas | 93,8% |
| Fluxo integrado: acerto entre as respostas dadas | 82,1% |
| Fluxo integrado: pedidos de nova foto | 8,9% |
| Sem o agente (classe top-1 + intenção prevista), nos mesmos 346 casos respondidos | 82,1% |
| Acerto sobre os 380 casos: com o agente / sem o agente | 74,7% / 80,0% |

Gráficos em `resultados/graficos/` e exemplos em `resultados/exemplos/`.

## Limitações

- O TrashNet tem fundo claro e um objeto por foto. Em fotos do dia a dia o desempenho deve ser pior, e isso **não foi medido**.
- A classe "rejeito" tem poucas imagens (20 no teste) e o F1 dela é o mais baixo (0,667).
- O corpus de PLN é pequeno e escrito pela própria equipe, então os 93,8% provavelmente são otimistas. A negação ("não está limpo") não é tratada.
- O agente **não melhora a acurácia média**: nos mesmos 346 casos em que responde, a versão sem agente também acerta 82,1%; sobre os 380 casos, o acerto cai de 80,0% (sem agente) para 74,7%, porque dos 34 pedidos de nova foto a versão sem agente teria acertado 20 e errado 14. O valor dele está em adaptar a orientação à condição citada (papel engordurado, vidro quebrado), arbitrar entre texto e imagem e explicar a decisão.
- A base de conhecimento usa o código de cores da Resolução CONAMA 275/2001 e práticas gerais. **Regras e pontos de coleta de Maceió precisam ser validados com o órgão municipal** e cadastrados em `dados/pontos_coleta.json`.

## Referências

- M. Yang e G. Thung, *Classification of Trash for Recyclability Status* (TrashNet), 2016 — https://github.com/garythung/trashnet
- A. Howard et al., *Searching for MobileNetV3*, ICCV 2019.
- R. Wightman, *PyTorch Image Models (timm)* — https://github.com/huggingface/pytorch-image-models
- F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011.
- CONAMA, Resolução nº 275/2001 (código de cores da coleta seletiva).

Lista completa no artigo (`artigo/main.tex`).
