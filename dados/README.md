# Dados
- `splits.csv`: divisão estratificada treino/val/teste (semente 42), gerada por `python -m src.dados.preparar_dados`.
- `trashnet/`: imagens (não versionadas; baixadas automaticamente do Hugging Face, `garythung/trashnet`, licença MIT).
- `pontos_coleta.json`: lista de pontos de coleta locais usada pelo agente. Preencha com ecopontos e cooperativas de Maceió, validados com a prefeitura. Formato:
  `[{"nome": "...", "endereco": "...", "materiais": ["plastic", "glass"]}]`
