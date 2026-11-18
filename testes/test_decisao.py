from src.decisao.agente import AgenteDescarte, Percepcao


def perc(**kw):
    base = dict(classe_visual="plastic", confianca_visual=0.95, segunda_classe="glass",
                segunda_confianca=0.02, intencao="cor_lixeira", confianca_intencao=0.9,
                materiais_texto=[], condicoes=[], problemas_imagem=[])
    base.update(kw)
    return Percepcao(**base)


def test_alta_confianca_responde_com_cor_correta():
    d = AgenteDescarte().decidir(perc())
    assert d.acao == "responder" and "vermelha" in d.resposta


def test_baixa_confianca_pede_nova_foto():
    d = AgenteDescarte().decidir(perc(confianca_visual=0.30, segunda_confianca=0.25))
    assert d.acao == "pedir_nova_foto" and d.classe_final is None


def test_foto_ruim_e_confianca_media_pede_nova_foto():
    d = AgenteDescarte().decidir(perc(confianca_visual=0.7, problemas_imagem=["foto desfocada"]))
    assert d.acao == "pedir_nova_foto"


def test_ambiguidade_pede_confirmacao():
    d = AgenteDescarte().decidir(perc(confianca_visual=0.55, segunda_confianca=0.40))
    assert d.acao == "confirmar_material" and "dúvida" in d.resposta


def test_texto_prevalece_quando_imagem_incerta():
    d = AgenteDescarte().decidir(perc(confianca_visual=0.6, segunda_confianca=0.1,
                                      materiais_texto=["glass"]))
    assert d.classe_final == "glass" and d.fonte_material == "texto"


def test_imagem_prevalece_quando_muito_confiante():
    d = AgenteDescarte().decidir(perc(confianca_visual=0.95, materiais_texto=["glass"]))
    assert d.classe_final == "plastic" and d.acao == "responder_com_ressalva"


def test_papel_engordurado_nao_e_reciclavel():
    d = AgenteDescarte().decidir(perc(classe_visual="paper", intencao="pode_reciclar",
                                      condicoes=["gordura"]))
    assert d.resposta.startswith("Não")


def test_vidro_quebrado_instrucao_de_seguranca():
    d = AgenteDescarte().decidir(perc(classe_visual="glass", intencao="preparar_antes",
                                      condicoes=["quebrado"]))
    assert "jornal" in d.resposta


def test_intencao_incerta_pede_reformulacao():
    d = AgenteDescarte().decidir(perc(confianca_intencao=0.2))
    assert d.acao == "esclarecer_pergunta"


def test_rastro_registra_regras():
    d = AgenteDescarte().decidir(perc())
    assert any(r.startswith("R") for r in d.rastro)
