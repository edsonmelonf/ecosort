"""Agente de decisão de descarte (agente reativo baseado em regras e limiares).

Percepção : previsão visual (classe + confiança), qualidade da foto, intenção da pergunta
            e entidades extraídas do texto (materiais e condições).
Ações     : responder | responder_com_ressalva | confirmar_material | pedir_nova_foto |
            esclarecer_pergunta
Função real no sistema: o agente NÃO apenas repassa a saída da CNN. Ele arbitra entre
evidências (imagem x texto), usa a confiança como critério de incerteza, ajusta a orientação
conforme a condição citada (ex.: papel engordurado) e escolhe quando é melhor pedir outra foto
do que arriscar uma orientação errada. Cada decisão registra o rastro das regras acionadas.
"""
from dataclasses import dataclass, field
from typing import Optional

from src.config import LIMIAR_ALTA, LIMIAR_REJEICAO, MARGEM_AMBIGUA
from src.decisao.base_conhecimento import BASE, DESTINO_GERAL, carregar_pontos_locais

LIMIAR_INTENCAO = 0.35  # abaixo disso o agente não confia na intenção reconhecida

NOMES_INTENCAO = {
    "como_descartar": "como descartar",
    "pode_reciclar": "se pode reciclar",
    "cor_lixeira": "a cor da lixeira",
    "preparar_antes": "como preparar antes do descarte",
    "onde_descartar": "onde descartar",
    "identificar_material": "qual é o material",
}


@dataclass
class Percepcao:
    classe_visual: str
    confianca_visual: float
    segunda_classe: str
    segunda_confianca: float
    intencao: str
    confianca_intencao: float
    materiais_texto: list = field(default_factory=list)
    condicoes: list = field(default_factory=list)
    problemas_imagem: list = field(default_factory=list)

    @property
    def margem(self) -> float:
        return self.confianca_visual - self.segunda_confianca


@dataclass
class Decisao:
    acao: str
    classe_final: Optional[str]
    fonte_material: str
    resposta: str
    rastro: list


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


class AgenteDescarte:
    # ---------- decisão sobre o material ----------
    def _decidir_material(self, p: Percepcao, rastro: list):
        # R1: foto ruim e modelo não muito seguro -> pedir nova foto
        if p.problemas_imagem and p.confianca_visual < LIMIAR_ALTA:
            rastro.append(f"R1: imagem com problemas ({', '.join(p.problemas_imagem)}) e "
                          f"confiança {_pct(p.confianca_visual)} < {_pct(LIMIAR_ALTA)}")
            return "pedir_nova_foto", None, "imagem"

        unico_texto = p.materiais_texto[0] if len(p.materiais_texto) == 1 else None

        # R2: confiança muito baixa e sem ajuda do texto -> pedir nova foto
        if p.confianca_visual < LIMIAR_REJEICAO and unico_texto is None:
            rastro.append(f"R2: confiança {_pct(p.confianca_visual)} < {_pct(LIMIAR_REJEICAO)} "
                          "e o texto não informa o material")
            return "pedir_nova_foto", None, "imagem"

        # R3: o usuário citou um material no texto
        if unico_texto is not None:
            if unico_texto == p.classe_visual:
                rastro.append("R3a: texto e imagem concordam no material")
                return "responder", p.classe_visual, "imagem+texto"
            if p.confianca_visual >= LIMIAR_ALTA:
                rastro.append(f"R3b: texto cita {BASE[unico_texto]['nome']}, mas a imagem indica "
                              f"{BASE[p.classe_visual]['nome']} com {_pct(p.confianca_visual)} "
                              "(alta): mantém a imagem e avisa a divergência")
                return "responder_com_ressalva", p.classe_visual, "imagem"
            rastro.append(f"R3c: texto cita {BASE[unico_texto]['nome']} e a confiança visual "
                          f"({_pct(p.confianca_visual)}) é menor que {_pct(LIMIAR_ALTA)}: "
                          "adota o material informado pelo usuário")
            return "responder_com_ressalva", unico_texto, "texto"

        # R4: resultado visual ambíguo entre duas classes
        if p.margem < MARGEM_AMBIGUA and p.confianca_visual < LIMIAR_ALTA:
            rastro.append(f"R4: margem top1-top2 = {_pct(p.margem)} < {_pct(MARGEM_AMBIGUA)} "
                          "(ambíguo): responde para o melhor palpite e pede confirmação")
            return "confirmar_material", p.classe_visual, "imagem"

        rastro.append(f"R5: confiança visual {_pct(p.confianca_visual)} suficiente")
        return "responder", p.classe_visual, "imagem"

    # ---------- composição da resposta ----------
    def _reciclavel_agora(self, classe: str, condicoes: list):
        info = BASE[classe]
        if not info["reciclavel"]:
            return False, "Este material é rejeito, sem reciclagem viável."
        impeditivas = [c for c in condicoes if c in info["nao_reciclar_quando"]]
        if impeditivas:
            motivo = {"gordura": "engordurado", "molhado": "molhado", "sujo": "sujo"}[impeditivas[0]]
            return False, (f"{info['nome'].capitalize()} {motivo} não é aproveitado na reciclagem: "
                           "descarte a parte contaminada no lixo comum (cinza) e recicle só "
                           "a parte limpa e seca, se houver.")
        return True, ""

    def _corpo(self, intencao: str, classe: str, condicoes: list, rastro: list) -> str:
        info = BASE[classe]
        nome = info["nome"]
        ok, motivo = self._reciclavel_agora(classe, condicoes)
        if not ok and info["reciclavel"]:
            rastro.append(f"R6: condição {condicoes} torna {nome} não reciclável agora")
        lixeira = info["lixeira"] if ok else "cinza"
        preparo = list(info["preparo"])
        if "quebrado" in condicoes and classe == "glass":
            rastro.append("R7: vidro quebrado -> instrução de segurança")
            preparo = ["embrulhe os cacos em jornal ou papelão grosso e identifique "
                       "como \"vidro quebrado\"", "nunca coloque cacos soltos no saco",
                       "use luvas e calçado fechado ao recolher"]
        elif condicoes and "sujo" in condicoes and ok:
            rastro.append("R8: item sujo -> reforça a limpeza antes do descarte")
            preparo.insert(0, "retire os restos de comida e enxágue bem antes de descartar")

        if intencao == "identificar_material":
            frase = f"Parece ser {nome}. "
            frase += (f"É reciclável (lixeira {info['lixeira']})." if info["reciclavel"]
                      else "É rejeito (lixeira cinza).")
            return frase + " " + info["observacao"]
        if intencao == "pode_reciclar":
            if ok:
                return f"Sim, {nome} é reciclável. {info['observacao']}"
            return f"Não. {motivo}"
        if intencao == "cor_lixeira":
            extra = "" if ok else f" (nas condições descritas, o item vai para a cinza em vez da {info['lixeira']})"
            return (f"Use a lixeira {lixeira}{extra}, conforme o código de cores da "
                    "Resolução CONAMA 275/2001. A cor pode variar em alguns locais.")
        if intencao == "preparar_antes":
            if not ok and info["reciclavel"]:
                return motivo
            return "Antes de descartar: " + "; ".join(preparo) + "."
        if intencao == "onde_descartar":
            pontos = [x for x in carregar_pontos_locais()
                      if not x.get("materiais") or classe in x["materiais"]]
            if pontos:
                lista = "; ".join(f"{x['nome']} ({x.get('endereco', 'endereço não informado')})"
                                  for x in pontos[:3])
                return f"Pontos cadastrados para este material: {lista}. {DESTINO_GERAL}"
            return DESTINO_GERAL
        # como_descartar
        if ok:
            return (f"{nome.capitalize()} vai na lixeira {lixeira}. Antes: " + "; ".join(preparo)
                    + ". " + DESTINO_GERAL)
        return motivo + " " + ("Lixeira cinza (lixo comum)." if info["reciclavel"] else
                               f"Lixeira cinza. {' '.join(preparo[:1])}.".capitalize())

    def decidir(self, p: Percepcao) -> Decisao:
        rastro = []
        intencao, conf_i = p.intencao, p.confianca_intencao
        acao, classe, fonte = self._decidir_material(p, rastro)

        if acao == "pedir_nova_foto":
            motivo = (f"A foto apresenta problemas ({', '.join(p.problemas_imagem)})"
                      if p.problemas_imagem else
                      f"Não tenho certeza do material (confiança {_pct(p.confianca_visual)})")
            return Decisao(acao, None, fonte,
                           f"{motivo}. Tire outra foto com boa luz, foco e o item centralizado, "
                           "sobre fundo liso, ou me diga qual é o material.", rastro)

        if conf_i < LIMIAR_INTENCAO:
            rastro.append(f"R9: intenção '{intencao}' com confiança {_pct(conf_i)} < "
                          f"{_pct(LIMIAR_INTENCAO)}: pede para reformular")
            nome = BASE[classe]["nome"]
            return Decisao("esclarecer_pergunta", classe, fonte,
                           f"Identifiquei {nome} na imagem, mas não entendi bem a pergunta. "
                           "Você quer saber se pode reciclar, a cor da lixeira, como preparar "
                           "ou onde descartar?", rastro)

        corpo = self._corpo(intencao, classe, p.condicoes, rastro)
        prefixo = ""
        if acao == "confirmar_material":
            seg = BASE[p.segunda_classe]["nome"]
            prefixo = (f"Estou em dúvida entre {BASE[classe]['nome']} ({_pct(p.confianca_visual)}) "
                       f"e {seg} ({_pct(p.segunda_confianca)}). Se for {seg}, me avise. "
                       f"Considerando {BASE[classe]['nome']}: ")
        elif acao == "responder_com_ressalva" and fonte == "imagem":
            prefixo = (f"Você citou {', '.join(BASE[m]['nome'] for m in p.materiais_texto)}, mas a "
                       f"imagem indica {BASE[classe]['nome']} ({_pct(p.confianca_visual)}). "
                       "Seguindo a imagem: ")
        elif acao == "responder_com_ressalva" and fonte == "texto":
            prefixo = (f"A imagem ficou incerta ({_pct(p.confianca_visual)}); usando o material "
                       f"que você informou ({BASE[classe]['nome']}): ")
        return Decisao(acao, classe, fonte, prefixo + corpo, rastro)
