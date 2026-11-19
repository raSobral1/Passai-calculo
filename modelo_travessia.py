"""Modelagem matemática portátil do projeto Passaí.

Aplica limites, derivadas e integrais a um cenário simulado de travessia.
O programa utiliza somente a biblioteca padrão do Python e gera seis gráficos
vetoriais em SVG, um painel HTML e um resumo textual da análise.

Guia de leitura:
1. Configuração e dados do cenário.
2. Modelos matemáticos: tempo, derivadas, integrais e otimização.
3. Validação e comparação dos cenários.
4. Desenho SVG: peças visuais compartilhadas pelos seis gráficos.
5. Gráficos: aplicação dos modelos às figuras do relatório.
6. Saídas: tabela, resumo em texto e painel HTML.
7. Execução: argumentos do terminal e função main.

SVG é uma imagem descrita em texto. Essa parte ocupa boa parte do arquivo,
mas permite executar o projeto sem instalar bibliotecas externas.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


# 1. CONFIGURAÇÃO E DADOS DO CENÁRIO
# Cores da apresentação; os parâmetros físicos ficam em CenarioTravessia.
COR_VINHO = "#A80F45"
COR_ROSA = "#E91E63"
COR_AZUL = "#245B78"
COR_VERDE = "#23856D"
COR_AMARELO = "#F1B24A"
COR_TEXTO = "#25242A"
COR_GRADE = "#D9D9DF"
COR_FUNDO = "#FAFAFC"
TOLERANCIA_MARGEM_S = 1e-9  # Tolerância de arredondamento na comparação.
LIMITE_BUSCA_EXTENSAO_S = 3600  # Proteção da busca; não é um tempo recomendado.


def configurar_terminal_utf8() -> None:
    """Evita caracteres quebrados ao executar o programa no Windows."""
    if os.name == "nt":
        os.system("chcp 65001 > nul")
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class CenarioTravessia:
    """Grandezas do cenário didático e suas respectivas unidades."""

    largura_m: float = 10.0
    tempo_reacao_s: float = 2.0
    tempo_base_s: float = 12.0
    extensao_s: float = 4.0
    velocidade_referencia_m_s: float = 1.0
    taxa_inicial_ped_s: float = 0.4
    crescimento_taxa_ped_s2: float = 0.05
    peso_acessibilidade: float = 100.0
    decaimento_beneficio_s_inv: float = 0.35
    peso_impacto_viario: float = 2.0
    velocidades_m_s: tuple[float, ...] = (1.25, 1.0, 0.8, 0.6)

    def __post_init__(self) -> None:
        """Valida os parâmetros assim que um cenário é criado."""
        positivos = {
            "largura_m": self.largura_m,
            "tempo_base_s": self.tempo_base_s,
            "velocidade_referencia_m_s": self.velocidade_referencia_m_s,
            "peso_acessibilidade": self.peso_acessibilidade,
            "decaimento_beneficio_s_inv": self.decaimento_beneficio_s_inv,
            "peso_impacto_viario": self.peso_impacto_viario,
        }
        for nome, valor in positivos.items():
            if not math.isfinite(valor) or valor <= 0:
                raise ValueError(f"{nome} deve ser um número positivo.")

        nao_negativos = {
            "tempo_reacao_s": self.tempo_reacao_s,
            "extensao_s": self.extensao_s,
            "taxa_inicial_ped_s": self.taxa_inicial_ped_s,
            "crescimento_taxa_ped_s2": self.crescimento_taxa_ped_s2,
        }
        for nome, valor in nao_negativos.items():
            if not math.isfinite(valor) or valor < 0:
                raise ValueError(f"{nome} deve ser um número não negativo.")

        if self.tempo_base_s <= self.tempo_reacao_s:
            raise ValueError("O tempo-base deve superar o tempo de reação.")
        if not self.velocidades_m_s:
            raise ValueError("Informe pelo menos uma velocidade para a análise.")
        if any(not math.isfinite(v) or v <= 0 for v in self.velocidades_m_s):
            raise ValueError("Todas as velocidades devem ser positivas.")

    @property
    def tempo_estendido_s(self) -> float:
        """Soma o tempo-base e a extensão, ambos em segundos."""
        return self.tempo_base_s + self.extensao_s


@dataclass(frozen=True)
class ResultadoCenario:
    """Reúne tempo, margens e classificações de uma velocidade."""

    velocidade_m_s: float
    tempo_necessario_s: float
    margem_base_s: float
    margem_estendida_s: float
    situacao_base: str
    situacao_estendida: str
    extensao_muda_resultado: bool


@dataclass(frozen=True)
class AreaGrafico:
    """Transforma coordenadas matemáticas em posições dentro de um SVG."""

    x: float
    y: float
    largura: float
    altura: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def px(self, valor: float) -> float:
        """Converte uma coordenada horizontal do modelo para pixels."""
        return self.x + (valor - self.x_min) * self.largura / (self.x_max - self.x_min)

    def py(self, valor: float) -> float:
        """Inverte o eixo vertical: na imagem, os pixels crescem para baixo."""
        return self.y + self.altura - (valor - self.y_min) * self.altura / (self.y_max - self.y_min)


# 2. MODELOS MATEMÁTICOS
# Cálculos independentes do desenho e da gravação de arquivos.
def tempo_necessario_s(velocidade_m_s: float, cenario: CenarioTravessia) -> float:
    """Calcula T(v) = r + L/v para v > 0."""

    if not math.isfinite(velocidade_m_s) or velocidade_m_s <= 0:
        raise ValueError("A velocidade deve ser positiva e finita.")
    return cenario.tempo_reacao_s + cenario.largura_m / velocidade_m_s


def derivada_tempo(velocidade_m_s: float, cenario: CenarioTravessia) -> float:
    """Calcula T'(v) = -L/v²."""

    if velocidade_m_s <= 0:
        raise ValueError("A velocidade deve ser positiva.")
    return -cenario.largura_m / velocidade_m_s**2


def segunda_derivada_tempo(velocidade_m_s: float, cenario: CenarioTravessia) -> float:
    """Calcula T''(v) = 2L/v³."""

    if velocidade_m_s <= 0:
        raise ValueError("A velocidade deve ser positiva.")
    return 2 * cenario.largura_m / velocidade_m_s**3


def velocidade_minima_m_s(tempo_disponivel_s: float, cenario: CenarioTravessia) -> float:
    """Resolve r + L/v <= G para encontrar a menor velocidade compatível."""

    if tempo_disponivel_s <= cenario.tempo_reacao_s:
        raise ValueError("O tempo disponível deve superar o tempo de reação.")
    return cenario.largura_m / (tempo_disponivel_s - cenario.tempo_reacao_s)


def taxa_chegada_ped_s(instante_s: float, cenario: CenarioTravessia) -> float:
    """Calcula λ(t) = λ₀ + βt."""

    if not math.isfinite(instante_s) or instante_s < 0:
        raise ValueError("O instante deve ser não negativo e finito.")
    return cenario.taxa_inicial_ped_s + cenario.crescimento_taxa_ped_s2 * instante_s


def chegadas_acumuladas_analitica(
    inicio_s: float,
    fim_s: float,
    cenario: CenarioTravessia,
) -> float:
    """Calcula exatamente a integral de λ(t) entre dois instantes."""

    validar_janela(inicio_s, fim_s)
    lambda_0 = cenario.taxa_inicial_ped_s
    beta = cenario.crescimento_taxa_ped_s2
    return lambda_0 * (fim_s - inicio_s) + beta * (fim_s**2 - inicio_s**2) / 2


def integrar_trapezios(
    funcao: Callable[[float, CenarioTravessia], float],
    inicio: float,
    fim: float,
    cenario: CenarioTravessia,
    pontos: int,
) -> float:
    """Integra uma função por trapézios; cada extremo recebe metade do peso."""
    if pontos < 2:
        raise ValueError("A integração numérica exige pelo menos dois pontos.")
    passo = (fim - inicio) / (pontos - 1)
    soma = (funcao(inicio, cenario) + funcao(fim, cenario)) / 2
    for indice in range(1, pontos - 1):
        soma += funcao(inicio + indice * passo, cenario)
    return soma * passo


def chegadas_acumuladas_numerica(
    inicio_s: float,
    fim_s: float,
    cenario: CenarioTravessia,
    pontos: int = 2001,
) -> float:
    """Valida a integral analítica pela regra numérica dos trapézios."""

    validar_janela(inicio_s, fim_s)
    if pontos < 2:
        raise ValueError("A integração numérica exige pelo menos dois pontos.")
    if math.isclose(inicio_s, fim_s):
        return 0.0

    return integrar_trapezios(taxa_chegada_ped_s, inicio_s, fim_s, cenario, pontos)


def integral_tempo_por_velocidade(
    velocidade_inicial_m_s: float,
    velocidade_final_m_s: float,
    cenario: CenarioTravessia,
) -> float:
    """Calcula ∫T(v)dv = r(b-a) + L ln(b/a), para 0 < a < b."""

    validar_intervalo_velocidade(velocidade_inicial_m_s, velocidade_final_m_s)
    a, b = velocidade_inicial_m_s, velocidade_final_m_s
    return cenario.tempo_reacao_s * (b - a) + cenario.largura_m * math.log(b / a)


def integral_tempo_numerica(
    velocidade_inicial_m_s: float,
    velocidade_final_m_s: float,
    cenario: CenarioTravessia,
    pontos: int = 4001,
) -> float:
    """Valida a integral de T(v) pela regra composta dos trapézios."""

    validar_intervalo_velocidade(velocidade_inicial_m_s, velocidade_final_m_s)
    return integrar_trapezios(
        tempo_necessario_s, velocidade_inicial_m_s, velocidade_final_m_s, cenario, pontos
    )


def tempo_medio_no_intervalo(
    velocidade_inicial_m_s: float,
    velocidade_final_m_s: float,
    cenario: CenarioTravessia,
) -> float:
    """Calcula o valor médio de T(v) em um intervalo de velocidades."""

    integral = integral_tempo_por_velocidade(
        velocidade_inicial_m_s, velocidade_final_m_s, cenario
    )
    return integral / (velocidade_final_m_s - velocidade_inicial_m_s)


def custo_acessibilidade(extensao_s: float, cenario: CenarioTravessia) -> float:
    """Índice decrescente do impacto sobre pedestres: A·exp(-k·e)."""

    validar_extensao(extensao_s)
    return cenario.peso_acessibilidade * math.exp(
        -cenario.decaimento_beneficio_s_inv * extensao_s
    )


def custo_impacto_viario(extensao_s: float, cenario: CenarioTravessia) -> float:
    """Índice crescente do impacto veicular: B·e²."""

    validar_extensao(extensao_s)
    return cenario.peso_impacto_viario * extensao_s**2


def custo_total_extensao(extensao_s: float, cenario: CenarioTravessia) -> float:
    """Soma os índices de acessibilidade e impacto viário."""
    return custo_acessibilidade(extensao_s, cenario) + custo_impacto_viario(extensao_s, cenario)


def derivada_custo_total(extensao_s: float, cenario: CenarioTravessia) -> float:
    """Calcula C'(e) para localizar o ponto de custo total mínimo."""

    validar_extensao(extensao_s)
    a = cenario.peso_acessibilidade
    k = cenario.decaimento_beneficio_s_inv
    b = cenario.peso_impacto_viario
    return -a * k * math.exp(-k * extensao_s) + 2 * b * extensao_s


def segunda_derivada_custo_total(extensao_s: float, cenario: CenarioTravessia) -> float:
    """Calcula C''(e); seu valor positivo confirma a convexidade do modelo."""

    validar_extensao(extensao_s)
    a = cenario.peso_acessibilidade
    k = cenario.decaimento_beneficio_s_inv
    b = cenario.peso_impacto_viario
    return a * k**2 * math.exp(-k * extensao_s) + 2 * b


def extensao_otima_s(cenario: CenarioTravessia, tolerancia: float = 1e-10) -> float:
    """Resolve C'(e)=0 por bisseção, sem depender de bibliotecas externas."""

    if tolerancia <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    if derivada_custo_total(0.0, cenario) >= 0:
        return 0.0
    # Encontramos extremos com sinais diferentes da derivada do custo.
    limite_superior = max(1.0, cenario.extensao_s)
    while derivada_custo_total(limite_superior, cenario) < 0:
        limite_superior *= 2
        if limite_superior > LIMITE_BUSCA_EXTENSAO_S:
            raise RuntimeError("Não foi possível limitar a raiz da função de custo.")
    # A convexidade permite descartar metade do intervalo a cada passo.
    inferior = 0.0
    while limite_superior - inferior > tolerancia:
        meio = (inferior + limite_superior) / 2
        if derivada_custo_total(meio, cenario) < 0:
            inferior = meio
        else:
            limite_superior = meio
    return (inferior + limite_superior) / 2


# 3. VALIDAÇÃO E COMPARAÇÃO DOS CENÁRIOS
def validar_janela(inicio_s: float, fim_s: float) -> None:
    """Rejeita instantes negativos ou uma janela com fim anterior ao início."""
    if not 0 <= inicio_s <= fim_s:
        raise ValueError("A janela deve obedecer 0 <= início <= fim.")


def validar_intervalo_velocidade(inicio_m_s: float, fim_m_s: float) -> None:
    """Exige velocidades finitas com 0 < início < fim."""
    if not (math.isfinite(inicio_m_s) and math.isfinite(fim_m_s)):
        raise ValueError("As velocidades devem ser finitas.")
    if not 0 < inicio_m_s < fim_m_s:
        raise ValueError("O intervalo deve obedecer 0 < velocidade inicial < velocidade final.")


def validar_extensao(extensao_s: float) -> None:
    """Exige uma extensão finita e não negativa."""
    if not math.isfinite(extensao_s) or extensao_s < 0:
        raise ValueError("A extensão deve ser não negativa e finita.")


def janela_adicional_s(cenario: CenarioTravessia) -> tuple[float, float]:
    """Retorna a janela de chegadas que passa a caber graças à extensão."""

    tempo = tempo_necessario_s(cenario.velocidade_referencia_m_s, cenario)
    limite_base = max(0.0, cenario.tempo_base_s - tempo)
    limite_estendido = max(0.0, cenario.tempo_estendido_s - tempo)
    return limite_base, limite_estendido


def analisar_velocidade(velocidade_m_s: float, cenario: CenarioTravessia) -> ResultadoCenario:
    """Compara o tempo necessário com as duas durações disponíveis."""
    tempo = tempo_necessario_s(velocidade_m_s, cenario)
    # Margem positiva é folga; negativa significa que ainda falta tempo.
    margem_base = cenario.tempo_base_s - tempo
    margem_estendida = cenario.tempo_estendido_s - tempo
    conclui_base = margem_base >= -TOLERANCIA_MARGEM_S
    conclui_estendida = margem_estendida >= -TOLERANCIA_MARGEM_S
    return ResultadoCenario(
        velocidade_m_s=velocidade_m_s,
        tempo_necessario_s=tempo,
        margem_base_s=margem_base,
        margem_estendida_s=margem_estendida,
        situacao_base="Conclui" if conclui_base else "Tempo insuficiente",
        situacao_estendida="Conclui" if conclui_estendida else "Tempo insuficiente",
        extensao_muda_resultado=not conclui_base and conclui_estendida,
    )


def analisar_cenarios(cenario: CenarioTravessia) -> list[ResultadoCenario]:
    """Aplica a mesma análise a cada velocidade cadastrada."""
    return [analisar_velocidade(v, cenario) for v in cenario.velocidades_m_s]


def valores_entre(inicio: float, fim: float, quantidade: int) -> list[float]:
    """Cria amostras igualmente espaçadas para desenhar curvas e eixos."""
    if quantidade < 2:
        return [inicio]
    passo = (fim - inicio) / (quantidade - 1)
    return [inicio + passo * indice for indice in range(quantidade)]


# 4. RECURSOS VISUAIS SVG
# Cada função devolve uma peça da imagem. A ordem determina as camadas.
def texto_svg(
    x: float,
    y: float,
    conteudo: str,
    tamanho: int = 18,
    ancora: str = "middle",
    cor: str = COR_TEXTO,
    peso: str = "normal",
    rotacao: float | None = None,
) -> str:
    """Escreve um rótulo SVG, escapando caracteres especiais do texto."""
    transformacao = f' transform="rotate({rotacao} {x:.2f} {y:.2f})"' if rotacao is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{ancora}" '
        f'font-family="Arial, sans-serif" font-size="{tamanho}" fill="{cor}" '
        f'font-weight="{peso}"{transformacao}>{html.escape(conteudo)}</text>'
    )


def linha_svg(x1: float, y1: float, x2: float, y2: float, cor: str, largura: float = 2,
              tracejado: str | None = None) -> str:
    """Desenha um segmento, opcionalmente tracejado."""
    traco = f' stroke-dasharray="{tracejado}"' if tracejado else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{cor}" stroke-width="{largura}"{traco}/>'
    )


def polilinha_svg(pontos: Iterable[tuple[float, float]], cor: str, largura: float = 4) -> str:
    """Une pontos para representar uma curva sem preenchimento."""
    coordenadas = " ".join(f"{x:.2f},{y:.2f}" for x, y in pontos)
    return (
        f'<polyline points="{coordenadas}" fill="none" stroke="{cor}" '
        f'stroke-width="{largura}" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def poligono_svg(pontos: Iterable[tuple[float, float]], preenchimento: str, opacidade: float = 1,
                 contorno: str = "none") -> str:
    """Fecha uma área de pontos e aplica seu preenchimento."""
    coordenadas = " ".join(f"{x:.2f},{y:.2f}" for x, y in pontos)
    return (
        f'<polygon points="{coordenadas}" fill="{preenchimento}" fill-opacity="{opacidade}" '
        f'stroke="{contorno}"/>'
    )


def circulo_svg(x: float, y: float, raio: float, cor: str, contorno: str = "white") -> str:
    """Cria um marcador para destacar um ponto da análise."""
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{raio}" fill="{cor}" '
        f'stroke="{contorno}" stroke-width="2"/>'
    )


def criar_documento_svg(titulo: str, subtitulo: str, elementos: Sequence[str],
                        largura: int = 1200, altura: int = 760) -> str:
    """Reúne fundo, títulos e elementos em uma imagem SVG completa."""
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" '
            f'viewBox="0 0 {largura} {altura}">',
            f'<rect width="{largura}" height="{altura}" fill="{COR_FUNDO}"/>',
            texto_svg(largura / 2, 43, titulo, 28, peso="bold"),
            texto_svg(largura / 2, 69, subtitulo, 14, cor="#66636B"),
            *elementos,
            "</svg>",
        ]
    )


def desenhar_eixos(
    elementos: list[str],
    area: AreaGrafico,
    valores_x: Sequence[float],
    valores_y: Sequence[float],
    rotulo_x: str,
    rotulo_y: str,
    casas_x: int = 1,
    casas_y: int = 0,
) -> None:
    """Acrescenta moldura, grade, marcas e rótulos à lista de elementos."""
    elementos.append(
        f'<rect x="{area.x}" y="{area.y}" width="{area.largura}" height="{area.altura}" '
        'fill="white" stroke="#CBCBD2" stroke-width="1.5"/>'
    )
    for valor in valores_x:
        x = area.px(valor)
        elementos.append(linha_svg(x, area.y, x, area.y + area.altura, COR_GRADE, 1, "6 5"))
        elementos.append(texto_svg(x, area.y + area.altura + 26, f"{valor:.{casas_x}f}", 14))
    for valor in valores_y:
        y = area.py(valor)
        elementos.append(linha_svg(area.x, y, area.x + area.largura, y, COR_GRADE, 1, "6 5"))
        elementos.append(texto_svg(area.x - 13, y + 5, f"{valor:.{casas_y}f}", 14, ancora="end"))
    elementos.append(texto_svg(area.x + area.largura / 2, area.y + area.altura + 59, rotulo_x, 17))
    elementos.append(texto_svg(area.x - 67, area.y + area.altura / 2, rotulo_y, 17, rotacao=-90))


def salvar_svg(caminho: Path, titulo: str, subtitulo: str, elementos: Sequence[str],
               largura: int = 1200, altura: int = 760) -> Path:
    """Grava a imagem em UTF-8 e devolve seu caminho."""
    caminho.write_text(criar_documento_svg(titulo, subtitulo, elementos, largura, altura), encoding="utf-8")
    return caminho


# 5. OS SEIS GRÁFICOS DO RELATÓRIO
# Cada figura monta eixos, curvas e anotações antes de gravar o SVG.
def grafico_tempo(cenario: CenarioTravessia, pasta: Path, v_min: float, v_max: float) -> Path:
    """Figura 1: compara T(v) aos tempos-base e estendido."""
    area = AreaGrafico(105, 95, 1035, 570, v_min, v_max, 5, 30)
    elementos: list[str] = []
    desenhar_eixos(
        elementos, area, valores_entre(v_min, v_max, 7), [5, 10, 15, 20, 25, 30],
        "Velocidade de deslocamento v (m/s)", "Tempo necessário T(v) (s)", casas_x=1,
    )
    v_base = velocidade_minima_m_s(cenario.tempo_base_s, cenario)
    v_ext = velocidade_minima_m_s(cenario.tempo_estendido_s, cenario)
    elementos.append(
        f'<rect x="{area.px(v_ext):.2f}" y="{area.y}" width="{area.px(v_base)-area.px(v_ext):.2f}" '
        f'height="{area.altura}" fill="#FBE1A7" fill-opacity="0.48"/>'
    )
    elementos.append(linha_svg(area.x, area.py(cenario.tempo_base_s), area.x + area.largura,
                               area.py(cenario.tempo_base_s), COR_AZUL, 3))
    elementos.append(linha_svg(area.x, area.py(cenario.tempo_estendido_s), area.x + area.largura,
                               area.py(cenario.tempo_estendido_s), COR_VERDE, 3))
    velocidades = valores_entre(v_min, v_max, 350)
    pontos = [(area.px(v), area.py(tempo_necessario_s(v, cenario))) for v in velocidades]
    elementos.append(polilinha_svg(pontos, COR_VINHO, 5))
    for velocidade in cenario.velocidades_m_s:
        elementos.append(circulo_svg(area.px(velocidade), area.py(tempo_necessario_s(velocidade, cenario)),
                                     7, COR_ROSA))
    elementos.append(circulo_svg(area.px(v_base), area.py(cenario.tempo_base_s), 8, COR_AZUL))
    elementos.append(circulo_svg(area.px(v_ext), area.py(cenario.tempo_estendido_s), 8, COR_VERDE))
    elementos.append(texto_svg(area.px(v_base) + 15, area.py(cenario.tempo_base_s) + 28,
                               f"v = {v_base:.3f} m/s", 15, ancora="start", cor=COR_AZUL, peso="bold"))
    elementos.append(texto_svg(area.px(v_ext) + 15, area.py(cenario.tempo_estendido_s) - 17,
                               f"v = {v_ext:.3f} m/s", 15, ancora="start", cor=COR_VERDE, peso="bold"))
    elementos.extend(
        [
            linha_svg(765, 110, 815, 110, COR_VINHO, 5), texto_svg(830, 116, "T(v) = r + L/v", 15, ancora="start"),
            linha_svg(765, 137, 815, 137, COR_AZUL, 3), texto_svg(830, 143, "Tempo-base", 15, ancora="start"),
            linha_svg(765, 164, 815, 164, COR_VERDE, 3), texto_svg(830, 170, "Tempo com extensão", 15, ancora="start"),
        ]
    )
    return salvar_svg(
        pasta / "01_tempo_por_velocidade.svg",
        "Tempo necessário em função da velocidade",
        "Faixa destacada em que a extensão altera o resultado do cenário simulado",
        elementos,
    )


def grafico_derivadas(cenario: CenarioTravessia, pasta: Path, v_min: float, v_max: float) -> Path:
    """Figura 2: apresenta a taxa de variação e a convexidade de T(v)."""
    area_1 = AreaGrafico(105, 105, 1035, 245, v_min, v_max, -65, 0)
    area_2 = AreaGrafico(105, 435, 1035, 245, v_min, v_max, 0, 330)
    elementos: list[str] = []
    desenhar_eixos(elementos, area_1, valores_entre(v_min, v_max, 7), [-60, -45, -30, -15, 0],
                   "", "T'(v)", casas_x=1)
    desenhar_eixos(elementos, area_2, valores_entre(v_min, v_max, 7), [0, 75, 150, 225, 300],
                   "Velocidade de deslocamento v (m/s)", "T''(v)", casas_x=1)
    velocidades = valores_entre(v_min, v_max, 350)
    elementos.append(polilinha_svg([(area_1.px(v), area_1.py(derivada_tempo(v, cenario))) for v in velocidades],
                                   COR_VINHO, 5))
    elementos.append(polilinha_svg([(area_2.px(v), area_2.py(segunda_derivada_tempo(v, cenario))) for v in velocidades],
                                   COR_AZUL, 5))
    elementos.append(texto_svg(120, 92, "Primeira derivada — taxa de variação", 19, ancora="start", peso="bold"))
    elementos.append(texto_svg(120, 422, "Segunda derivada — convexidade", 19, ancora="start", peso="bold"))
    elementos.append(texto_svg(1125, 325, "T'(v) < 0: velocidades maiores reduzem o tempo.",
                               15, ancora="end", cor=COR_VINHO, peso="bold"))
    elementos.append(texto_svg(1125, 470, "T''(v) > 0: maior sensibilidade nas velocidades menores.",
                               15, ancora="end", cor=COR_AZUL, peso="bold"))
    return salvar_svg(
        pasta / "02_derivadas_e_sensibilidade.svg",
        "Taxa de variação e sensibilidade do modelo",
        "A mesma redução de velocidade produz um aumento maior de tempo em velocidades menores",
        elementos,
    )


def grafico_integral(cenario: CenarioTravessia, pasta: Path) -> Path:
    """Figura 3: destaca a integral de chegadas na janela adicional."""
    limite_t = cenario.tempo_estendido_s
    inicio, fim = janela_adicional_s(cenario)
    area = AreaGrafico(105, 95, 1035, 570, 0, limite_t, 0, 1.4)
    elementos: list[str] = []
    desenhar_eixos(elementos, area, [0, 2, 4, 6, 8, 10, 12, 14, 16],
                   [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
                   "Instante desde o início da fase (s)", "Taxa λ(t) (pedestres/s)", casas_x=0, casas_y=1)
    # A área sob a taxa representa chegadas esperadas, não medições reais.
    instantes_janela = valores_entre(inicio, fim, 100)
    poligono = [(area.px(inicio), area.py(0))]
    poligono.extend((area.px(t), area.py(taxa_chegada_ped_s(t, cenario))) for t in instantes_janela)
    poligono.append((area.px(fim), area.py(0)))
    elementos.append(poligono_svg(poligono, "#FBE1A7", 0.82, "#E2B55C"))
    instantes = valores_entre(0, limite_t, 350)
    elementos.append(polilinha_svg([(area.px(t), area.py(taxa_chegada_ped_s(t, cenario))) for t in instantes],
                                   COR_VINHO, 5))
    elementos.append(linha_svg(area.px(inicio), area.y, area.px(inicio), area.y + area.altura,
                               COR_VERDE, 2, "8 6"))
    elementos.append(linha_svg(area.px(fim), area.y, area.px(fim), area.y + area.altura,
                               COR_VERDE, 2, "8 6"))
    acumulado = chegadas_acumuladas_analitica(inicio, fim, cenario)
    elementos.append(texto_svg(area.px((inicio + fim) / 2), area.py(0.18),
                               f"Integral = {acumulado:.3f} chegadas esperadas", 17, peso="bold"))
    elementos.append(texto_svg(1120, 120, "λ(t) = λ₀ + βt", 16, ancora="end", cor=COR_VINHO, peso="bold"))
    return salvar_svg(
        pasta / "03_integral_chegadas_acumuladas.svg",
        "Integral da taxa de chegada",
        "A área destacada representa a grandeza acumulada na janela adicional",
        elementos,
    )


def grafico_mapa(cenario: CenarioTravessia, pasta: Path, v_min: float, v_max: float) -> Path:
    """Figura 4: classifica combinações de largura e velocidade."""
    l_min, l_max = 4.0, max(16.0, cenario.largura_m * 1.5)
    area = AreaGrafico(105, 95, 1035, 570, v_min, v_max, l_min, l_max)
    elementos: list[str] = []
    desenhar_eixos(elementos, area, valores_entre(v_min, v_max, 7), [4, 6, 8, 10, 12, 14, 16],
                   "Velocidade de deslocamento v (m/s)", "Largura da travessia L (m)", casas_x=1)
    velocidades = valores_entre(v_min, v_max, 300)
    def fronteira(tempo_disponivel: float) -> list[tuple[float, float]]:
        """Calcula a largura-limite e recorta a curva para caber nos eixos."""
        pontos = []
        for velocidade in velocidades:
            largura_limite = velocidade * (tempo_disponivel - cenario.tempo_reacao_s)
            largura_visivel = min(l_max, max(l_min, largura_limite))
            pontos.append((area.px(velocidade), area.py(largura_visivel)))
        return pontos

    base = fronteira(cenario.tempo_base_s)
    estendida = fronteira(cenario.tempo_estendido_s)

    verde = [(area.px(v_min), area.py(l_min)), (area.px(v_max), area.py(l_min)), *reversed(base)]
    amarelo = [*base, *reversed(estendida)]
    vermelho = [*estendida, (area.px(v_max), area.py(l_max)), (area.px(v_min), area.py(l_max))]
    elementos.append(poligono_svg(vermelho, "#F3B8B8", 0.9))
    elementos.append(poligono_svg(amarelo, "#FBE1A7", 0.9))
    elementos.append(poligono_svg(verde, "#B9E1D3", 0.9))
    elementos.append(polilinha_svg(base, COR_AZUL, 4))
    elementos.append(polilinha_svg(estendida, COR_VERDE, 4))
    elementos.append(circulo_svg(area.px(cenario.velocidade_referencia_m_s), area.py(cenario.largura_m),
                                 9, COR_VINHO))
    elementos.append(texto_svg(125, 125, "Verde: conclui no tempo-base", 15, ancora="start", peso="bold"))
    elementos.append(texto_svg(125, 151, "Amarelo: conclui somente com extensão", 15, ancora="start", peso="bold"))
    elementos.append(texto_svg(125, 177, "Vermelho: tempo ainda insuficiente", 15, ancora="start", peso="bold"))
    return salvar_svg(
        pasta / "04_mapa_decisao.svg",
        "Mapa de decisão por largura e velocidade",
        "Comparação dos resultados com 12 segundos e com a extensão até 16 segundos",
        elementos,
    )


def intervalo_velocidades_cenario(cenario: CenarioTravessia) -> tuple[float, float]:
    """Obtém uma faixa não degenerada para a análise integral das velocidades."""

    inicio = min(cenario.velocidades_m_s)
    fim = max(cenario.velocidades_m_s)
    if math.isclose(inicio, fim):
        inicio = 0.8 * inicio
        fim = 1.2 * fim
    return inicio, fim


def grafico_tempo_medio_integral(cenario: CenarioTravessia, pasta: Path) -> Path:
    """Figura 5: compara T(v) ao seu valor médio uniforme no intervalo."""
    inicio, fim = intervalo_velocidades_cenario(cenario)
    # A média dá peso igual a todas as velocidades do intervalo.
    media = tempo_medio_no_intervalo(inicio, fim, cenario)
    y_max = math.ceil(tempo_necessario_s(inicio, cenario) * 1.18 / 5) * 5
    area = AreaGrafico(105, 95, 1035, 570, inicio, fim, 0, y_max)
    elementos: list[str] = []
    desenhar_eixos(
        elementos,
        area,
        valores_entre(inicio, fim, 7),
        valores_entre(0, y_max, 6),
        "Velocidade considerada v (m/s)",
        "Tempo necessário T(v) (s)",
        casas_x=2,
        casas_y=0,
    )
    velocidades = valores_entre(inicio, fim, 350)
    curva = [(area.px(v), area.py(tempo_necessario_s(v, cenario))) for v in velocidades]
    preenchimento = [(area.px(inicio), area.py(0)), *curva, (area.px(fim), area.py(0))]
    elementos.append(poligono_svg(preenchimento, "#E8C4D2", 0.62, "#CF8DA5"))
    elementos.append(polilinha_svg(curva, COR_VINHO, 5))
    elementos.append(linha_svg(area.x, area.py(media), area.x + area.largura, area.py(media),
                               COR_VERDE, 4, "10 7"))
    elementos.append(texto_svg(1120, area.py(media) - 12,
                               f"Valor médio = {media:.3f} s", 16, ancora="end",
                               cor=COR_VERDE, peso="bold"))
    integral = integral_tempo_por_velocidade(inicio, fim, cenario)
    elementos.append(texto_svg(125, 130,
                               f"∫ T(v)dv = {integral:.3f} s·(m/s)", 16,
                               ancora="start", peso="bold"))
    elementos.append(texto_svg(125, 157,
                               "T̄ = [1/(b-a)] ∫ T(v)dv", 16,
                               ancora="start", cor=COR_AZUL, peso="bold"))
    return salvar_svg(
        pasta / "05_tempo_medio_por_integral.svg",
        "Tempo médio de travessia por integral definida",
        "A área sob T(v) permite avaliar conjuntamente a faixa de velocidades simulada",
        elementos,
    )


def grafico_otimizacao_extensao(cenario: CenarioTravessia, pasta: Path) -> Path:
    """Figura 6: apresenta os custos, o mínimo de C(e) e sua derivada."""
    # O mínimo depende de pesos didáticos, não garante atender todos.
    otima = extensao_otima_s(cenario)
    limite = max(8.0, cenario.extensao_s * 2, otima * 2)
    extensoes = valores_entre(0, limite, 400)
    custos_pedestre = [custo_acessibilidade(e, cenario) for e in extensoes]
    custos_viario = [custo_impacto_viario(e, cenario) for e in extensoes]
    custos_totais = [custo_total_extensao(e, cenario) for e in extensoes]
    derivadas = [derivada_custo_total(e, cenario) for e in extensoes]

    y_max = math.ceil(max(*custos_pedestre, *custos_viario, *custos_totais) / 20) * 20
    d_min = math.floor(min(derivadas) / 10) * 10
    d_max = math.ceil(max(derivadas) / 10) * 10
    area_custo = AreaGrafico(105, 95, 1035, 430, 0, limite, 0, y_max)
    area_derivada = AreaGrafico(105, 640, 1035, 145, 0, limite, d_min, d_max)
    marcas_x = valores_entre(0, limite, 6)
    elementos: list[str] = []
    desenhar_eixos(
        elementos, area_custo, marcas_x, valores_entre(0, y_max, 6),
        "", "Índice de impacto", casas_x=1, casas_y=0,
    )
    desenhar_eixos(
        elementos, area_derivada, marcas_x, [d_min, 0, d_max],
        "Extensão adicional e (s)", "C'(e)", casas_x=1, casas_y=0,
    )
    # Área, valores, cor e espessura de cada curva.
    curvas = [
        (area_custo, custos_pedestre, COR_VINHO, 4),
        (area_custo, custos_viario, COR_AZUL, 4),
        (area_custo, custos_totais, COR_VERDE, 6),
        (area_derivada, derivadas, COR_ROSA, 4),
    ]
    for area, valores, cor, espessura in curvas:
        pontos = [(area.px(e), area.py(valor)) for e, valor in zip(extensoes, valores)]
        elementos.append(polilinha_svg(pontos, cor, espessura))
    x_otima_custo = area_custo.px(otima)
    x_otima_derivada = area_derivada.px(otima)
    elementos.append(linha_svg(x_otima_custo, area_custo.y, x_otima_custo,
                               area_custo.y + area_custo.altura, COR_VERDE, 2, "8 6"))
    elementos.append(linha_svg(x_otima_derivada, area_derivada.y, x_otima_derivada,
                               area_derivada.y + area_derivada.altura, COR_VERDE, 2, "8 6"))
    elementos.append(circulo_svg(x_otima_custo, area_custo.py(custo_total_extensao(otima, cenario)),
                                 9, COR_VERDE))
    elementos.append(circulo_svg(x_otima_derivada, area_derivada.py(0), 8, COR_ROSA))
    x_atual = area_custo.px(min(cenario.extensao_s, limite))
    elementos.append(linha_svg(x_atual, area_custo.y, x_atual,
                               area_custo.y + area_custo.altura, COR_AMARELO, 3, "5 5"))
    elementos.append(texto_svg(x_atual + 10, 500, f"Atual: {cenario.extensao_s:.2f} s",
                               14, ancora="start", cor="#9A6510", peso="bold"))
    elementos.extend([
        linha_svg(750, 118, 800, 118, COR_VINHO, 4),
        texto_svg(815, 124, "Impacto sobre pedestres", 14, ancora="start"),
        linha_svg(750, 145, 800, 145, COR_AZUL, 4),
        texto_svg(815, 151, "Impacto viário", 14, ancora="start"),
        linha_svg(750, 172, 800, 172, COR_VERDE, 6),
        texto_svg(815, 178, "Custo total C(e)", 14, ancora="start"),
        texto_svg(125, 125, f"Mínimo em e = {otima:.3f} s", 16,
                  ancora="start", cor=COR_VERDE, peso="bold"),
        texto_svg(125, 153, "C'(e) = 0 e C''(e) > 0", 15,
                  ancora="start", peso="bold"),
        texto_svg(120, 625, "Derivada do custo total — mudança de sinal no ponto ótimo",
                  18, ancora="start", peso="bold"),
    ])
    return salvar_svg(
        pasta / "06_otimizacao_extensao.svg",
        "Otimização matemática da extensão do sinal",
        "Equilíbrio simulado entre benefício de acessibilidade e impacto no fluxo viário",
        elementos,
        altura=880,
    )


# 6. SAÍDAS EM TEXTO E HTML
# Os textos extensos são o conteúdo dos arquivos gerados.
def formatar_tabela_texto(resultados: Sequence[ResultadoCenario]) -> str:
    """Alinha as comparações em colunas para o terminal e o resumo."""
    cabecalho = (
        f"{'v (m/s)':>9} | {'T(v) (s)':>9} | {'base':>19} | "
        f"{'com extensão':>19} | {'muda resultado':>14}"
    )
    linhas = [cabecalho, "-" * len(cabecalho)]
    for item in resultados:
        linhas.append(
            f"{item.velocidade_m_s:>9.3f} | {item.tempo_necessario_s:>9.3f} | "
            f"{item.situacao_base:>19} | {item.situacao_estendida:>19} | "
            f"{('Sim' if item.extensao_muda_resultado else 'Não'):>14}"
        )
    return "\n".join(linhas)


def salvar_resumo(cenario: CenarioTravessia, resultados: Sequence[ResultadoCenario], pasta: Path) -> Path:
    """Calcula indicadores e grava suas interpretações no arquivo TXT."""
    v_base = velocidade_minima_m_s(cenario.tempo_base_s, cenario)
    v_estendida = velocidade_minima_m_s(cenario.tempo_estendido_s, cenario)
    inicio, fim = janela_adicional_s(cenario)
    integral_exata = chegadas_acumuladas_analitica(inicio, fim, cenario)
    integral_numerica = chegadas_acumuladas_numerica(inicio, fim, cenario)
    variacao_alta = tempo_necessario_s(1.1, cenario) - tempo_necessario_s(1.2, cenario)
    variacao_baixa = tempo_necessario_s(0.7, cenario) - tempo_necessario_s(0.8, cenario)
    velocidade_inicio, velocidade_fim = intervalo_velocidades_cenario(cenario)
    integral_tempo = integral_tempo_por_velocidade(velocidade_inicio, velocidade_fim, cenario)
    integral_tempo_num = integral_tempo_numerica(velocidade_inicio, velocidade_fim, cenario)
    tempo_medio = tempo_medio_no_intervalo(velocidade_inicio, velocidade_fim, cenario)
    extensao_otima = extensao_otima_s(cenario)
    custo_otimo = custo_total_extensao(extensao_otima, cenario)
    custo_atual = custo_total_extensao(cenario.extensao_s, cenario)
    curvatura_otima = segunda_derivada_custo_total(extensao_otima, cenario)

    conteudo = f"""PASSAÍ — MODELAGEM MATEMÁTICA DA TRAVESSIA
================================================

Natureza dos dados
------------------
Todos os valores pertencem a um cenário didático simulado.

Parâmetros
----------
Largura da faixa (L): {cenario.largura_m:.3f} m
Tempo de reação/início (r): {cenario.tempo_reacao_s:.3f} s
Tempo-base (G): {cenario.tempo_base_s:.3f} s
Extensão: {cenario.extensao_s:.3f} s
Tempo total com extensão: {cenario.tempo_estendido_s:.3f} s
Velocidade de referência: {cenario.velocidade_referencia_m_s:.3f} m/s
Taxa de chegada: λ(t) = {cenario.taxa_inicial_ped_s:.3f} + {cenario.crescimento_taxa_ped_s2:.3f}t pedestres/s
Índice de acessibilidade: A = {cenario.peso_acessibilidade:.3f}
Taxa de redução do impacto: k = {cenario.decaimento_beneficio_s_inv:.3f} 1/s
Peso do impacto viário: B = {cenario.peso_impacto_viario:.3f}

Modelo de cálculo
-----------------
Tempo necessário: T(v) = r + L/v
Primeira derivada: T'(v) = -L/v²
Segunda derivada: T''(v) = 2L/v³
Limite quando v tende a 0 pela direita: +∞
Limite quando v tende ao infinito: r
Chegadas acumuladas: N(t) = λ₀t + βt²/2
Integral do tempo: ∫T(v)dv = r(b-a) + L ln(b/a)
Custo da extensão: C(e) = A exp(-ke) + Be²
C'(e) = -Ak exp(-ke) + 2Be
C''(e) = Ak² exp(-ke) + 2B

Interpretação
-------------
T'(v) é negativa, portanto velocidades maiores reduzem o tempo necessário.
T''(v) é positiva, portanto a curva é mais sensível nas velocidades menores.
Reduzir de 1.2 para 1.1 m/s acrescenta {variacao_alta:.3f} s.
Reduzir de 0.8 para 0.7 m/s acrescenta {variacao_baixa:.3f} s.

Velocidades-limite
------------------
Com o tempo-base: {v_base:.6f} m/s
Com a extensão: {v_estendida:.6f} m/s
A extensão muda o resultado no intervalo {v_estendida:.6f} <= v < {v_base:.6f} m/s.

Integral da taxa de chegada
---------------------------
Janela adicional: [{inicio:.3f}, {fim:.3f}] s
Integral analítica: {integral_exata:.6f} chegada(s) esperada(s)
Integral numérica: {integral_numerica:.6f} chegada(s) esperada(s)
Erro absoluto: {abs(integral_exata - integral_numerica):.3e}

Tempo médio por integral definida
---------------------------------
Faixa de velocidades: [{velocidade_inicio:.3f}, {velocidade_fim:.3f}] m/s
Integral analítica de T(v): {integral_tempo:.6f} s·(m/s)
Integral numérica de T(v): {integral_tempo_num:.6f} s·(m/s)
Erro absoluto: {abs(integral_tempo - integral_tempo_num):.3e}
Valor médio de T(v): {tempo_medio:.6f} s

Otimização da extensão
----------------------
Extensão configurada: {cenario.extensao_s:.6f} s
Extensão que minimiza C(e): {extensao_otima:.6f} s
Custo total no valor configurado: {custo_atual:.6f}
Custo total no ponto ótimo: {custo_otimo:.6f}
C'(e*) = {derivada_custo_total(extensao_otima, cenario):.3e}
C''(e*) = {curvatura_otima:.6f} > 0, confirmando um mínimo.
Os índices são didáticos e servem para demonstrar a técnica de otimização.

Tabela de cenários
------------------
{formatar_tabela_texto(resultados)}

Conclusão
---------
No cenário simulado, a extensão reduz a velocidade mínima necessária de {v_base:.3f}
para {v_estendida:.3f} m/s. Ela altera o resultado dentro desse intervalo.
Velocidades menores ainda exigem mais tempo. A integral estima {integral_exata:.3f}
chegada(s) na janela adicional. Os resultados subsidiam a análise do protótipo.
Na faixa de velocidades estudada, o tempo médio é {tempo_medio:.3f} s. O modelo
de otimização encontra {extensao_otima:.3f} s como equilíbrio matemático entre
acessibilidade e impacto viário, sujeito à validação posterior com dados reais.
"""
    caminho = pasta / "resumo_analise.txt"
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def salvar_painel_html(graficos: Sequence[Path], pasta: Path) -> Path:
    """Monta uma página local com referências relativas aos seis gráficos."""
    cartoes = "\n".join(
        f'<article><img src="{html.escape(grafico.name)}" alt="{html.escape(grafico.stem)}"></article>'
        for grafico in graficos
    )
    conteudo = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Passaí — Gráficos da modelagem</title>
<style>
body {{ margin: 0; padding: 32px; background: #f3f2f6; color: #25242a; font-family: Arial, sans-serif; }}
h1 {{ margin: 0 0 8px; text-align: center; }}
p {{ margin: 0 0 28px; text-align: center; color: #66636b; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 22px; }}
article {{ background: white; border: 1px solid #dddbe2; border-radius: 14px; overflow: hidden; box-shadow: 0 8px 24px #25242a14; }}
img {{ display: block; width: 100%; height: auto; }}
@media (max-width: 620px) {{ body {{ padding: 14px; }} main {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Passaí — Modelagem matemática</h1>
<p>Limites, derivadas, integrais, valor médio e otimização de cenários simulados</p>
<main>{cartoes}</main>
</body>
</html>
"""
    caminho = pasta / "painel_graficos.html"
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# 7. EXECUÇÃO DO PROGRAMA
# Fluxo: argumentos -> cenário -> cálculos -> arquivos -> navegador.
def pasta_da_aplicacao() -> Path:
    """Localiza a pasta-base mesmo quando o terminal está em outro local."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def executar_analise(
    cenario: CenarioTravessia,
    pasta_saida: Path,
    velocidade_min_grafico: float = 0.4,
    velocidade_max_grafico: float = 1.6,
) -> list[Path]:
    """Coordena os seis gráficos, o resumo e o painel; devolve seus caminhos."""
    if velocidade_min_grafico <= 0 or velocidade_max_grafico <= velocidade_min_grafico:
        raise ValueError("O intervalo de velocidades é inválido.")
    pasta_saida.mkdir(parents=True, exist_ok=True)
    resultados = analisar_cenarios(cenario)
    graficos = [
        grafico_tempo(cenario, pasta_saida, velocidade_min_grafico, velocidade_max_grafico),
        grafico_derivadas(cenario, pasta_saida, velocidade_min_grafico, velocidade_max_grafico),
        grafico_integral(cenario, pasta_saida),
        grafico_mapa(cenario, pasta_saida, velocidade_min_grafico, velocidade_max_grafico),
        grafico_tempo_medio_integral(cenario, pasta_saida),
        grafico_otimizacao_extensao(cenario, pasta_saida),
    ]
    resumo = salvar_resumo(cenario, resultados, pasta_saida)
    painel = salvar_painel_html(graficos, pasta_saida)
    return [*graficos, resumo, painel]


def abrir_painel(caminho: Path) -> bool:
    """Abre o HTML no navegador padrão e informa se a tentativa funcionou."""
    try:
        if os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(str(caminho.resolve()))
            return True
        return webbrowser.open(caminho.resolve().as_uri())
    except OSError:
        return False


def parse_velocidades(valor: str) -> tuple[float, ...]:
    """Converte a lista do terminal, como 1.25;1.0;0.8, em uma tupla."""
    try:
        velocidades = tuple(float(item.strip().replace(",", ".")) for item in valor.split(";"))
    except ValueError as erro:
        raise argparse.ArgumentTypeError("Use velocidades separadas por ponto e vírgula.") from erro
    if not velocidades or any(v <= 0 for v in velocidades):
        raise argparse.ArgumentTypeError("Todas as velocidades devem ser positivas.")
    return velocidades


def criar_parser() -> argparse.ArgumentParser:
    """Define as opções aceitas pelo terminal e seus valores iniciais."""
    # Uma única fonte de valores padrão evita divergência entre terminal e modelo.
    padrao = CenarioTravessia()
    parser = argparse.ArgumentParser(
        description="Modelagem de limites, derivadas, integrais e otimização aplicada ao Passaí.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--largura", type=float, default=padrao.largura_m)
    parser.add_argument("--reacao", type=float, default=padrao.tempo_reacao_s)
    parser.add_argument("--tempo-base", type=float, default=padrao.tempo_base_s)
    parser.add_argument("--extensao", type=float, default=padrao.extensao_s)
    parser.add_argument("--velocidade-referencia", type=float, default=padrao.velocidade_referencia_m_s)
    parser.add_argument("--taxa-inicial", type=float, default=padrao.taxa_inicial_ped_s)
    parser.add_argument("--crescimento-taxa", type=float, default=padrao.crescimento_taxa_ped_s2)
    parser.add_argument("--peso-acessibilidade", type=float, default=padrao.peso_acessibilidade)
    parser.add_argument("--decaimento-beneficio", type=float, default=padrao.decaimento_beneficio_s_inv)
    parser.add_argument("--peso-impacto-viario", type=float, default=padrao.peso_impacto_viario)
    parser.add_argument(
        "--velocidades", type=parse_velocidades, default=padrao.velocidades_m_s,
        help='Lista separada por ponto e vírgula, por exemplo "1.25;1.0;0.8;0.6".',
    )
    parser.add_argument("--velocidade-min-grafico", type=float, default=0.4)
    parser.add_argument("--velocidade-max-grafico", type=float, default=1.6)
    parser.add_argument("--pasta-saida", type=Path, default=pasta_da_aplicacao() / "resultados")
    parser.add_argument("--nao-exibir", action="store_true", help="Não abre o painel no navegador.")
    return parser


def main() -> None:
    """Executa o fluxo principal ao iniciar este arquivo."""
    # Preparação do terminal e leitura das opções informadas pelo usuário.
    configurar_terminal_utf8()
    if sys.version_info < (3, 10):
        raise SystemExit("Este projeto requer Python 3.10 ou superior.")
    argumentos = criar_parser().parse_args()
    # A criação do cenário também valida seus parâmetros físicos.
    cenario = CenarioTravessia(
        largura_m=argumentos.largura,
        tempo_reacao_s=argumentos.reacao,
        tempo_base_s=argumentos.tempo_base,
        extensao_s=argumentos.extensao,
        velocidade_referencia_m_s=argumentos.velocidade_referencia,
        taxa_inicial_ped_s=argumentos.taxa_inicial,
        crescimento_taxa_ped_s2=argumentos.crescimento_taxa,
        peso_acessibilidade=argumentos.peso_acessibilidade,
        decaimento_beneficio_s_inv=argumentos.decaimento_beneficio,
        peso_impacto_viario=argumentos.peso_impacto_viario,
        velocidades_m_s=argumentos.velocidades,
    )
    # Geração dos seis gráficos e dos arquivos de consulta.
    arquivos = executar_analise(
        cenario,
        argumentos.pasta_saida,
        argumentos.velocidade_min_grafico,
        argumentos.velocidade_max_grafico,
    )
    # Apresentação dos resultados e abertura opcional do painel.
    print("\nPASSAÍ — MODELAGEM MATEMÁTICA")
    print("=" * 72)
    print(formatar_tabela_texto(analisar_cenarios(cenario)))
    print("\nArquivos gerados:")
    for arquivo in arquivos:
        print(f"- {arquivo.resolve()}")
    print("\nTodos os parâmetros pertencem a uma simulação didática.")
    if not argumentos.nao_exibir:
        painel = next(arquivo for arquivo in arquivos if arquivo.suffix == ".html")
        if abrir_painel(painel):
            print("O painel com os gráficos foi aberto no navegador.")
        else:
            print(f"Abra manualmente: {painel.resolve()}")


if __name__ == "__main__":
    main()


