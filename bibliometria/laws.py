"""Comparações descritivas com leis clássicas; não são testes de aderência."""
from collections import Counter
import re
import unicodedata

import numpy as np
import pandas as pd

from .data import author_ranking


def zipf(df):
    counts = Counter()
    covered = 0
    for text in df.text:
        # Letras Unicode, incluindo palavras de uma letra; mantém stopwords.
        tokens = re.findall(r'[^\W\d_]+', unicodedata.normalize('NFC', text).casefold())
        covered += bool(tokens)
        counts.update(tokens)
    table = pd.DataFrame(sorted(counts.items(), key=lambda x: (-x[1], x[0])), columns=['Palavra', 'Ocorrências'])
    table.insert(0, 'Posição', np.arange(1, len(table) + 1))
    total = sum(counts.values())
    harmonic = sum(1 / r for r in range(1, len(table) + 1))
    table['Zipf esperado (s=1)'] = total / harmonic / table['Posição'] if harmonic else pd.Series(dtype=float)
    return table, {'documentos_com_tokens': covered, 'documentos_sem_tokens': len(df) - covered,
                   'ocorrencias': total, 'vocabulario': len(table), 'expoente_fixo': 1}


def lotka(df):
    authors = author_ranking(df)
    maximum = int(authors.Documentos.max()) if not authors.empty else 0
    x = np.arange(1, maximum + 1)
    observed = authors.Documentos.value_counts().reindex(x, fill_value=0)
    table = pd.DataFrame({'Documentos por autor': x, 'Autores observados': observed.to_numpy()})
    # Distribuição discreta clássica no suporte infinito: p(x)=6/(pi² x²).
    table['Lotka esperado (a=2)'] = len(authors) * 6 / np.pi**2 / x.astype(float)**2
    tail = max(0., len(authors) - float(table['Lotka esperado (a=2)'].sum()))
    return table, {'autores': len(authors), 'documentos_sem_autoria': int(df.author_list.map(len).eq(0).sum()),
                   'expoente_fixo': 2, 'autores_esperados_acima_maximo_observado': tail,
                   'contagem': 'integral; nomes sem desambiguação'}


def bradford(df):
    counts = df.loc[df.source.ne(''), 'source'].value_counts()
    sources = counts.rename_axis('Fonte').reset_index(name='Documentos').sort_values(
        ['Documentos', 'Fonte'], ascending=[False, True]).reset_index(drop=True)
    sources.insert(0, 'Posição', np.arange(1, len(sources) + 1))
    sources['Documentos acumulados'] = sources.Documentos.cumsum()
    sources['Zona'] = pd.Series(pd.NA, index=sources.index, dtype='Int64')
    columns = ['Zona', 'Fontes', 'Documentos', 'Percentual de documentos', 'Desvio da meta (documentos)',
               'Fontes relativas ao núcleo', 'Razão para zona anterior', 'Fontes esperadas (1:b:b²)']
    info = {'documentos_com_fonte': int(sources.Documentos.sum()),
            'documentos_sem_fonte': int(df.source.eq('').sum()), 'fontes': len(sources)}
    if len(sources) < 3:
        return sources, pd.DataFrame(columns=columns), {**info, 'status': 'Indisponível: são necessárias ao menos três fontes.'}
    cumulative = sources['Documentos acumulados'].to_numpy()
    target = cumulative[-1] / 3
    # Cortes sequenciais mais próximos de 1/3 e 2/3; reserva uma fonte por zona.
    cut1 = int(np.argmin(np.abs(cumulative[:len(sources)-2] - target))) + 1
    cut2 = cut1 + int(np.argmin(np.abs(cumulative[cut1:-1] - 2 * target))) + 1
    sources['Zona'] = [1] * cut1 + [2] * (cut2-cut1) + [3] * (len(sources)-cut2)
    zones = sources.groupby('Zona').agg(Fontes=('Fonte', 'size'), Documentos=('Documentos', 'sum')).reset_index()
    zones['Percentual de documentos'] = 100 * zones.Documentos / cumulative[-1]
    zones['Desvio da meta (documentos)'] = zones.Documentos - target
    zones['Fontes relativas ao núcleo'] = zones.Fontes / zones.Fontes.iloc[0]
    zones['Razão para zona anterior'] = zones.Fontes / zones.Fontes.shift(1)
    factor = float(np.sqrt(zones.Fontes.iloc[2] / zones.Fontes.iloc[0]))
    zones['Fontes esperadas (1:b:b²)'] = zones.Fontes.iloc[0] * factor ** np.arange(3)
    return sources, zones[columns], {**info, 'status': 'Comparação descritiva', 'meta_documentos_por_zona': target,
                                   'multiplicador_b': factor}


def bibliometric_laws(df):
    z, zi = zipf(df)
    l, li = lotka(df)
    b, zones, bi = bradford(df)
    return {'zipf': z, 'lotka': l, 'bradford_fontes': b, 'bradford_zonas': zones}, {
        'zipf': zi, 'lotka': li, 'bradford': bi,
        'interpretacao': 'Comparações descritivas com expoentes clássicos fixos. Não há teste de aderência nem confirmação automática das leis.'}
