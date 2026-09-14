"""Importação e indicadores. Ausências nunca são substituídas por zero."""
import io
import re
import unicodedata
from datetime import date

import pandas as pd

FIELDS = {
    'title': ('Título', ['title', 'titulo', 'article title', 'ti']),
    'abstract': ('Resumo', ['abstract', 'resumo', 'ab']),
    'authors': ('Autores', ['authors', 'autores', 'autoria', 'author full names', 'au']),
    'year': ('Ano', ['year', 'ano', 'publication year', 'py']),
    'source': ('Periódico / fonte', ['source title', 'source', 'periodico', 'journal', 'so']),
    'citations': ('Citações', ['cited by', 'citations', 'citacoes', 'times cited, all databases', 'tc']),
    'doi': ('DOI', ['doi', 'di']),
    'keywords': ('Palavras-chave', ['author keywords', 'keywords', 'palavras-chave', 'palavras chave', 'de']),
    'type': ('Tipo de documento', ['document type', 'type', 'tipo', 'dt']),
}


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)) if not unicodedata.combining(c)).strip().lower()


def read_table(content, filename):
    if filename.lower().endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(content), dtype=str).fillna('')
    for encoding in ('utf-8-sig', 'cp1252'):
        try:
            return pd.read_csv(io.StringIO(content.decode(encoding)), sep=None, engine='python', dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError('Não foi possível identificar a codificação do arquivo.')


def guess_mapping(columns):
    lookup = {normalized(c): c for c in columns}
    return {key: next((lookup[a] for a in aliases if a in lookup), None) for key, (_, aliases) in FIELDS.items()}


def prepare(raw, mapping, separator=';', deduplicate=True, keyword_separator=';'):
    df = pd.DataFrame(index=raw.index)
    for key in FIELDS:
        col = mapping.get(key)
        df[key] = raw[col].fillna('').astype(str).str.strip() if col else ''
    df['record_id'] = ['D%06d' % (i + 1) for i in range(len(df))]
    invalid = {}
    for key in ('year', 'citations'):
        numeric = pd.to_numeric(df[key], errors='coerce')
        valid = numeric.ge(0) & numeric.mod(1).eq(0)
        if key == 'year':
            valid &= numeric.between(1000, date.today().year + 1)
        invalid[key] = int((df[key].ne('') & ~valid).sum())
        df[key] = numeric.where(valid)
    doi = df.doi.str.lower().str.replace(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', regex=True)
    # Sem DOI, só deduplicar títulos com ano; não fundir versões com DOIs distintos.
    keys = pd.Series(['unique:' + x for x in df.record_id], index=df.index)
    fallback = doi.eq('') & df.title.ne('') & df.year.notna()
    keys.loc[fallback] = 'title:' + df.loc[fallback, 'title'].map(normalized) + ':' + df.loc[fallback, 'year'].astype(str)
    keys.loc[doi.ne('')] = 'doi:' + doi[doi.ne('')]
    duplicate = keys.duplicated()
    removed = int(duplicate.sum()) if deduplicate else 0
    if deduplicate:
        df = df.loc[~duplicate].copy()
    df['author_list'] = df.authors.map(lambda x: list(dict.fromkeys(a.strip() for a in x.split(separator) if a.strip())))
    df['author_count'] = df.author_list.map(lambda x: len(x) if x else float('nan'))
    df['keyword_list'] = df.keywords.map(lambda x: list(dict.fromkeys(k.strip().casefold() for k in x.split(keyword_separator) if k.strip())))
    df['text'] = (df.title + '. ' + df.abstract).str.strip('. ')
    return df.reset_index(drop=True), {'input_records': len(raw), 'removed_duplicates': removed, 'invalid_numeric': invalid}


def descriptive(df):
    rows = []
    for col, label in [('citations', 'Citações'), ('author_count', 'Autores por documento')]:
        s = df[col].dropna()
        rows.append({'Indicador': label, 'N válido': len(s), 'Ausentes': int(df[col].isna().sum()),
                     'Média': s.mean(), 'Mediana': s.median(), 'Desvio padrão': s.std(),
                     'Mínimo': s.min(), 'P25': s.quantile(.25), 'P75': s.quantile(.75),
                     'P90': s.quantile(.9), 'Máximo': s.max()})
    return pd.DataFrame(rows)


def author_ranking(df):
    s = df.author_list.explode().dropna()
    return s.value_counts().rename_axis('Autor').reset_index(name='Documentos')


def keyword_ranking(df):
    return df.keyword_list.explode().dropna().value_counts().rename_axis('Palavra-chave').reset_index(name='Documentos')
