import io
import json
import zipfile
from html import escape
from datetime import datetime, timezone

import pandas as pd

from .data import descriptive, author_ranking, keyword_ranking


def export_zip(df, categories, assignments, metadata, figures=None):
    """Relatório autossuficiente; texto de entrada sempre escapado no HTML."""
    output = io.BytesIO()
    docs = df.drop(columns=['author_list', 'keyword_list', 'text'], errors='ignore')
    linked = assignments.merge(categories, on='category_id', how='left').merge(docs, on='record_id', how='left') if not assignments.empty else assignments
    payload = {**metadata, 'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'documents': len(df)}
    tables = {'documentos': docs, 'categorias': categories, 'associacoes': linked,
              'estatisticas': descriptive(df), 'autores': author_ranking(df), 'palavras_chave': keyword_ranking(df)}
    tables['sem_categoria'] = docs.loc[~docs.record_id.isin(assignments.record_id)]
    if not assignments.empty:
        summary = linked.drop_duplicates(['record_id', 'name']).groupby('name').record_id.nunique().reset_index(name='Documentos')
        summary['Percentual da base'] = 100 * summary.Documentos / len(df)
        tables['resumo_categorias'] = summary
    body = '<h1>Observatório bibliométrico</h1><p>Análise exploratória. Categorias automáticas são propostas para revisão.</p>'
    body += '<h2>Método e parâmetros</h2><pre>' + escape(json.dumps(payload, ensure_ascii=False, indent=2, default=str)) + '</pre>'
    for name, table in tables.items():
        body += '<h2>' + escape(name.capitalize()) + '</h2>' + table.to_html(index=False, escape=True, na_rep='Não disponível')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, table in tables.items():
            archive.writestr(name + '.csv', table.to_csv(index=False).encode('utf-8-sig'))
        archive.writestr('metodologia.json', json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        for i, (name, fig) in enumerate((figures or {}).items()):
            html = fig.to_html(full_html=True, include_plotlyjs=True)
            archive.writestr('graficos/' + name + '.html', html)
            body += fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
        archive.writestr('relatorio.html', '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Análise bibliométrica</title><style>body{font:15px system-ui;margin:40px;color:#172b4d}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>' + body + '</html>')
    return output.getvalue()
