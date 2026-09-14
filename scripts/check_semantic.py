"""Teste de integração opcional: baixa o modelo e usa apenas a base fictícia."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sentence_transformers import SentenceTransformer
from bibliometria.data import read_table, guess_mapping, prepare
from bibliometria.topics import MODEL, analyze

raw = read_table((ROOT / 'examples/base_demonstracao.csv').read_bytes(), 'demo.csv')
df, _ = prepare(raw, guess_mapping(raw.columns))
encoder = SentenceTransformer(MODEL, cache_folder=str(ROOT / '.models'))
inputs = pd.DataFrame([{'name': 'Manejo florestal', 'definition': 'Exploração sustentável de madeira, manejo de florestas e certificação florestal.'}])
summary = {}
for mode in ['Descobrir categorias', 'Classificar por categorias', 'Combinar os dois']:
    cats, links, notes = analyze(df, inputs, mode, 'SBERT + BERTopic', threshold=.5, min_size=3, encoder=encoder)
    assert set(links.record_id).issubset(set(df.record_id))
    assert set(links.category_id).issubset(set(cats.category_id))
    if mode == 'Classificar por categorias':
        assert links.record_id.nunique() > 0
    summary[mode] = {'categories': len(cats), 'assigned_documents': links.record_id.nunique(), 'notes': notes}
print(json.dumps(summary, ensure_ascii=True, indent=2))
