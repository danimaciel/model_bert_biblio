import io
import json
import unittest
import zipfile

import numpy as np
import pandas as pd

from bibliometria.data import prepare, guess_mapping
from bibliometria.laws import zipf, lotka, bradford, bibliometric_laws
from bibliometria.export import export_zip
from bibliometria.topics import CAT_COLUMNS, ASSIGN_COLUMNS


def corpus(rows):
    raw = pd.DataFrame(rows)
    return prepare(raw, guess_mapping(raw.columns), deduplicate=False)[0]


class LawTests(unittest.TestCase):
    def test_zipf_counts_occurrences_and_normalizes(self):
        df = corpus([{'title': 'A a Á árvore', 'abstract': 'árvore a 123'}, {'title': 'a\u0301'}])
        table, info = zipf(df)
        self.assertEqual(dict(zip(table.Palavra, table['Ocorrências'])), {'a': 3, 'á': 2, 'árvore': 2})
        self.assertEqual(info['ocorrencias'], 7)
        self.assertAlmostEqual(table['Zipf esperado (s=1)'].sum(), 7)
        self.assertAlmostEqual(table['Zipf esperado (s=1)'].iloc[0] / table['Zipf esperado (s=1)'].iloc[2], 3)

    def test_lotka_full_credit_gaps_and_infinite_tail(self):
        df = corpus([{'authors': 'A; A; B'}, {'authors': 'A'}, {'authors': 'A'}, {'authors': ''}])
        table, info = lotka(df)
        self.assertEqual(table['Autores observados'].tolist(), [1, 0, 1])
        self.assertEqual(info['documentos_sem_autoria'], 1)
        self.assertAlmostEqual(table['Lotka esperado (a=2)'].iloc[0], 12 / np.pi**2)
        self.assertAlmostEqual(table['Lotka esperado (a=2)'].sum() + info['autores_esperados_acima_maximo_observado'], 2)

    def test_bradford_known_geometric_zones(self):
        rows = [{'source': name} for name, count in [('A', 4), ('B', 2), ('C', 2), ('D', 1), ('E', 1), ('F', 1), ('G', 1)] for _ in range(count)]
        sources, zones, info = bradford(corpus(rows + [{'source': ''}]))
        self.assertEqual(zones.Fontes.tolist(), [1, 2, 4])
        self.assertEqual(zones.Documentos.tolist(), [4, 4, 4])
        self.assertEqual(info['multiplicador_b'], 2)
        self.assertEqual(info['documentos_sem_fonte'], 1)
        self.assertEqual(sources.Documentos.sum(), 12)
        np.testing.assert_allclose(zones['Fontes esperadas (1:b:b²)'], [1, 2, 4])

    def test_bradford_concentration_ties_and_small_samples(self):
        df = corpus([{'source': 'A'}] * 10 + [{'source': 'C'}, {'source': 'B'}])
        sources, zones, _ = bradford(df)
        self.assertEqual(sources.Fonte.tolist(), ['A', 'B', 'C'])
        self.assertEqual(zones.Documentos.tolist(), [10, 1, 1])
        self.assertEqual(zones['Desvio da meta (documentos)'].tolist(), [6, -3, -3])
        _, small, info = bradford(df.loc[df.source.ne('C')])
        self.assertTrue(small.empty)
        self.assertIn('Indisponível', info['status'])

    def test_missing_data_and_filtered_corpus(self):
        tables, info = bibliometric_laws(corpus([{'title': '', 'authors': '', 'source': ''}]))
        self.assertTrue(all(t.empty for t in tables.values()))
        self.assertEqual(info['zipf']['documentos_sem_tokens'], 1)
        df = corpus([{'title': 'floresta', 'year': '2020'}, {'title': 'mar mar', 'year': '2021'}])
        tables, _ = bibliometric_laws(df.loc[df.year.eq(2021)])
        self.assertEqual(tables['zipf'].Palavra.tolist(), ['mar'])
        self.assertEqual(tables['zipf']['Ocorrências'].tolist(), [2])

    def test_export_includes_laws_formulas_and_sources(self):
        df = corpus([{'title': '<script>alert(1)</script>', 'authors': 'A', 'source': '<b>X</b>'}])
        result = export_zip(df, pd.DataFrame(columns=CAT_COLUMNS), pd.DataFrame(columns=ASSIGN_COLUMNS), {})
        with zipfile.ZipFile(io.BytesIO(result)) as archive:
            for name in ['zipf.csv', 'lotka.csv', 'bradford_fontes.csv', 'bradford_zonas.csv', 'metodologia_referencias.md']:
                self.assertIn(name, archive.namelist())
            metadata = json.loads(archive.read('metodologia.json'))
            self.assertEqual(metadata['bibliometric_laws']['lotka']['autores'], 1)
            self.assertTrue(any(ref['id'] == 'LOTKA' for ref in metadata['references']))
            html = archive.read('relatorio.html').decode()
            self.assertIn('https://www.jstor.org/stable/24529203', html)
            self.assertIn('6/(π²x²)', html)
            self.assertNotIn('<b>X</b>', html)
            self.assertNotIn('<script>alert(1)</script>', html)
