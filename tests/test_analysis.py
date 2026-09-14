import io
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from bibliometria.data import prepare, descriptive, read_table, guess_mapping, keyword_ranking
from bibliometria.topics import analyze, validate_categories
from bibliometria.export import export_zip


class AnalysisTests(unittest.TestCase):
    def test_author_and_keyword_separators(self):
        for delimiter, filename in [(';', 'base_demonstracao.csv'), (',', 'base_demonstracao_virgula.csv')]:
            raw = read_table(Path('examples', filename).read_bytes(), filename)
            df, _ = prepare(raw, guess_mapping(raw.columns), separator=delimiter, keyword_separator=delimiter)
            self.assertEqual(len(df), 18)
            self.assertEqual(df.author_list.iloc[0], ['Silva A.', 'Costa B.'])
            self.assertEqual(df.keyword_list.iloc[0], ['manejo', 'madeira'])
        raw = pd.DataFrame({'autoria': ['Silva, A.; Costa, B.'], 'palavras-chave': ['Manejo, manejo, Floresta, ']})
        df, _ = prepare(raw, guess_mapping(raw.columns), separator=';', keyword_separator=',')
        self.assertEqual(df.author_count.iloc[0], 2)
        self.assertEqual(df.keyword_list.iloc[0], ['manejo', 'floresta'])
        self.assertEqual(keyword_ranking(df).Documentos.sum(), 2)

    def test_missing_duplicate_and_invalid(self):
        raw = pd.DataFrame({'title': ['A', 'A', 'B', 'C'], 'doi': ['https://doi.org/10/a', '10/A', '', ''], 'year': ['2020', '2020', 'bad', '2024'], 'citations': ['', '2', '-1', '0'], 'authors': ['Silva, A.; Costa, B.', '', '', 'X']})
        df, quality = prepare(raw, guess_mapping(raw.columns))
        self.assertEqual(len(df), 3)
        self.assertEqual(quality['removed_duplicates'], 1)
        self.assertEqual(df.author_count.iloc[0], 2)
        self.assertTrue(pd.isna(df.citations.iloc[0]))
        self.assertEqual(descriptive(df).iloc[0]['N válido'], 1)
        self.assertTrue(pd.isna(df.year.iloc[1]))

    def test_real_lda_and_export(self):
        raw = read_table(Path('examples/base_demonstracao.csv').read_bytes(), 'demo.csv')
        df, quality = prepare(raw, guess_mapping(raw.columns))
        cats, links, notes = analyze(df, pd.DataFrame(columns=['name', 'definition']), 'Descobrir categorias', 'LDA + classificação lexical', n_topics=3)
        self.assertGreaterEqual(len(cats), 2)
        self.assertEqual(links.record_id.nunique(), len(df))
        df.loc[0, 'title'] = '<script>alert(1)</script>'
        payload = export_zip(df, cats, links, {'quality': quality})
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            html = archive.read('relatorio.html').decode()
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertIn('associacoes.csv', archive.namelist())

    def test_semantic_multilabel_and_residual(self):
        class Encoder:
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.] if 'floresta' in t else [0., 1.] for t in texts])
        raw = pd.DataFrame({'title': ['floresta com manejo sustentável', 'política pública territorial']})
        df, _ = prepare(raw, guess_mapping(raw.columns))
        inputs = pd.DataFrame([{'name': 'floresta A', 'definition': 'floresta'}, {'name': 'floresta B', 'definition': 'floresta'}])
        cats, links, notes = analyze(df, inputs, 'Combinar os dois', 'SBERT + BERTopic', encoder=Encoder())
        self.assertEqual(len(links), 2)
        self.assertEqual(links.record_id.nunique(), 1)
        self.assertTrue(any('menos de três' in n for n in notes))

    def test_empty_categories_validation(self):
        with self.assertRaises(ValueError):
            validate_categories(pd.DataFrame([{'name': 'Tema', 'definition': ''}]))


if __name__ == '__main__':
    unittest.main()
