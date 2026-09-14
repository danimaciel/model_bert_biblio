import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class AppFlowTests(unittest.TestCase):
    def test_demo_three_modes_and_filters(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(app.metric[0].value, '18')
        for mode in ['Descobrir categorias', 'Classificar por categorias', 'Combinar os dois']:
            next(x for x in app.radio if x.label == 'Modo de análise').set_value(mode).run()
            next(x for x in app.button if x.label == 'Analisar categorias').click().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 0, str(app.error))
            self.assertIn('analysis', app.session_state)
        next(x for x in app.multiselect if x.label.startswith('Filtrar anos')).set_value([2024]).run()
        self.assertEqual(app.metric[0].value, '3')
        self.assertNotIn('analysis', app.session_state)
        self.assertEqual(len(app.exception), 0, str(app.exception))
