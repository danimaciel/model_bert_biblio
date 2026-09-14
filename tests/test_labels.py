import unittest
from unittest.mock import patch, Mock

from bibliometria.labels import suggest_labels


class LabelTests(unittest.TestCase):
    @patch('bibliometria.labels.requests.post')
    def test_validated_response_and_nonstored_request(self, post):
        post.return_value = Mock(ok=True)
        post.return_value.json.return_value = {'status': 'completed', 'output': [{'content': [{'type': 'output_text', 'text': '{"categories":[{"category_id":"E001","name":"Manejo florestal","definition":"Gestão de florestas","rationale":"Documento D000001"}]}'}]}]}
        labels = suggest_labels([{'category_id': 'E001'}], 'test-key', 'test-model')
        self.assertEqual(labels[0]['name'], 'Manejo florestal')
        self.assertFalse(post.call_args.kwargs['json']['store'])
        self.assertNotIn('test-key', str(post.call_args.kwargs['json']))
        with self.assertRaises(ValueError):
            suggest_labels([{'category_id': 'E002'}], 'test-key', 'test-model')
