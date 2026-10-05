import tempfile
import unittest
from pathlib import Path

from scripts.result_files import select_response_files


class ResponseFileSelectionTests(unittest.TestCase):
    def select(self, filenames):
        with tempfile.TemporaryDirectory() as directory:
            for filename in filenames:
                Path(directory, filename).touch()
            return {
                model: path.name
                for model, path in select_response_files(directory).items()
            }

    def test_largest_subset_uses_numeric_size(self):
        self.assertEqual(
            self.select(['codon-gpt-5.4_responses_subset_40.csv',
                         'codon-gpt-5.4_responses_subset_100.csv']),
            {'codon-gpt-5.4': 'codon-gpt-5.4_responses_subset_100.csv'},
        )

    def test_full_results_take_precedence(self):
        self.assertEqual(
            self.select(['model_responses.csv', 'model_responses_subset_100.csv']),
            {'model': 'model_responses.csv'},
        )

    def test_selection_is_independent_per_model(self):
        self.assertEqual(
            self.select(['a_responses_subset_40.csv', 'a_responses_subset_100.csv',
                         'b_responses_subset_40.csv', 'c_responses.csv']),
            {'a': 'a_responses_subset_100.csv',
             'b': 'b_responses_subset_40.csv', 'c': 'c_responses.csv'},
        )

    def test_unrelated_and_malformed_files_are_ignored(self):
        self.assertEqual(
            self.select(['compiled.csv', 'model_responses_subset_bad.csv']), {},
        )


if __name__ == '__main__':
    unittest.main()
