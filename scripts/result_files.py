"""Select one response file per model for graph generation."""

from pathlib import Path
import re


def select_response_files(results_dir):
    """Prefer full results; otherwise use the largest numbered subset."""
    selected = {}
    ranks = {}
    for path in sorted(Path(results_dir).glob('*.csv')):
        if not path.is_file():
            continue
        match = re.fullmatch(r'(.+)_responses(?:_subset_(\d+))?\.csv', path.name)
        if match is None:
            continue
        model, subset_size = match.groups()
        rank = (subset_size is None, int(subset_size or 0))
        if model not in ranks or rank > ranks[model]:
            selected[model] = path
            ranks[model] = rank
    return dict(sorted(selected.items()))
