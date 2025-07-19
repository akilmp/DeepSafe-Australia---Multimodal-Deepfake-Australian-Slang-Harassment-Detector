
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from deepsafe_sydney.scrape_slang import AUSSIE_SLURS


def test_slur_list():
    assert len(AUSSIE_SLURS) >= 6

