import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("src"))

from deepsafe_sydney.label_tool import parse_env_file


def test_parse_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("PRODIGY=true\n# comment\nOTHER=val")
    env = parse_env_file(env_file)
    assert env["PRODIGY"] == "true"
    assert env["OTHER"] == "val"

