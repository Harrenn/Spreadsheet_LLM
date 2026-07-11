from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_notebooks_are_clean_and_environment_configurable(self) -> None:
        notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        combined_source = ""
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
            combined_source += source
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])
        self.assertIn("SPREADSHEET_LLM_INPUT_PATH", combined_source)
        self.assertIn("GGUF_MODEL_PATH", combined_source)
        self.assertNotIn("/root/.cache/", combined_source)

    def test_no_models_or_real_spreadsheets_are_tracked_candidates(self) -> None:
        files = [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and "working" not in path.parts and "examples" not in path.parts
        ]
        self.assertFalse(any(path.suffix.lower() in {".gguf", ".xlsx", ".xls"} for path in files))


if __name__ == "__main__":
    unittest.main()
