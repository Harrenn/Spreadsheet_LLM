from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "branch_cleaner_colab.ipynb",
    ROOT / "notebooks" / "branch_cleaner_vast_ai.ipynb",
]


class FakeLlm:
    def create_chat_completion(self, messages, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "barangay": "Barangay Poblacion",
                                "city_municipality": "Bangued",
                                "province": "Abra",
                                "region": "WRONG LLM REGION",
                                "zip_code": None,
                                "country": "Philippines",
                                "island_group": "Luzon",
                            }
                        )
                    }
                }
            ]
        }


def load_helper_namespace(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    helper_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def load_psgc_region_map" in "".join(cell.get("source", []))
    )
    namespace = {
        "__name__": "notebook_test",
        "INPUT_PATH": "",
        "PSGC_PATH": "",
        "SHEET_NAME": "branch",
        "PROCESS_ROW_LIMIT": None,
        "LONGITUDE_COLUMN": "longitude",
        "LATITUDE_COLUMN": "latitude",
        "FULL_ADDRESS_COLUMN": "full_address",
        "GOOGLE_GEOCODING_API_KEY": "",
        "GGUF_MODEL_PATH": "",
        "MAX_TOKENS": 180,
        "LLM": None,
    }
    exec(compile(helper_source, str(path), "exec"), namespace)
    return namespace


def write_psgc_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["province_name,region_name"]
    lines.extend(f'"{province}","{region}"' for province, region in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class BranchCleanerNotebookTests(unittest.TestCase):
    def test_psgc_path_is_required_and_ambiguous_mappings_fail(self) -> None:
        for notebook_path in NOTEBOOKS:
            with self.subTest(notebook=notebook_path.name), tempfile.TemporaryDirectory() as tmp:
                namespace = load_helper_namespace(notebook_path)
                with self.assertRaisesRegex(ValueError, "Set PSGC_PATH"):
                    namespace["load_psgc_region_map"]("")

                csv_path = Path(tmp) / "ambiguous.csv"
                write_psgc_csv(
                    csv_path,
                    [
                        ("Abra", "Cordillera Administrative Region (CAR)"),
                        ("Abra", "Region I (Ilocos Region)"),
                    ],
                )
                with self.assertRaisesRegex(ValueError, "multiple regions"):
                    namespace["load_psgc_region_map"](csv_path)

    def test_llm_region_is_ignored_and_region2_comes_from_psgc(self) -> None:
        for notebook_path in NOTEBOOKS:
            with self.subTest(notebook=notebook_path.name), tempfile.TemporaryDirectory() as tmp:
                namespace = load_helper_namespace(notebook_path)
                csv_path = Path(tmp) / "psgc.csv"
                write_psgc_csv(
                    csv_path,
                    [
                        ("Abra", "Cordillera Administrative Region (CAR)"),
                        ("Samar", "Region VIII (Eastern Visayas)"),
                        ("Davao del Sur", "Region XI (Davao Region)"),
                    ],
                )
                region_map = namespace["load_psgc_region_map"](csv_path)

                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "branch"
                sheet.append(["full_address"])
                sheet.append(["Bangued, Abra"])

                summary = namespace["parse_full_address"](
                    workbook,
                    sheet,
                    "xlsx",
                    llm=FakeLlm(),
                    psgc_region_map=region_map,
                )
                headers = [cell.value for cell in sheet[1]]
                values = [cell.value for cell in sheet[2]]
                row = dict(zip(headers, values, strict=True))

                self.assertNotIn("region", namespace["ADDRESS_FIELDS"])
                self.assertNotIn("province, region", namespace["build_prompt"]("Address"))
                self.assertEqual(row["province2"], "Abra")
                self.assertEqual(row["region2"], "CAR")
                self.assertEqual(summary["region_rows_matched"], 1)
                self.assertEqual(summary["region_rows_unmatched"], 0)

                self.assertEqual(namespace["psgc_region_for_province"]("Western Samar", region_map), "Region VIII")
                self.assertEqual(namespace["psgc_region_for_province"]("Davao Del Sur", region_map), "Region XI")
                self.assertEqual(namespace["psgc_region_for_province"]("Metro Manila", region_map), "NCR")
                self.assertIsNone(namespace["psgc_region_for_province"]("Eastern Cape", region_map))


if __name__ == "__main__":
    unittest.main()
