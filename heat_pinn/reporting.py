from __future__ import annotations

"""Βοηθητικά για αναφορές σε Excel"""

import re
from pathlib import Path

import pandas as pd


def _sheet_name(title: str) -> str:
    """Καθαρίζει τον τίτλο ώστε να είναι έγκυρο Excel sheet name"""

    cleaned = re.sub(r"[:\\/?*\[\]]", "_", title).strip()
    return cleaned[:31] or "Sheet1"


def write_dataframe_report(
    output_path: Path,
    title: str,
    df: pd.DataFrame,
) -> None:
    """Γράφει DataFrame σε αρχείο Excel"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=_sheet_name(title))


def write_mapping_report(
    output_path: Path,
    title: str,
    mapping: dict[str, object],
) -> None:
    """Γράφει dictionary report ως μία γραμμή Excel"""

    write_dataframe_report(output_path, title, pd.DataFrame([mapping]))
