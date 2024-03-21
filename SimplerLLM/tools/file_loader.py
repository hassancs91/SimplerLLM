import csv
import os
from pydantic import BaseModel
from typing import Optional, List


class CSVDocument(BaseModel):
    file_size: Optional[int] = None
    row_count: int
    column_count: int
    total_fields: int
    content: List[List[str]]  # This represents the CSV data as a list of rows
    title: Optional[str] = None
    url_or_path: Optional[str] = None


def read_csv_file(file_path: str) -> CSVDocument:
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    file_size = os.path.getsize(file_path)
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    total_fields = sum(len(row) for row in rows)

    return CSVDocument(
        file_size=file_size,
        row_count=row_count,
        column_count=column_count,
        total_fields=total_fields,
        content=rows,
        url_or_path=file_path,
    )
