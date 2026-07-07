import csv
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import Dataset


class GPQADataset(Dataset):
    DEFAULT_PATHS = (
        Path(__file__).resolve().parent / "data" / "gpqa_extended.csv",
        Path(__file__).resolve().parents[2] / "data" / "gpqa_extended.csv",
    )
    DOMAIN_COLUMN = "High-level domain"
    ALLOWED_DOMAINS = frozenset({"physics", "math"})

    def __init__(
        self,
        csv_path: str | Path | None = None,
        *,
        columns: list[str] | None = None,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else self._default_path()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"GPQA CSV file not found: {self.csv_path}")

        records: list[dict[str, Any]] = []
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"GPQA CSV file has no header: {self.csv_path}")
            if self.DOMAIN_COLUMN not in reader.fieldnames:
                raise ValueError(f"GPQA CSV file is missing column: {self.DOMAIN_COLUMN}")

            self.column_names = columns if columns is not None else list(reader.fieldnames)
            missing_columns = set(self.column_names) - set(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"GPQA CSV file is missing columns: {missing}")

            for row in reader:
                if row[self.DOMAIN_COLUMN].casefold() in self.ALLOWED_DOMAINS:
                    records.append(
                        {column: row[column] for column in self.column_names}
                        if columns is not None
                        else row
                    )

        self.records = records
        self.transform = transform

    @classmethod
    def _default_path(cls) -> Path:
        for path in cls.DEFAULT_PATHS:
            if path.exists():
                return path
        return cls.DEFAULT_PATHS[-1]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("GPQADataset index out of range")

        row = self.records[idx]
        if self.transform is not None:
            return self.transform(row)
        return row
