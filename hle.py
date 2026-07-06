from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
from torch.utils.data import Dataset


class HLEDataset(Dataset):
    DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "hle.parquet"

    def __init__(
        self,
        parquet_path: str | Path | None = None,
        *,
        columns: list[str] | None = None,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path) if parquet_path is not None else self.DEFAULT_PATH
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"HLE parquet file not found: {self.parquet_path}")

        self.table = pq.read_table(self.parquet_path, columns=columns)
        self.column_names = self.table.column_names
        self.transform = transform

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("HLEDataset index out of range")

        row = self.table.slice(idx, 1).to_pylist()[0]
        if self.transform is not None:
            return self.transform(row)
        return row
