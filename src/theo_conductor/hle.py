from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
import pyarrow.compute as pc
from torch.utils.data import Dataset


class HLEDataset(Dataset):
    DEFAULT_PATHS = (
        Path(__file__).resolve().parent / "data" / "hle.parquet",
        Path(__file__).resolve().parents[2] / "data" / "hle.parquet",
    )
    CATEGORY_COLUMN = "category"
    CATEGORY = "Physics"

    def __init__(
        self,
        parquet_path: str | Path | None = None,
        *,
        columns: list[str] | None = None,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path) if parquet_path is not None else self._default_path()
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"HLE parquet file not found: {self.parquet_path}")

        read_columns = columns
        if read_columns is not None and self.CATEGORY_COLUMN not in read_columns:
            read_columns = [*read_columns, self.CATEGORY_COLUMN]

        table = pq.read_table(self.parquet_path, columns=read_columns)
        self.table = table.filter(pc.equal(table[self.CATEGORY_COLUMN], self.CATEGORY))
        if columns is not None:
            self.table = self.table.select(columns)
        self.column_names = self.table.column_names
        self.transform = transform

    @classmethod
    def _default_path(cls) -> Path:
        for path in cls.DEFAULT_PATHS:
            if path.exists():
                return path
        return cls.DEFAULT_PATHS[-1]

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
