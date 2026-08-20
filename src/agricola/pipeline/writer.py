# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Output writers used by agricola pipelines."""

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds


class ParquetRotatingWriter:
    """Buffer PyArrow tables and write them as rotating Parquet fragments."""

    def __init__(
        self,
        output_dir: Path,
        partition_phenotype: bool = True,
        max_rows: int = 5_000_000,
    ):
        self.output_dir = output_dir
        self.max_rows = max_rows
        self.buffer: list[pa.Table] = []
        self.buffer_rows = 0
        self.partition_phenotype = partition_phenotype
        self.idx = 0

        output_dir.mkdir(parents=True, exist_ok=False)

    def __enter__(self) -> "ParquetRotatingWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def write(self, table: pa.Table) -> None:
        self.buffer.append(table)
        self.buffer_rows += table.num_rows

        if self.buffer_rows >= self.max_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        partitioning = ["phenotype"] if self.partition_phenotype else None
        ds.write_dataset(
            data=self.buffer,
            base_dir=self.output_dir,
            format="parquet",
            basename_template=f"part-{{i}}_{self.idx:06d}.parquet",
            partitioning=partitioning,
            max_partitions=5000,
            max_open_files=5000,
            existing_data_behavior="overwrite_or_ignore",
        )

        self.buffer.clear()
        self.buffer_rows = 0
        self.idx += 1

    def close(self) -> None:
        self.flush()
