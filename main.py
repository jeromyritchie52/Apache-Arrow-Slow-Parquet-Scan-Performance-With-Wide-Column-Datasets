import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

def generate_dataset(file_path, num_cols=1000, num_rows=100000):
    if not os.path.exists(file_path):
        print(f"Generating synthetic wide dataset with {num_cols} columns and {num_rows} rows...")
        data = {f"col_{i}": np.random.randn(num_rows) for i in range(num_cols)}
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, row_group_size=10000)
        print("Dataset generation complete.")

def run_optimized_scan(file_path):
    # Set CPU and IO thread counts to optimize multi-threaded execution
    cpu_count = os.cpu_count() or 4
    pa.set_cpu_count(cpu_count)
    pa.set_io_thread_count(cpu_count * 2)

    # Configure scan options for wide column datasets
    scan_options = ds.ParquetFragmentScanOptions(
        pre_buffer=True,
        use_buffered_stream=True,
        buffer_size=1 << 20  # 1MB buffer size
    )
    
    format = ds.ParquetFileFormat(default_fragment_scan_options=scan_options)
    dataset = ds.dataset(file_path, format=format)
    
    # Project a subset of columns (10 columns out of 1000)
    projection = [f"col_{i}" for i in range(0, 100, 10)]
    
    start_time = time.time()
    # Use optimized scanner options
    scanner = dataset.scanner(
        columns=projection,
        batch_size=128 * 1024,  # Optimize batch size for memory/CPU balance
        batch_readahead=16,
        fragment_readahead=4
    )
    table = scanner.to_table()
    duration = time.time() - start_time
    print(f"Scan completed in: {duration:.4f} seconds")
    return table, duration

if __name__ == "__main__":
    file_path = "wide_dataset.parquet"
    generate_dataset(file_path)
    run_optimized_scan(file_path)
