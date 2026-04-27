# Data folder

Place optional public IDS CSV files here if you want to reproduce CICIDS-style or user-supplied dataset experiments.

Large raw datasets are intentionally not included in this repository. The benchmark can be run with:

```bash
python code/cisc650_payload_free_ids_benchmark.py --dataset cicids --data-dir data --output-dir outputs --max-rows-per-file 50000
```

For a user-supplied CSV file, place the file in this folder and run:

```bash
python code/cisc650_payload_free_ids_benchmark.py --dataset user --input-csv data/my_flows.csv --label-column Label --output-dir outputs
```
