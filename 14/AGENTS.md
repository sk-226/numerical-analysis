## How to run the code

Use `uv run` to run the code.

## Data augmentation policy

Whenever we enrich `inputs/runs.csv` with new derived data (for example, condition numbers), ensure the CSV on disk is updated so the added columns persist for future runs.

Example: `utils/condition_numbers.py` writes the file only when the column is missing so we avoid redundant overwrites while still persisting the new data:

```python
if "condition_number" not in runs_df.columns:
    merged_df.to_csv(runs_path, index=False)
```
