# Notebook sources

Each notebook is written as a plain Python file in "percent" format (cells start with `# %%`): easy to
diff and review. Rebuild and execute a notebook from `ml/`:

```bash
uv run jupytext --to ipynb -o notebooks/03_results.ipynb notebooks/src/03_results.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_results.ipynb
```
