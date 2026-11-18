# Data Mining Homework 4

## Environment

This project uses Python 3.12 and `uv`. The analysis is implemented as a regular Python script; Jupyter is not required.

Install or synchronize the project environment:

```bash
uv sync
```

Run the analysis:

```bash
uv run python main.py
```

The Python workflow uses:

- NumPy for numerical calculations;
- pandas for result tables;
- scikit-learn for the Iris dataset and decision-tree experiments; and
- Matplotlib for figures that can be included in the LaTeX report.

Generated figures belong in `figures/`. The final report is compiled with:

```bash
tectonic report/main.tex --outdir output/pdf
```
