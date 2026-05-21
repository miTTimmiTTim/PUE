# PUE — Interactive Notebooks

marimo notebooks served as WebAssembly through GitHub Pages. Students open a URL and run Python in their browser — no installation, no molab account.

## Layout

```
notebooks/   editable notebooks (--mode edit)
apps/        read-only apps with hidden code (--mode run --no-show-code)
public/      optional data files (place inside notebooks/ or apps/ next to the .py)
templates/   Jinja template for the landing page
.github/     build script + Pages deploy workflow
```

## Add a notebook

Drop a `.py` file into `notebooks/` (editable) or `apps/` (read-only). Declare deps inline as PEP-723 script metadata so they get baked into the WASM bundle:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
# ]
# ///
```

Author locally with `marimo edit --sandbox notebooks/your_notebook.py` so the inline deps resolve.

## Data files

Place CSVs/etc. in a `public/` folder next to the notebook and load via:

```python
import marimo as mo
path = mo.notebook_location() / "public" / "data.csv"
```

The build copies `public/` into the export.

## Build locally

```bash
uv run .github/scripts/build.py
python -m http.server -d _site
```

Open <http://localhost:8000>.

## Deploy

1. Push to GitHub.
2. Repo Settings → Pages → Source: **GitHub Actions**.
3. Push to `main` (or trigger the workflow manually). The action exports every notebook and publishes to `https://<user>.github.io/<repo>/`.

## WASM caveats

- Pure-Python wheels from PyPI plus NumPy, SciPy, scikit-learn, duckdb, polars work. C-extension packages outside that list will not.
- 2 GB browser memory cap; no multithreading/multiprocessing; no PDB.
- Chrome is the smoothest; Firefox/Safari/Edge also work.
