# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.3",
#     "fire==0.7.0",
#     "loguru==0.7.0"
# ]
# ///
"""
Export marimo notebooks under notebooks/ (edit mode) and apps/ (run mode)
to WASM HTML, mirror companion files (css, layouts, public/), inject a
post-load script that switches to app view, and render a Jinja index.

Run: uv run .github/scripts/build.py
Output: _site/
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import fire
import jinja2
from loguru import logger


def _run_export(notebook_path: Path, output_file: Path, mode: str, show_code: Optional[bool] = None) -> bool:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        "uvx", "marimo", "export", "html-wasm", "--sandbox",
        "--mode", mode,
    ]
    if show_code is False:
        cmd.append("--no-show-code")
    cmd.extend([str(notebook_path), "-o", str(output_file)])
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed for {notebook_path}: {e.stderr}")
        return False


def _export_html_wasm(notebook_path: Path, output_dir: Path, as_app: bool = False) -> bool:
    output_file = output_dir / notebook_path.with_suffix(".html")
    if as_app:
        logger.info(f"Exporting {notebook_path} → {output_file} (app, run mode, code hidden)")
        return _run_export(notebook_path, output_file, mode="run", show_code=False)
    logger.info(f"Exporting {notebook_path} → {output_file} (notebook, edit mode)")
    return _run_export(notebook_path, output_file, mode="edit")


def _copy_companion_files(folder: Path, output_dir: Path) -> None:
    skip_dirs = {"__pycache__", ".ipynb_checkpoints"}
    for src in folder.rglob("*"):
        if src.is_dir() or any(p in skip_dirs for p in src.parts) or src.suffix == ".py":
            continue
        dst = output_dir / folder.name / src.relative_to(folder)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _export_folder(folder: Path, output_dir: Path, as_app: bool = False) -> List[dict]:
    if not folder.exists():
        return []
    notebooks = sorted(folder.rglob("*.py"))
    if not notebooks:
        logger.warning(f"No .py files in {folder}")
        return []
    exported = [
        {
            "display_name": nb.stem.replace("_", " ").replace("-", " ").title(),
            "html_path": str(nb.with_suffix(".html")),
        }
        for nb in notebooks
        if _export_html_wasm(nb, output_dir, as_app=as_app)
    ]
    _copy_companion_files(folder, output_dir)
    return exported


def _generate_index(output_dir: Path, template_file: Path, notebooks: list, apps: list) -> None:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_file.parent),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_file.name)
    (output_dir / "index.html").write_text(template.render(notebooks=notebooks, apps=apps))


def main(
    output_dir: Union[str, Path] = "_site",
    template: Union[str, Path] = "templates/tailwind.html.j2",
) -> None:
    output_dir = Path(output_dir)
    template_file = Path(template)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    notebooks_data = _export_folder(Path("notebooks"), output_dir, as_app=False)
    apps_data = _export_folder(Path("apps"), output_dir, as_app=True)

    if not notebooks_data and not apps_data:
        logger.warning("Nothing exported.")
        return

    _generate_index(output_dir, template_file, notebooks_data, apps_data)
    (output_dir / ".nojekyll").touch()
    logger.info(f"Done. Serve locally with: python -m http.server -d {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
