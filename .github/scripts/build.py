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
to WASM HTML, drop a .nojekyll marker, and render a Jinja index page.

Run: uv run .github/scripts/build.py
Output: _site/
"""

import re
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import jinja2
import fire
from loguru import logger


_LAYOUT_RE = re.compile(r'^\s*layout_file\s*=.*,?\s*\n', re.MULTILINE)


def _has_slides_layout(notebook_path: Path) -> bool:
    text = notebook_path.read_text()
    return bool(_LAYOUT_RE.search(text)) and "slides" in text


def _strip_layout(source: str) -> str:
    return _LAYOUT_RE.sub("", source)


def _run_export(
    notebook_path: Path,
    output_file: Path,
    mode: str,
    show_code: Optional[bool] = None,
) -> bool:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        "uvx", "marimo", "export", "html-wasm", "--sandbox",
        "--mode", mode,
    ]
    if show_code is False:
        cmd.append("--no-show-code")
    cmd.extend([str(notebook_path), "-o", str(output_file)])
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed for {notebook_path}: {e.stderr}")
        return False


def _export_html_wasm(notebook_path: Path, output_dir: Path, as_app: bool = False) -> bool:
    """Export a notebook. If it declares a slides layout_file, also emit a
    `-scroll.html` companion with the layout stripped so students can switch."""
    output_file = output_dir / notebook_path.with_suffix(".html")

    if as_app:
        logger.info(f"Exporting {notebook_path} → {output_file} (app, run, code hidden)")
        ok = _run_export(notebook_path, output_file, mode="run", show_code=False)
    else:
        logger.info(f"Exporting {notebook_path} → {output_file} (notebook, edit)")
        ok = _run_export(notebook_path, output_file, mode="edit")

    if not ok:
        return False

    if _has_slides_layout(notebook_path):
        scroll_name = notebook_path.with_name(notebook_path.stem + "-scroll.py")
        scroll_out = output_dir / scroll_name.with_suffix(".html").relative_to(notebook_path.parent.parent) \
            if False else output_dir / notebook_path.parent.name / (notebook_path.stem + "-scroll.html")
        scroll_out.parent.mkdir(parents=True, exist_ok=True)

        stripped = _strip_layout(notebook_path.read_text())
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=notebook_path.parent) as tf:
            tf.write(stripped)
            tmp_path = Path(tf.name)
        try:
            logger.info(f"Exporting scroll variant → {scroll_out}")
            _run_export(tmp_path, scroll_out, mode="edit")
        finally:
            tmp_path.unlink(missing_ok=True)

    return True


def _copy_companion_files(folder: Path, output_dir: Path) -> None:
    """Mirror non-.py files (CSS, layouts/, public/, JSON, images) so marimo's
    css_file / layout_file references resolve in the exported HTML."""
    skip_dirs = {"__pycache__", ".ipynb_checkpoints"}
    skip_exts = {".py"}
    for src in folder.rglob("*"):
        if src.is_dir() or any(part in skip_dirs for part in src.parts):
            continue
        if src.suffix in skip_exts:
            continue
        rel = src.relative_to(folder)
        dst = output_dir / folder.name / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Copied companion {src} → {dst}")


def _export_folder(folder: Path, output_dir: Path, as_app: bool = False) -> List[dict]:
    if not folder.exists():
        logger.warning(f"Skipping missing folder: {folder}")
        return []

    notebooks = sorted(folder.rglob("*.py"))
    if not notebooks:
        logger.warning(f"No .py files in {folder}")
        return []

    exported = []
    for nb in notebooks:
        if not _export_html_wasm(nb, output_dir, as_app=as_app):
            continue
        display = nb.stem.replace("_", " ").replace("-", " ").title()
        entry = {
            "display_name": display,
            "html_path": str(nb.with_suffix(".html")),
        }
        if (output_dir / nb.parent.name / (nb.stem + "-scroll.html")).exists():
            entry["scroll_path"] = str(nb.parent / (nb.stem + "-scroll.html"))
        exported.append(entry)

    _copy_companion_files(folder, output_dir)
    return exported


def _generate_index(output_dir: Path, template_file: Path, notebooks: list, apps: list) -> None:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_file.parent),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_file.name)
    (output_dir / "index.html").write_text(template.render(notebooks=notebooks, apps=apps))
    logger.info(f"Wrote {output_dir / 'index.html'}")


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
    logger.info("Created .nojekyll")

    logger.info(f"Done. Serve locally with: python -m http.server -d {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
