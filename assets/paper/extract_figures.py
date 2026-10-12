"""Render the six original paper graphics as tightly cropped, lossless PNGs.

Requires PyMuPDF 1.28.2 for byte-identical reproduction of the checked-in PNGs:
    python -m pip install PyMuPDF==1.28.2
    python assets/paper/extract_figures.py

The crop rectangles follow the original embedded vector graphics, rather than
reconstructing plots or editing labels. The source manuscript is never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf


SOURCE_SHA256 = "2cff5d39b71ab1c6eceb8aac3f76620074850ebd6f87bcd5aa37d823d629395e"
DPI = 300
PADDING_PT = 2.0

# Embedded graphic bounds are in PDF points with a top-left page origin.
# Values retain the original form-object dimensions and placement transforms.
GRAPHICS = (
    ("figure1_compression_success", "Figure 1: compression, success and memory", 1,
     (311.978, 792 - 437.447 - 136.8000030517578 * 1.00006,
      311.978 + 251.05853271484375 * 1.00006, 792 - 437.447)),
    ("figure2_framework", "Figure 2: AGQ framework", 2,
     (48.964, 792 - 437.373 - 695.9979858398438 * .43715,
      48.964 + 1175.9969482421875 * .43715, 792 - 437.373)),
    ("figure3_allocation_sensitivity", "Figure 3: allocation and matrix-level probes", 3,
     (311.978, 792 - 226.972 - 206.06585693359375 * 1.00006,
      311.978 + 251.05853271484375 * 1.00006, 792 - 226.972)),
    ("figure4_lambda_success", "Figure 4: W2.5 success across lambda settings", 4,
     (48.964, 792 - 601.206 - 144.0 * 1.00006,
      48.964 + 251.05853271484375 * 1.00006, 792 - 601.206)),
    ("table1_task_success", "Task success at ultra-low bitwidths", 4,
     (48.964, 792 - 204.075 - 184.65481567382812,
      48.964 + 251.05999755859375, 792 - 204.075)),
    ("table2_storage_performance", "Storage and inference efficiency", 4,
     (311.978, 792 - 360.026 - 115.37384033203125,
      311.978 + 251.05999755859375, 792 - 360.026)),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[2] / "paper" / "AGQ_ISCAS_27.pdf")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = args.source.resolve()
    if digest(source) != SOURCE_SHA256:
        raise SystemExit("Source PDF differs from the reviewed manuscript; crop positions require review.")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_pdf": "../../paper/AGQ_ISCAS_27.pdf",
        "source_sha256": SOURCE_SHA256,
        "source_pages": 6,
        "renderer": "PyMuPDF",
        "renderer_version": pymupdf.VersionBind,
        "dpi": DPI,
        "padding_points": PADDING_PT,
        "coordinate_system": "PDF points; page top-left origin; x0, y0, x1, y1",
        "method": "Direct rendering of original embedded-graphic regions; no data or label edits.",
        "notes": [
            "The two tables have no numbered captions in the manuscript; filenames provide stable gallery identifiers.",
            "Figure 1 preserves the original BF16 axis label. Its 96.9% reference comes from ActQuant FP16, not a new BF16 evaluation.",
            "Figure 1 minimum memory and the performance table use different coding settings; consult the results documentation.",
        ],
        "graphics": [],
    }
    with pymupdf.open(source) as document:
        if len(document) != manifest["source_pages"]:
            raise SystemExit("Unexpected manuscript page count.")
        for stem, title, page_number, bounds in GRAPHICS:
            page = document[page_number - 1]
            clip = pymupdf.Rect(bounds)
            clip = pymupdf.Rect(clip.x0 - PADDING_PT, clip.y0 - PADDING_PT,
                                clip.x1 + PADDING_PT, clip.y1 + PADDING_PT) & page.rect
            pixmap = page.get_pixmap(clip=clip, dpi=DPI, colorspace=pymupdf.csRGB, alpha=False)
            target = args.output / (stem + ".png")
            pixmap.save(target)
            manifest["graphics"].append({
                "id": stem,
                "title": title,
                "page": page_number,
                "graphic_bounds_points": [round(value, 6) for value in bounds],
                "render_clip_points": [round(value, 6) for value in clip],
                "file": target.name,
                "width_px": pixmap.width,
                "height_px": pixmap.height,
                "sha256": digest(target),
            })
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graphics": len(manifest["graphics"]), "dpi": DPI,
                      "source_unchanged": digest(source) == SOURCE_SHA256}))


if __name__ == "__main__":
    main()
