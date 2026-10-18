"""Builds the browser demo, docs/index.html, from web/template.html.

Converts the folded float32 weights (models/weights.bin, from `experiment final`) to float16,
saves them as web/weights.f16 (what scripts/check-infer.mjs tests), and embeds them gzip-compressed
along with web/infer.js and the results files. Every number on the page comes from results/*.json.

    python3 scripts/build-demo.py
"""
import array
import base64
import gzip
import json
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def main():
    floats = array.array("f")
    floats.frombytes((ROOT / "models" / "weights.bin").read_bytes())
    half = struct.pack(f"<{len(floats)}e", *floats)
    (ROOT / "web" / "weights.f16").write_bytes(half)

    final = load("final.json")
    runs = [load(f"run-{name}.json") for name in ("plain", "augment", final["model"])]
    browser = load("browser-check.json") if (ROOT / "results" / "browser-check.json").exists() else {
        "images": 0, "sameClass": 0, "maxLogitDifference": 0, "msPerImage": 0}
    infer = (ROOT / "web" / "infer.js").read_text().replace("export function", "function")

    page = (ROOT / "web" / "template.html").read_text()
    fills = {
        "{{WEIGHTS}}": base64.b64encode(gzip.compress(half, compresslevel=9, mtime=0)).decode("ascii"),
        "{{INFER}}": infer,
        "{{FINAL}}": json.dumps(final, separators=(",", ":")),
        "{{BASELINES}}": json.dumps(load("baselines.json"), separators=(",", ":")),
        "{{RUNS}}": json.dumps(runs, separators=(",", ":")),
        "{{SAMPLES}}": json.dumps(load("samples.json"), separators=(",", ":")),
        "{{BROWSER}}": json.dumps(browser, separators=(",", ":")),
    }
    for key, value in fills.items():
        assert page.count(key) == 1, key
        page = page.replace(key, value)
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(f"Wrote web/weights.f16 ({len(half):,} bytes) and docs/index.html ({len(page.encode()):,} bytes)")


if __name__ == "__main__":
    main()
