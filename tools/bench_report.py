#!/usr/bin/env python3
"""Turn sglang bench_serving JSONL into a shareable report.

Reads the rows written by `sglang.bench_serving --output-file` — the same schema
sglang-jax emits — and renders throughput, TTFT and TPOT against concurrency plus
the full metric table. Backend-agnostic: any bench_serving output works, not just
Neuron and not just one model.

    python3 tools/bench_report.py --in artifacts/bench --out report.html
    python3 tools/bench_report.py --in artifacts/bench --out report --format html,pdf,md

Series come from each row's `tag` (set it with `bench_serving --tag`), so one
directory holding several tagged runs renders as a comparison. Output is a single
self-contained file — no network, no CDN, no runtime — so it survives being
emailed, committed, or dropped in a ticket.

PDF export shells out to a headless Chrome/Chromium/Edge if one is installed; no
Python dependencies are added either way.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

# Categorical slots (light, dark) from the validated reference palette; the fixed order is the CVD-safety
# mechanism — re-validate before changing.
SERIES_COLORS = [
    ("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"),   # blue, orange
    ("#1baf7a", "#199e70"), ("#eda100", "#c98500"),   # aqua, yellow
    ("#e87ba4", "#d55181"), ("#008300", "#008300"),   # magenta, green
    ("#4a3aa7", "#9085e9"), ("#e34948", "#e66767"),   # violet, red
]


def series_var(i: int) -> str:
    """CSS var for series `i` — slots are assigned in fixed order, repeating past
    the last slot (legend and table disambiguate if that ever happens)."""
    return f"var(--s{i % len(SERIES_COLORS) + 1})"

PLOT_W, PLOT_H = 680, 300
PAD_L, PAD_R, PAD_T, PAD_B = 64, 116, 20, 44


def load_points(root: pathlib.Path, pattern: str = "**/*.jsonl") -> list[dict[str, Any]]:
    """One record per benchmarked concurrency point.

    Driven by the JSONL fields, not by filenames: `max_concurrency` gives the x
    position and `tag` groups rows into series. Filenames are only consulted as a
    fallback for runs recorded without `--tag`. bench_serving *appends*, so the
    last non-empty line of a file is its most recent run.
    """
    points = []
    for path in sorted(root.glob(pattern)):
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        try:
            row = json.loads(lines[-1])
        except json.JSONDecodeError:
            print(f"  skipped (not JSON): {path}", file=sys.stderr)
            continue
        if not isinstance(row, dict) or row.get("max_concurrency") is None:
            # Fallback: parse concurrency from filename if row lacks max_concurrency.
            m = re.search(r"_c(\d+)", path.stem)
            if not (isinstance(row, dict) and m):
                print(f"  skipped (no max_concurrency): {path}", file=sys.stderr)
                continue
            row["max_concurrency"] = int(m.group(1))
        if not row.get("tag"):
            m = re.search(r"_(tp\d+|[a-z0-9]+)_c\d+", path.stem)
            row["tag"] = m.group(1) if m else "run"
        points.append(row)
    points.sort(key=lambda r: (str(r.get("tag")), int(r["max_concurrency"])))
    return points


def nice_max(v: float) -> float:
    """Round a data max up to a clean axis bound."""
    if v <= 0:
        return 1.0
    import math

    mag = 10 ** math.floor(math.log10(v))
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        if v <= step * mag:
            return step * mag
    return 10 * mag


def fmt(v: float) -> str:
    # Integer counts without trailing .00.
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def line_chart(cid: str, title: str, sub: str, xs: list[int],
               series: list[tuple[str, list[float]]], unit: str) -> str:
    """Line chart: 2px lines, >=8px markers with a 2px surface ring, hairline
    grid, endpoint direct labels, and a crosshair hover layer driven by
    data-* attributes (see the shared script at the foot of the page)."""
    ymax = nice_max(max((v for _, ys in series for v in ys if v is not None), default=1))
    n = len(xs)
    iw = PLOT_W - PAD_L - PAD_R
    ih = PLOT_H - PAD_T - PAD_B
    # Equal x spacing: concurrency is categorical here.
    xpos = [PAD_L + (iw * i / (n - 1) if n > 1 else iw / 2) for i in range(n)]

    def ypos(v: float) -> float:
        return PAD_T + ih - (v / ymax) * ih

    parts = [
        f'<figure class="card"><figcaption><h3>{html.escape(title)}</h3>'
        f'<p class="sub">{html.escape(sub)}</p></figcaption>'
    ]
    if len(series) >= 2:
        keys = "".join(
            f'<span class="key"><svg width="16" height="8" aria-hidden="true">'
            f'<line x1="1" y1="4" x2="15" y2="4" stroke="{series_var(i)}" '
            f'stroke-width="2" stroke-linecap="round"/></svg>'
            f'{html.escape(name)}</span>'
            for i, (name, _) in enumerate(series)
        )
        parts.append(f'<div class="legend">{keys}</div>')

    svg = [
        f'<svg class="chart" id="{cid}" viewBox="0 0 {PLOT_W} {PLOT_H}" '
        f'role="img" aria-label="{html.escape(title)}. Values are in the table below.">'
    ]
    for t in range(5):
        v = ymax * t / 4
        y = ypos(v)
        svg.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PLOT_W - PAD_R}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" class="tick ta-end">{fmt(v)}</text>'
        )
    for i, x in enumerate(xs):
        svg.append(
            f'<text x="{xpos[i]:.1f}" y="{PLOT_H - PAD_B + 22}" class="tick ta-mid">{x}</text>'
        )
    svg.append(
        f'<text x="{PAD_L + iw / 2:.1f}" y="{PLOT_H - 4}" class="axis-title ta-mid">'
        f'concurrency</text>'
    )
    # End labels only if the series separate at the right edge.
    ends = [ys[-1] for _, ys in series if ys and ys[-1] is not None]
    label_ends = len(ends) < 2 or min(
        abs(ypos(a) - ypos(b)) for i, a in enumerate(ends) for b in ends[i + 1:]
    ) >= 14

    for si, (name, ys) in enumerate(series):
        pts = [(xpos[i], ypos(v)) for i, v in enumerate(ys) if v is not None]
        if not pts:
            continue
        d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(pts))
        svg.append(f'<path d="{d}" fill="none" stroke="{series_var(si)}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{series_var(si)}" '
                       f'stroke="var(--surface-1)" stroke-width="2"/>')
        if label_ends:
            ex, ey = pts[-1]
            svg.append(
                f'<text x="{ex + 12:.1f}" y="{ey + 4:.1f}" class="endlab">'
                f'{fmt(ys[-1])} <tspan class="endlab-name">{html.escape(name)}</tspan></text>'
            )
    svg.append(f'<line class="crosshair" id="{cid}-cross" y1="{PAD_T}" '
               f'y2="{PAD_T + ih}" style="display:none"/>')
    for i, x in enumerate(xs):
        half = (iw / (n - 1) / 2) if n > 1 else iw / 2
        svg.append(
            f'<rect class="hit" x="{max(PAD_L, xpos[i] - half):.1f}" y="{PAD_T}" '
            f'width="{min(half * 2, iw):.1f}" height="{ih}" data-i="{i}" tabindex="0" '
            f'role="button" aria-label="concurrency {x}"/>'
        )
    svg.append("</svg>")
    parts.append("".join(svg))

    payload = {
        "xs": xs, "unit": unit,
        "series": [{"name": nm, "vals": ys} for nm, ys in series],
        "geom": {"xpos": xpos, "top": PAD_T, "bot": PAD_T + ih},
    }
    parts.append(
        f'<div class="tip" id="{cid}-tip" role="status"></div>'
        f'<script type="application/json" class="cdata" data-for="{cid}">'
        f"{json.dumps(payload)}</script></figure>"
    )
    return "".join(parts)


def summary_tiles(points: list[dict[str, Any]], tag: str) -> str:
    """Headline stats, all derived from the data — a benchmark page is scanned
    before it is read, so the conclusion goes above the charts."""
    rows = sorted((p for p in points if str(p.get("tag")) == tag),
                  key=lambda r: int(r["max_concurrency"]))
    if len(rows) < 2:
        return ""
    lo, hi = rows[0], rows[-1]
    c_lo, c_hi = int(lo["max_concurrency"]), int(hi["max_concurrency"])
    peak_lo = lo.get("max_output_tokens_per_s") or 0
    peak_hi = hi.get("max_output_tokens_per_s") or 0
    # Efficiency = observed / ideal speedup; the spread is on ITL (not TPOT).
    eff = ((peak_hi / peak_lo) / (c_hi / c_lo) * 100) if peak_lo and c_lo else 0
    itls = [r.get("median_itl_ms") for r in rows if r.get("median_itl_ms")]
    spread = ((max(itls) - min(itls)) / min(itls) * 100) if itls else 0
    avg_hi = hi.get("output_throughput") or 0
    util = (avg_hi / peak_hi * 100) if peak_hi else 0

    # Name the series in the caption when multiple tags exist.
    tiles = [
        ("Peak output", f"{fmt(peak_hi)}", f"{tag} · tok/s at c={c_hi}"),
        ("Decode scaling", f"{eff:.0f}%", f"{tag} · of ideal, c={c_lo}→{c_hi}"),
        ("ITL spread", f"{spread:.0f}%", f"{tag} · decode, c={c_lo}→{c_hi}"),
        ("Sustained vs peak", f"{util:.0f}%", f"{tag} · avg / peak at c={c_hi}"),
    ]
    cells = "".join(
        f'<div class="tile"><span class="tl">{html.escape(l)}</span>'
        f'<span class="tv">{html.escape(v)}</span>'
        f'<span class="tc">{html.escape(c)}</span></div>'
        for l, v, c in tiles
    )
    return cells


def build(points: list[dict[str, Any]], title: str, meta: dict[str, Any]) -> str:
    tags = sorted({str(p.get("tag")) for p in points})
    xs = sorted({int(p["max_concurrency"]) for p in points})
    if len(tags) > len(SERIES_COLORS):
        print(f"warning: {len(tags)} series vs {len(SERIES_COLORS)} palette slots — "
              f"colors repeat; filter or facet the input", file=sys.stderr)

    def col(tag: str, key: str) -> list[float | None]:
        by_c = {int(p["max_concurrency"]): p.get(key) for p in points if str(p.get("tag")) == tag}
        return [by_c.get(c) for c in xs]

    charts = []
    if len(tags) == 1:
        t = tags[0]
        charts.append(line_chart(
            "c-thr", "Output throughput", "peak vs average, tokens/s", xs,
            [("peak", col(t, "max_output_tokens_per_s")),
             ("average", col(t, "output_throughput"))], "tok/s"))
        charts.append(line_chart(
            "c-ttft", "Time to first token", "median vs p99, ms", xs,
            [("median", col(t, "median_ttft_ms")), ("p99", col(t, "p99_ttft_ms"))], "ms"))
        charts.append(line_chart(
            "c-tpot", "Time per output token", "median vs p99, ms", xs,
            [("median", col(t, "median_tpot_ms")), ("p99", col(t, "p99_tpot_ms"))], "ms"))
    else:
        charts.append(line_chart(
            "c-thr", "Peak output throughput", "tokens/s, by series", xs,
            [(t, col(t, "max_output_tokens_per_s")) for t in tags], "tok/s"))
        charts.append(line_chart(
            "c-ttft", "Time to first token", "median, ms", xs,
            [(t, col(t, "median_ttft_ms")) for t in tags], "ms"))
        charts.append(line_chart(
            "c-tpot", "Time per output token", "median, ms", xs,
            [(t, col(t, "median_tpot_ms")) for t in tags], "ms"))

    head = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl, _ in TABLE_COLS)
    rows = []
    for p in points:
        cells = []
        for key, _, _ in TABLE_COLS:
            v = p.get(key)
            cells.append(f"<td>{html.escape(str(v)) if isinstance(v, str) else (fmt(v) if isinstance(v, (int, float)) else '—')}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    m = ", ".join(f"{html.escape(str(k))} {html.escape(str(v))}" for k, v in meta.items())
    return TEMPLATE.format(
        title=html.escape(title), meta=m,
        tiles=f'<div class="tiles">{"".join(summary_tiles(points, t) for t in tags)}</div>',
        charts="".join(charts), thead=head, tbody="".join(rows),
        svars_l="".join(f"--s{i + 1}:{lo};" for i, (lo, _) in enumerate(SERIES_COLORS)),
        svars_d="".join(f"--s{i + 1}:{dk};" for i, (_, dk) in enumerate(SERIES_COLORS)),
    )


TEMPLATE = """<title>{title}</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);{svars_l}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{
color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#fff;
--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);{svars_d}}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;
--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);{svars_d}}}
.viz-root{{background:var(--plane);color:var(--text-primary);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px 20px 56px;}}
.wrap{{max-width:820px;margin:0 auto;}}
h1{{font-size:24px;margin:0 0 4px;letter-spacing:-.01em;}}
.meta{{color:var(--text-secondary);font-size:13px;margin:0 0 28px;}}
.card{{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
margin:0 0 20px;padding:18px 18px 10px;position:relative;overflow-x:auto;}}
figcaption h3{{font-size:15px;margin:0;font-weight:600;}}
.sub{{color:var(--text-secondary);font-size:13px;margin:2px 0 10px;}}
.legend{{display:flex;gap:16px;margin:0 0 6px;font-size:13px;color:var(--text-secondary);}}
.key{{display:inline-flex;align-items:center;gap:6px;}}
.chart{{display:block;width:100%;height:auto;min-width:520px;overflow:visible;}}
.grid{{stroke:var(--grid);stroke-width:1;}}
.tick{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums;}}
.axis-title{{fill:var(--text-secondary);font-size:12px;}}
.ta-end{{text-anchor:end;}}.ta-mid{{text-anchor:middle;}}
.endlab{{fill:var(--text-primary);font-size:12px;font-weight:600;}}
.endlab-name{{fill:var(--text-secondary);font-weight:400;}}
.crosshair{{stroke:var(--axis);stroke-width:1;}}
.hit{{fill:transparent;cursor:crosshair;}}
.hit:focus{{outline:2px solid var(--s1);outline-offset:-2px;}}
.tip{{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;
background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.13);min-width:120px;}}
.tip b{{display:block;color:var(--text-secondary);font-weight:500;margin:0 0 4px;}}
.tip .r{{display:flex;align-items:center;gap:6px;white-space:nowrap;}}
.tip .v{{font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-primary);}}
.tip .n{{color:var(--text-secondary);}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;
font-variant-numeric:tabular-nums;}}
th,td{{padding:7px 9px;text-align:right;border-bottom:1px solid var(--border);
white-space:nowrap;}}
th:first-child,td:first-child{{text-align:left;}}
th{{color:var(--text-secondary);font-weight:600;font-size:11.5px;text-transform:uppercase;
letter-spacing:.03em;}}
.tblwrap{{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
padding:6px 14px 4px;overflow-x:auto;}}
.note{{color:var(--text-secondary);font-size:13px;margin:22px 0 0;}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;
margin:0 0 22px;}}
.tile{{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
padding:14px 16px;display:flex;flex-direction:column;gap:3px;}}
.tl{{font-size:11.5px;text-transform:uppercase;letter-spacing:.03em;color:var(--text-secondary);
font-weight:600;}}
.tv{{font-size:26px;font-weight:600;letter-spacing:-.02em;line-height:1.1;}}
.tc{{font-size:12px;color:var(--muted);}}
@media print{{
  /* Print and headless-Chrome PDF: pin the light tokens (paper is white
     regardless of the OS theme) and keep each card whole across page breaks. */
  .viz-root{{color-scheme:light;--surface-1:#fff;--plane:#fff;--text-primary:#0b0b0b;
  --text-secondary:#52514e;--muted:#6b6a66;--grid:#e1e0d9;--axis:#c3c2b7;
  --border:rgba(11,11,11,.22);{svars_l}padding:0;}}
  .card,.tblwrap,.tile{{break-inside:avoid;page-break-inside:avoid;box-shadow:none;}}
  .card{{overflow:visible;}}
  .chart{{min-width:0;}}
  .tip{{display:none;}}
  h1{{font-size:20px;}}
  @page{{margin:14mm;}}
}}
</style>
<div class="viz-root"><div class="wrap">
<h1>{title}</h1>
<p class="meta">{meta}</p>
{tiles}
{charts}
<h3 style="font-size:15px;margin:26px 0 10px;">All metrics</h3>
<div class="tblwrap"><table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>
<p class="note">Metrics come from <code>sglang.bench_serving</code> — the same tool and
definitions sglang-jax uses — so numbers are comparable across backends. Peak
(<code>max_output_tokens_per_s</code>) is the windowed steady-state max and is the
robust headline; the average is depressed by prefill/TTFT ramp on short runs.</p>
</div></div>
<script>
for (const el of document.querySelectorAll('script.cdata')) {{
  const cfg = JSON.parse(el.textContent);
  const svg = document.getElementById(el.dataset.for);
  const tip = document.getElementById(el.dataset.for + '-tip');
  const cross = document.getElementById(el.dataset.for + '-cross');
  const show = (i, ev) => {{
    const gx = cfg.geom.xpos[i];
    cross.setAttribute('x1', gx); cross.setAttribute('x2', gx);
    cross.style.display = '';
    tip.textContent = '';
    const h = document.createElement('b');
    h.textContent = 'concurrency ' + cfg.xs[i];   // textContent: labels are untrusted
    tip.appendChild(h);
    for (let s = 0; s < cfg.series.length; s++) {{
      const v = cfg.series[s].vals[i];
      if (v === null || v === undefined) continue;
      const row = document.createElement('div'); row.className = 'r';
      const k = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      k.setAttribute('width', 14); k.setAttribute('height', 8);
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', 1); ln.setAttribute('y1', 4);
      ln.setAttribute('x2', 13); ln.setAttribute('y2', 4);
      ln.setAttribute('stroke', 'var(--s' + (s + 1) + ')');
      ln.setAttribute('stroke-width', 2); ln.setAttribute('stroke-linecap', 'round');
      k.appendChild(ln); row.appendChild(k);
      const val = document.createElement('span'); val.className = 'v';
      val.textContent = (v >= 1000 ? v.toLocaleString(undefined, {{maximumFractionDigits: 0}})
                                   : v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)) + ' ' + cfg.unit;
      const nm = document.createElement('span'); nm.className = 'n';
      nm.textContent = cfg.series[s].name;
      row.appendChild(val); row.appendChild(nm); tip.appendChild(row);
    }}
    const box = svg.getBoundingClientRect();
    const card = svg.closest('.card').getBoundingClientRect();
    const px = box.left - card.left + (gx / {{plot_w}}) * box.width;
    tip.style.opacity = 1;
    tip.style.left = Math.min(Math.max(px + 14, 8), card.width - tip.offsetWidth - 8) + 'px';
    tip.style.top = (box.top - card.top + 16) + 'px';
  }};
  const hide = () => {{ tip.style.opacity = 0; cross.style.display = 'none'; }};
  for (const hit of svg.querySelectorAll('.hit')) {{
    const i = +hit.dataset.i;
    hit.addEventListener('pointermove', (e) => show(i, e));
    hit.addEventListener('focus', () => show(i));
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('blur', hide);
  }}
}}
</script>
"""


#: (json key, label in the rendered table, column name in md/csv exports)
TABLE_COLS = [
    ("tag", "config", "config"),
    ("max_concurrency", "c", "concurrency"),
    ("max_output_tokens_per_s", "peak tok/s", "peak_tok_s"),
    ("output_throughput", "avg tok/s", "avg_tok_s"),
    ("median_ttft_ms", "TTFT p50", "ttft_p50_ms"),
    ("p99_ttft_ms", "TTFT p99", "ttft_p99_ms"),
    ("median_tpot_ms", "TPOT p50", "tpot_p50_ms"),
    ("p99_tpot_ms", "TPOT p99", "tpot_p99_ms"),
    ("median_itl_ms", "ITL p50", "itl_p50_ms"),
    ("p99_itl_ms", "ITL p99", "itl_p99_ms"),
    ("median_e2e_latency_ms", "E2E p50", "e2e_p50_ms"),
    ("completed", "reqs", "requests"),
]


def find_browser() -> str | None:
    """A Chromium-family binary that can print a local file to PDF."""
    for name in ("google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser", "microsoft-edge"):
        if (found := shutil.which(name)):
            return found
    for path in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "/Applications/Chromium.app/Contents/MacOS/Chromium",
                 "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"):
        if pathlib.Path(path).is_file():
            return path
    return None


def write_pdf(page: str, out: pathlib.Path) -> bool:
    browser = find_browser()
    if not browser:
        print("  pdf: no Chrome/Chromium/Edge found — open the HTML and use "
              "Print → Save as PDF (the page has a print stylesheet).", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "report.html"
        src.write_text(page)
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
               "--no-pdf-header-footer", "--virtual-time-budget=4000",
               f"--print-to-pdf={out.resolve()}", src.resolve().as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not out.exists() or out.stat().st_size == 0:
        print(f"  pdf: {browser} failed\n{proc.stderr[-500:]}", file=sys.stderr)
        return False
    return True


def write_markdown(points: list[dict[str, Any]], title: str, meta: dict[str, Any],
                   out: pathlib.Path) -> None:
    """A table that pastes into a PR, an issue, or the setup log."""
    lines = [f"# {title}", "",
             ", ".join(f"{k} {v}" for k, v in meta.items()), "",
             "| " + " | ".join(col for _, _, col in TABLE_COLS) + " |",
             "|" + "|".join("---" for _ in TABLE_COLS) + "|"]
    for p in points:
        cells = []
        for key, _, _ in TABLE_COLS:
            v = p.get(key)
            cells.append(fmt(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                         else (str(v) if v is not None else "—"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Measured with `sglang.bench_serving`. Peak "
              "(`max_output_tokens_per_s`) is the windowed steady-state max; the "
              "average is depressed by prefill/TTFT ramp on short runs."]
    out.write_text("\n".join(lines) + "\n")


def write_csv(points: list[dict[str, Any]], out: pathlib.Path) -> None:
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([col for _, _, col in TABLE_COLS])
        for p in points:
            w.writerow([p.get(key) for key, _, _ in TABLE_COLS])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, type=pathlib.Path,
                    help="directory of bench_serving JSONL files")
    ap.add_argument("--out", dest="out", required=True, type=pathlib.Path,
                    help="output path; the extension is set per --format")
    ap.add_argument("--format", default="html",
                    help="comma-separated: html, pdf, md, csv (default: html)")
    ap.add_argument("--glob", default="**/*.jsonl",
                    help="pattern under --in (default: **/*.jsonl)")
    ap.add_argument("--title", default=None,
                    help="report heading (default: derived from the run manifest)")
    args = ap.parse_args()

    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    if bad := [f for f in formats if f not in {"html", "pdf", "md", "csv"}]:
        raise SystemExit(f"unknown format(s): {', '.join(bad)}")

    points = load_points(args.indir, args.glob)
    if not points:
        raise SystemExit(f"no bench_serving JSONL found under {args.indir}")

    meta: dict[str, Any] = {}
    for mf in sorted(args.indir.rglob("manifest.json")):
        man = json.loads(mf.read_text())
        meta = {k: v for k, v in (
            ("model", man.get("model")), ("ctx", man.get("ctx_bucket")),
            ("in/out", f"{man.get('input_len')}/{man.get('output_len')}"
                       if man.get("input_len") else None),
            ("max_bs", man.get("max_bs"))) if v is not None}
    if not meta:  # no manifest — fall back to what the rows themselves carry
        first = points[0]
        meta = {k: v for k, v in (
            ("backend", first.get("backend")),
            ("in/out", f"{first.get('random_input_len')}/{first.get('random_output_len')}"
                       if first.get("random_input_len") else None)) if v}
    meta["points"] = len(points)

    title = args.title or (
        f"{meta.get('model', 'Serving')} — benchmark" if meta.get("model")
        else "Serving benchmark")
    page = build(points, title, meta).replace("{plot_w}", str(PLOT_W))

    stem = args.out.with_suffix("")
    for f in formats:
        target = args.out if args.out.suffix.lstrip(".") == f else stem.with_suffix(f".{f}")
        if f == "html":
            target.write_text(page)
        elif f == "pdf":
            if not write_pdf(page, target):
                continue
        elif f == "md":
            write_markdown(points, title, meta, target)
        elif f == "csv":
            write_csv(points, target)
        print(f"wrote {target}  ({len(points)} points)")


if __name__ == "__main__":
    main()
