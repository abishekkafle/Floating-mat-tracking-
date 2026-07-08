"""Render results.json as a polished, self-contained HTML report.

No external dependencies (no CDN, no build step): the page is a single
static HTML file with inline CSS and a small vanilla-JS chart renderer, so
it opens directly in a browser and travels as a supplementary artifact
alongside the paper. Colors reuse the Wong (2011) colorblind-safe pairing
already used in plot.py, so the report matches the paper's figures.
"""
from __future__ import annotations

import json
from pathlib import Path

# Wong (2011) colorblind-safe palette — same mapping as plot.py, validated
# (CVD separation + contrast) for this exact two-color categorical use.
COLORS = {"Tension-guide": "#D55E00", "Distributed-modular": "#0072B2"}


def _ordered_names(results: dict) -> list[str]:
    return [cfg["config"] for cfg in results["per_config"].values()]


def _metric(results: dict, key: str, ci_key: str, label: str, unit: str) -> dict:
    cfgs = results["per_config"]
    return {
        "label": label,
        "unit": unit,
        "values": [cfg[key] for cfg in cfgs.values()],
        "ci": [cfg[ci_key] for cfg in cfgs.values()],
    }


def build_report_data(results: dict) -> dict:
    cfgs = results["per_config"]
    names = _ordered_names(results)

    convergence = {}
    for cfg in cfgs.values():
        points = {int(t): v for t, v in cfg["convergence_mean_drift_cm"].items()}
        convergence[cfg["config"]] = dict(sorted(points.items()))

    block_length = {
        cfg["config"]: {
            "sec": cfg["block_lag_sec"],
            "samples": cfg["block_lag_samples"],
            "is_lower_bound": cfg["block_lag_is_lower_bound"],
        }
        for cfg in cfgs.values()
    }

    return {
        "configs": names,
        "colors": {name: COLORS.get(name, "#666666") for name in names},
        "n_samples": {cfg["config"]: cfg["n"] for cfg in cfgs.values()},
        "dt_sec": next(iter(cfgs.values()))["dt_sec"],
        "metrics": {
            "mean_drift": _metric(results, "mean_drift_cm", "mean_drift_ci", "Mean radial drift", "cm"),
            "p95_drift": _metric(results, "p95_drift_cm", "p95_drift_ci", "95th-pct radial drift", "cm"),
            "rms_heading": _metric(results, "rms_heading_deg", "rms_heading_ci", "RMS heading deviation", "deg"),
            "tension_cov": _metric(results, "tension_cov_mbb_mean", "tension_cov_ci", "Cross-channel tension CoV", ""),
        },
        "tensions": {cfg["config"]: cfg["mean_tensions_N"] for cfg in cfgs.values()},
        "convergence": convergence,
        "block_length": block_length,
        "cost": results["cost_summary"],
    }


_CSS = """
:root {
  color-scheme: light dark;
  --bg: #fcfcfb; --panel: #ffffff; --border: #e4e2df;
  --ink: #1a1a19; --ink-muted: #55534f; --ink-faint: #86837d;
  --grid: #e8e6e2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17171a; --panel: #1f1f22; --border: #333338;
    --ink: #f2f1ee; --ink-muted: #b7b4ad; --ink-faint: #837f77;
    --grid: #333338;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 20px 60px;
  background: var(--bg); color: var(--ink);
  font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
header { padding: 40px 0 20px; border-bottom: 1px solid var(--border); }
h1 { font-size: 1.7rem; margin: 0 0 6px; letter-spacing: -0.01em; }
.subtitle { color: var(--ink-muted); font-size: 0.95rem; margin: 0; }
h2 { font-size: 1.15rem; margin: 0 0 4px; }
.section { padding: 34px 0; border-bottom: 1px solid var(--border); }
.section:last-of-type { border-bottom: none; }
.section-desc { color: var(--ink-muted); font-size: 0.88rem; margin: 0 0 20px; max-width: 68ch; }

.legend { display: flex; gap: 20px; margin: 14px 0 24px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.88rem; color: var(--ink-muted); }
.swatch { width: 12px; height: 12px; border-radius: 3px; flex: none; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 16px 0 4px; }
.tile { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.tile .label { font-size: 0.78rem; color: var(--ink-muted); margin-bottom: 6px; }
.tile .value { font-size: 1.5rem; font-variant-numeric: tabular-nums; font-weight: 600; }
.tile .value .unit { font-size: 0.85rem; font-weight: 400; color: var(--ink-muted); margin-left: 3px; }
.tile .sub { font-size: 0.78rem; color: var(--ink-faint); margin-top: 4px; font-variant-numeric: tabular-nums; }

.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 18px; margin-bottom: 22px; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 14px 6px; }
.panel-title { font-size: 0.82rem; color: var(--ink-muted); margin-bottom: 6px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.axis-label { fill: var(--ink-faint); font-size: 9.5px; }
.value-label { fill: var(--ink); font-size: 10.5px; font-weight: 600; font-variant-numeric: tabular-nums; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.ci-line { stroke: var(--ink); stroke-opacity: 0.55; stroke-width: 1.2; }
.bar { cursor: pointer; }
.dot { cursor: pointer; }
.line-path { fill: none; stroke-width: 2; }

table { width: 100%; border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-muted); font-weight: 500; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.02em; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); }
.table-wrap table { margin: 0; }
.table-wrap th, .table-wrap td { padding: 9px 14px; }

.tooltip {
  position: fixed; pointer-events: none; z-index: 50;
  background: var(--ink); color: var(--bg);
  padding: 6px 9px; border-radius: 6px; font-size: 0.78rem;
  transform: translate(-50%, -100%); margin-top: -10px;
  white-space: nowrap; opacity: 0; transition: opacity 0.08s ease;
  font-variant-numeric: tabular-nums;
}
.tooltip.show { opacity: 1; }
footer { padding: 24px 0 10px; color: var(--ink-faint); font-size: 0.78rem; }
footer p { margin: 4px 0; }
"""

_JS = r"""
const tooltip = document.getElementById('tooltip');
function showTip(evt, html) {
  tooltip.innerHTML = html;
  tooltip.style.left = evt.clientX + 'px';
  tooltip.style.top = evt.clientY + 'px';
  tooltip.classList.add('show');
}
function moveTip(evt) {
  tooltip.style.left = evt.clientX + 'px';
  tooltip.style.top = evt.clientY + 'px';
}
function hideTip() { tooltip.classList.remove('show'); }

function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function fmt(v, d) {
  if (v === null || v === undefined) return '—';
  return Number(v).toFixed(d === undefined ? 2 : d);
}

// Grouped single-value bar chart with an optional CI whisker, one bar per config.
function renderBarPanel(svg, categories, values, ci, colors, decimals) {
  const W = 260, H = 200, padL = 8, padR = 8, padT = 22, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const allVals = values.concat(ci ? ci.flat() : []);
  const maxV = Math.max(...allVals) * 1.22 || 1;
  const y = v => padT + plotH - (v / maxV) * plotH;
  const n = categories.length;
  const slot = plotW / n;
  const barW = Math.min(56, slot * 0.5);

  // gridlines (3 ticks)
  for (let i = 0; i <= 2; i++) {
    const v = (maxV / 2) * i;
    const gy = y(v);
    svg.appendChild(svgEl('line', { class: 'gridline', x1: padL, x2: W - padR, y1: gy, y2: gy }));
  }

  categories.forEach((cat, i) => {
    const cx = padL + slot * i + slot / 2;
    const v = values[i];
    const barX = cx - barW / 2;
    const barY = y(v);
    const barH = padT + plotH - barY;
    const color = colors[cat];

    const rect = svgEl('rect', {
      class: 'bar', x: barX, y: barY, width: barW, height: Math.max(barH, 1),
      rx: 4, ry: 4, fill: color,
    });
    svg.appendChild(rect);

    if (ci && ci[i]) {
      const [lo, hi] = ci[i];
      const yLo = y(lo), yHi = y(hi);
      svg.appendChild(svgEl('line', { class: 'ci-line', x1: cx, x2: cx, y1: yLo, y2: yHi }));
      svg.appendChild(svgEl('line', { class: 'ci-line', x1: cx - 5, x2: cx + 5, y1: yLo, y2: yLo }));
      svg.appendChild(svgEl('line', { class: 'ci-line', x1: cx - 5, x2: cx + 5, y1: yHi, y2: yHi }));
    }

    const label = svgEl('text', { class: 'value-label', x: cx, y: barY - 6, 'text-anchor': 'middle' });
    label.textContent = fmt(v, decimals);
    svg.appendChild(label);

    const catLabel = svgEl('text', { class: 'axis-label', x: cx, y: H - 8, 'text-anchor': 'middle' });
    catLabel.textContent = cat;
    svg.appendChild(catLabel);

    const hit = svgEl('rect', { x: padL + slot * i, y: padT, width: slot, height: plotH, fill: 'transparent', class: 'bar' });
    const ciText = ci && ci[i] ? ` (95% CI ${fmt(ci[i][0], decimals)}–${fmt(ci[i][1], decimals)})` : '';
    hit.addEventListener('mouseenter', e => showTip(e, `<b>${cat}</b><br>${fmt(v, decimals)}${ciText}`));
    hit.addEventListener('mousemove', moveTip);
    hit.addEventListener('mouseleave', hideTip);
    svg.appendChild(hit);
  });
}

// Convergence line chart: one line per config across checkpoint x-values.
function renderLineChart(svg, xValues, seriesByConfig, colors, unit) {
  const W = 620, H = 260, padL = 46, padR = 20, padT = 20, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const allY = Object.values(seriesByConfig).flat();
  const minY = 0;
  const maxY = Math.max(...allY) * 1.15;
  const xMin = Math.min(...xValues), xMax = Math.max(...xValues);
  const xPos = x => padL + ((x - xMin) / (xMax - xMin)) * plotW;
  const yPos = v => padT + plotH - ((v - minY) / (maxY - minY)) * plotH;

  for (let i = 0; i <= 4; i++) {
    const v = (maxY / 4) * i;
    const gy = yPos(v);
    svg.appendChild(svgEl('line', { class: 'gridline', x1: padL, x2: W - padR, y1: gy, y2: gy }));
    const t = svgEl('text', { class: 'axis-label', x: padL - 6, y: gy + 3, 'text-anchor': 'end' });
    t.textContent = fmt(v, 2);
    svg.appendChild(t);
  }
  xValues.forEach(x => {
    const t = svgEl('text', { class: 'axis-label', x: xPos(x), y: H - 12, 'text-anchor': 'middle' });
    t.textContent = x + ' min';
    svg.appendChild(t);
  });

  Object.entries(seriesByConfig).forEach(([cat, ys]) => {
    const color = colors[cat];
    const d = xValues.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xPos(x)} ${yPos(ys[i])}`).join(' ');
    svg.appendChild(svgEl('path', { class: 'line-path', d, stroke: color }));
    xValues.forEach((x, i) => {
      const cx = xPos(x), cy = yPos(ys[i]);
      const dot = svgEl('circle', { class: 'dot', cx, cy, r: 4, fill: color });
      dot.addEventListener('mouseenter', e => showTip(e, `<b>${cat}</b><br>${x} min: ${fmt(ys[i], 3)} ${unit}`));
      dot.addEventListener('mousemove', moveTip);
      dot.addEventListener('mouseleave', hideTip);
      svg.appendChild(dot);
    });
  });
}

function renderLegend(containerId, configs, colors) {
  const el = document.getElementById(containerId);
  configs.forEach(cat => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="swatch" style="background:${colors[cat]}"></span>${cat}`;
    el.appendChild(item);
  });
}

function buildAll(DATA) {
  renderLegend('legend-top', DATA.configs, DATA.colors);

  const metricOrder = ['mean_drift', 'p95_drift', 'rms_heading', 'tension_cov'];
  metricOrder.forEach(key => {
    const m = DATA.metrics[key];
    const svg = document.getElementById('panel-' + key);
    const decimals = key === 'tension_cov' ? 4 : 2;
    renderBarPanel(svg, DATA.configs, m.values, m.ci, DATA.colors, decimals);
  });

  const convSvg = document.getElementById('panel-convergence');
  const xValues = Object.keys(Object.values(DATA.convergence)[0]).map(Number);
  renderLineChart(convSvg, xValues, DATA.convergence, DATA.colors, 'cm');
}
"""


def _fmt(v, decimals=2):
    return f"{v:.{decimals}f}"


def _render_metric_table(data: dict, key: str) -> str:
    m = data["metrics"][key]
    rows = []
    for cat, v, (lo, hi) in zip(data["configs"], m["values"], m["ci"]):
        decimals = 4 if key == "tension_cov" else 2
        rows.append(
            f"<tr><td>{cat}</td><td>{_fmt(v, decimals)}</td>"
            f"<td>[{_fmt(lo, decimals)}, {_fmt(hi, decimals)}]</td></tr>"
        )
    return (
        "<table><thead><tr><th>Configuration</th><th>Value</th><th>95% CI</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_tension_table(data: dict) -> str:
    rows = []
    for cat in data["configs"]:
        vals = data["tensions"][cat]
        cells = "".join(f"<td>{_fmt(v, 3)}</td>" for v in vals)
        rows.append(f"<tr><td>{cat}</td>{cells}</tr>")
    return (
        "<table><thead><tr><th>Configuration</th><th>N1 (N)</th><th>N2 (N)</th><th>N3 (N)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_block_length_table(data: dict) -> str:
    rows = []
    for cat in data["configs"]:
        bl = data["block_length"][cat]
        bound = "≥" if bl["is_lower_bound"] else "="
        rows.append(
            f"<tr><td>{cat}</td><td>{data['n_samples'][cat]}</td>"
            f"<td>{bound} {bl['samples']}</td><td>{bound} {_fmt(bl['sec'], 0)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Configuration</th><th>Samples (n)</th>"
        "<th>Block length L*</th><th>Block length (s)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_cost_tiles(cost: dict) -> str:
    return f"""
    <div class="tiles">
      <div class="tile"><div class="label">Mats</div><div class="value">{cost['n_mats']}</div></div>
      <div class="tile"><div class="label">Plants per mat</div><div class="value">{cost['plants_per_mat']}</div></div>
      <div class="tile"><div class="label">Total plants</div><div class="value">{cost['total_plants']:,}</div></div>
      <div class="tile"><div class="label">Cost per mat</div>
        <div class="value">${cost['cost_per_mat_usd'][0]:,.0f}–${cost['cost_per_mat_usd'][1]:,.0f}</div></div>
      <div class="tile"><div class="label">Total installation cost</div>
        <div class="value">${cost['total_cost_usd'][0]:,.0f}–${cost['total_cost_usd'][1]:,.0f}</div></div>
    </div>
    """


def generate_html_report(results: dict, out_path: Path) -> None:
    data = build_report_data(results)
    configs = data["configs"]
    md = data["metrics"]["mean_drift"]
    drift_reduction_pct = None
    if len(md["values"]) == 2 and md["values"][0]:
        drift_reduction_pct = 100 * (1 - md["values"][1] / md["values"][0])

    summary_tiles = ""
    for cat in configs:
        i = configs.index(cat)
        summary_tiles += f"""
        <div class="tile">
          <div class="label">{cat} — mean radial drift</div>
          <div class="value">{_fmt(md['values'][i])}<span class="unit">cm</span></div>
          <div class="sub">95% CI [{_fmt(md['ci'][i][0])}, {_fmt(md['ci'][i][1])}] · n={data['n_samples'][cat]}</div>
        </div>"""
    if drift_reduction_pct is not None:
        summary_tiles += f"""
        <div class="tile">
          <div class="label">Drift reduction, {configs[1]} vs {configs[0]}</div>
          <div class="value">{_fmt(drift_reduction_pct, 0)}<span class="unit">%</span></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wave-Tank Performance Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Wave-Tank Performance Report</h1>
    <p class="subtitle">Floating mat station-keeping — {' vs '.join(configs)} · 30-min runs, {_fmt(data['dt_sec'], 0)}s sampling</p>
    <div class="legend" id="legend-top"></div>
  </header>

  <section class="section">
    <h2>Executive summary</h2>
    <p class="section-desc">Headline station-keeping performance across the two mooring configurations, with 95% moving-block-bootstrap confidence intervals.</p>
    <div class="tiles">{summary_tiles}</div>
  </section>

  <section class="section">
    <h2>Station-keeping performance</h2>
    <p class="section-desc">Mean and 95th-percentile radial drift, RMS heading deviation, and cross-channel tension coefficient of variation. Whiskers show the 95% CI from a moving block bootstrap ({5000} replicates).</p>
    <div class="panels">
      <div class="panel"><div class="panel-title">Mean radial drift (cm)</div><svg id="panel-mean_drift"></svg></div>
      <div class="panel"><div class="panel-title">95th-pct radial drift (cm)</div><svg id="panel-p95_drift"></svg></div>
      <div class="panel"><div class="panel-title">RMS heading (deg)</div><svg id="panel-rms_heading"></svg></div>
      <div class="panel"><div class="panel-title">Tension CoV</div><svg id="panel-tension_cov"></svg></div>
    </div>
    <div class="panels" style="grid-template-columns: 1fr 1fr;">
      <div class="table-wrap">{_render_metric_table(data, 'mean_drift')}</div>
      <div class="table-wrap">{_render_metric_table(data, 'p95_drift')}</div>
      <div class="table-wrap">{_render_metric_table(data, 'rms_heading')}</div>
      <div class="table-wrap">{_render_metric_table(data, 'tension_cov')}</div>
    </div>
  </section>

  <section class="section">
    <h2>Convergence of mean radial drift</h2>
    <p class="section-desc">Running mean of radial drift computed over increasing observation windows, showing how quickly each configuration's estimate stabilizes.</p>
    <div class="panel" style="max-width:640px;">
      <svg id="panel-convergence"></svg>
    </div>
  </section>

  <section class="section">
    <h2>Mooring line tensions</h2>
    <p class="section-desc">Mean tension per tether channel, and the ACF-derived block length used for the moving block bootstrap (the smallest lag at which the drift autocorrelation first drops below 0.05).</p>
    <div class="panels" style="grid-template-columns: 1fr 1fr;">
      <div class="table-wrap">{_render_tension_table(data)}</div>
      <div class="table-wrap">{_render_block_length_table(data)}</div>
    </div>
  </section>

  <section class="section">
    <h2>Indicative cost summary</h2>
    <p class="section-desc">Reference bill-of-materials for a full-scale installation (per-mat cost bundles HDPE mat, anchors, rope, hardware, plant stock, and installation labor).</p>
    {_render_cost_tiles(data['cost'])}
  </section>

  <footer>
    <p>Generated from results.json by html_report.py. Colors match the Wong (2011) colorblind-safe palette used in the paper's figures (plot.py).</p>
  </footer>
</div>

<div class="tooltip" id="tooltip"></div>
<script>
const REPORT_DATA = {json.dumps(data)};
{_JS}
buildAll(REPORT_DATA);
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
