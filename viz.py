"""runs/*.json -> one self-contained interactive page.

The deliverable of this task is a Pareto front, and a front is only readable next to
the fronts it is supposed to beat. So the page puts all repair methods on the same
axes, keeps the generated text behind each point, and shows the training curves that
produced the models — the numbers and the samples that justify them in one place.

    python viz.py                      # -> runs/report.html
"""

import argparse
import json
import pathlib

TEMPLATE = """<title>Стиринг — парето-фронты</title>
<style>
:root { --bg:#fff; --fg:#111; --mut:#666; --line:#ddd; --acc:#2b6cb0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038; --acc:#7aa7dd; } }
:root[data-theme=dark] { --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038; --acc:#7aa7dd; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,system-ui,sans-serif;
       margin:0 auto; max-width:1100px; padding:24px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:26px 0 8px; font-weight:600; }
p.note { color:var(--mut); margin:2px 0 12px; }
table { border-collapse:collapse; font-size:13px; width:100%; }
td,th { padding:4px 10px 4px 0; text-align:right; vertical-align:top; }
th:first-child,td:first-child,td.txt { text-align:left; }
td.txt { color:var(--mut); font-size:12px; max-width:520px; }
th { border-bottom:1px solid var(--line); font-weight:600; }
tr.sep td { border-top:1px solid var(--line); }
.card { border:1px solid var(--line); border-radius:6px; padding:8px 10px; overflow-x:auto;
        display:inline-block; margin-right:12px; }
.card b { font-size:12px; } .card span { color:var(--mut); font-size:11px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--mut); margin:6px 0; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
svg { display:block; margin-top:4px; }
.ax { stroke:var(--line); } .tick { fill:var(--mut); font-size:10px; }
</style>
<h1>Стиринг: чем платим за концепт</h1>
<p class="note">Ось x — перплексия продолжений под чистой моделью, ось y — присутствие
концепта. Идеальный метод уводит фронт влево-вверх.</p>
<div id="app"></div>
<script>
const DATA = __DATA__;
const PAL = {none: "#c05621", mse: "#2b6cb0", glp: "#2f855a"};
const SVG = new Set(["svg", "g", "path", "line", "text", "rect", "circle"]);
function el(tag, attrs, kids) {
  const n = document.createElementNS(SVG.has(tag) ? "http://www.w3.org/2000/svg"
    : "http://www.w3.org/1999/xhtml", tag);
  for (const k in (attrs || {})) n.setAttribute(k, attrs[k]);
  for (const c of (kids || [])) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return n;
}
const fmt = v => Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)
  ? v.toExponential(1) : (+v.toFixed(3)).toString();

function chart(series, o) {
  o = Object.assign({w: 480, h: 300, logx: false, logy: false, pad: 42, dots: false}, o);
  const pts = series.flatMap(s => s.pts).filter(p => isFinite(p[0]) && isFinite(p[1]));
  if (!pts.length) return el("svg", {width: o.w, height: o.h});
  const tx = v => o.logx ? Math.log10(Math.max(v, 1e-9)) : v;
  const ty = v => o.logy ? Math.log10(Math.max(v, 1e-9)) : v;
  let x0 = Math.min(...pts.map(p => tx(p[0]))), x1 = Math.max(...pts.map(p => tx(p[0])));
  let y0 = Math.min(...pts.map(p => ty(p[1]))), y1 = Math.max(...pts.map(p => ty(p[1])));
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { y0 -= .5; y1 += .5; }
  const mx = (x1 - x0) * .08, my = (y1 - y0) * .08;
  x0 -= mx; x1 += mx; y0 -= my; y1 += my;
  const X = v => o.pad + (tx(v) - x0) / (x1 - x0) * (o.w - o.pad - 10);
  const Y = v => o.h - 24 - (ty(v) - y0) / (y1 - y0) * (o.h - 34);
  const g = el("svg", {width: o.w, height: o.h});
  g.appendChild(el("line", {x1: o.pad, y1: o.h - 24, x2: o.w - 10, y2: o.h - 24, class: "ax"}));
  g.appendChild(el("line", {x1: o.pad, y1: 12, x2: o.pad, y2: o.h - 24, class: "ax"}));
  const inv = (v, log) => log ? Math.pow(10, v) : v;
  g.appendChild(el("text", {x: 2, y: 16, class: "tick"}, [fmt(inv(y1 - my, o.logy))]));
  g.appendChild(el("text", {x: 2, y: o.h - 26, class: "tick"}, [fmt(inv(y0 + my, o.logy))]));
  g.appendChild(el("text", {x: o.pad, y: o.h - 8, class: "tick"}, [fmt(inv(x0 + mx, o.logx))]));
  g.appendChild(el("text", {x: o.w - 10, y: o.h - 8, class: "tick", "text-anchor": "end"},
    [fmt(inv(x1 - mx, o.logx))]));
  for (const s of series) {
    const p = s.pts.filter(q => isFinite(q[0]) && isFinite(q[1]));
    g.appendChild(el("path", {fill: "none", stroke: s.color || "#888", "stroke-width": 1.8,
      d: p.map((q, i) => (i ? "L" : "M") + X(q[0]) + " " + Y(q[1])).join(" ")}));
    if (o.dots) for (const q of p)
      g.appendChild(el("circle", {cx: X(q[0]), cy: Y(q[1]), r: 3.5, fill: s.color || "#888"}));
  }
  return g;
}
function card(title, sub, svg) {
  return el("div", {class: "card"}, [el("b", {}, [title]), sub ? el("span", {}, [" — " + sub]) : "", svg]);
}
function legend(names) {
  return el("div", {class: "legend"}, names.map(n =>
    el("span", {}, [el("i", {style: "background:" + (PAL[n] || "#888")}), n])));
}

const app = document.getElementById("app");
const kinds = [...new Set(DATA.pareto.map(r => r.repair))];

// --- парето
if (DATA.pareto.length) {
  app.appendChild(el("h2", {}, ["Парето: связность против концепта"]));
  app.appendChild(legend(kinds));
  const box = el("div", {});
  const by = k => DATA.pareto.filter(r => r.repair === k).sort((a, b) => a.alpha - b.alpha);
  box.appendChild(card("концепт от перплексии", "точки — alpha по возрастанию",
    chart(kinds.map(k => ({pts: by(k).map(r => [r.ppl, r.concept]), color: PAL[k]})),
      {logx: true, dots: true})));
  box.appendChild(card("dist-2 от концепта", "разнообразие биграмм",
    chart(kinds.map(k => ({pts: by(k).map(r => [r.concept, r.dist2]), color: PAL[k]})),
      {dots: true, w: 380})));
  app.appendChild(box);

  app.appendChild(el("h2", {}, ["Точки фронта и что модель при этом писала"]));
  const t = el("table", {}, [el("tr", {}, ["метод", "alpha", "ppl", "dist-2", "концепт", "продолжение"]
    .map(h => el("th", {}, [h])))]);
  let prev = null;
  for (const r of DATA.pareto) {
    const tr = el("tr", r.repair !== prev && prev !== null ? {class: "sep"} : {});
    prev = r.repair;
    for (const [v, cls] of [[r.repair, ""], [fmt(r.alpha), ""], [fmt(r.ppl), ""],
                            [fmt(r.dist2), ""], [fmt(r.concept), ""], [r.sample || "", "txt"]])
      tr.appendChild(el("td", cls ? {class: cls} : {}, [String(v)]));
    t.appendChild(tr);
  }
  app.appendChild(t);
}

// --- обучение моделей активаций
if (Object.keys(DATA.training).length) {
  app.appendChild(el("h2", {}, ["Обучение моделей активаций"]));
  app.appendChild(el("p", {class: "note"}, ["Лоссы у двух целей разные по смыслу и "
    + "несравнимы между собой: у flow это ошибка поля скоростей, у mse — ошибка "
    + "восстановления. Смотреть надо на форму, а не на уровень."]));
  const box = el("div", {});
  for (const [tag, rows] of Object.entries(DATA.training))
    box.appendChild(card(tag, "шаг, лог", chart(
      [{pts: rows.map(r => [r.step, r.train_loss]), color: "#888"},
       {pts: rows.map(r => [r.step, r.val_loss]), color: PAL[tag] || "#2b6cb0"}],
      {logy: true, w: 380, h: 240})));
  app.appendChild(box);
  app.appendChild(el("div", {class: "legend"}, ["серое — train, цветное — val"]));
}
</script>
"""


def collect(root, only=None):
    root = pathlib.Path(root)
    pareto = []
    for path in sorted(root.glob("*.json")):
        if only and path.stem not in only:
            continue
        blob = json.loads(path.read_text())
        rows = blob.get("rows") or []
        if rows and "repair" in rows[0]:
            pareto += rows
        elif rows:  # baseline.py --sweep: тот же наивный фронт, но без колонки
            pareto += [{**r, "repair": "none"} for r in rows]
    training = {}
    for path in sorted(root.glob("*/history.jsonl")):
        training[path.parent.name] = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {"pareto": pareto, "training": training}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/report.html")
    p.add_argument("--only", nargs="+", default=None,
                   help="какие runs/<tag>.json брать; концепт-скоры разных векторов "
                        "на одном графике смысла не имеют")
    args = p.parse_args()

    data = collect(args.runs, args.only)
    pathlib.Path(args.out).write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    print(f"{len(data['pareto'])} точек фронта, {len(data['training'])} прогонов -> {args.out}")


if __name__ == "__main__":
    main()
