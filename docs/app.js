import { analyze, settings, ASSETS, FACTORS } from "./core.js";
const $ = (id) => document.getElementById(id),
  pct = (x) => (x * 100).toFixed(3) + "%",
  num = (x, n = 3) => x.toFixed(n),
  text = (id, v) => ($(id).textContent = v);
let current;
function cell(tag, value) {
  const e = document.createElement(tag);
  e.textContent = value;
  return e;
}
function bars(id, labels, values) {
  const root = $(id);
  root.replaceChildren();
  if (values === null) {
    root.setAttribute(
      "aria-label",
      "Euler contributions undefined at zero total risk.",
    );
    root.append(cell("p", "Undefined at zero total risk."));
    return;
  }
  root.setAttribute(
    "aria-label",
    labels
      .map((label, i) => label + ": " + pct(values[i] * Math.sqrt(252)))
      .join("; ") + ". Annualized volatility contributions.",
  );
  const maximum = Math.max(...values.map(Math.abs), 1e-15);
  labels.forEach((label, i) => {
    const row = cell("div", "");
    row.className = "bar-row";
    const track = cell("span", "");
    track.className = "bar-track";
    const fill = cell("span", "");
    fill.className = "bar-fill" + (values[i] < 0 ? " negative" : "");
    fill.style.width = (Math.abs(values[i]) / maximum) * 49 + "%";
    track.append(fill);
    row.append(
      cell("span", label),
      track,
      cell("span", pct(values[i] * Math.sqrt(252))),
    );
    row.lastChild.className = "bar-number";
    root.append(row);
  });
}
function matrix(id, values, labels) {
  const t = $(id);
  t.replaceChildren();
  const head = document.createElement("thead"),
    tr = document.createElement("tr");
  tr.append(cell("th", "×10⁶"));
  labels.forEach((l) => {
    const th = cell("th", l);
    th.scope = "col";
    tr.append(th);
  });
  head.append(tr);
  t.append(head);
  const body = document.createElement("tbody");
  values.forEach((r, i) => {
    const row = document.createElement("tr"),
      th = cell("th", labels[i]);
    th.scope = "row";
    row.append(th);
    r.forEach((v) => {
      const td = cell("td", (v * 1e6).toFixed(3));
      td.className = v < 0 ? "negative-cell" : v > 0 ? "positive-cell" : "";
      row.append(td);
    });
    body.append(row);
  });
  t.append(body);
}
function svgNode(tag, attrs = {}, value) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v)));
  if (value !== undefined) n.textContent = value;
  return n;
}
function chart(r) {
  const svg = $("returns-chart"),
    width = Math.max(235, svg.clientWidth),
    height = svg.clientHeight,
    left = 45,
    right = 12,
    top = 27,
    bottom = 31,
    w = width - left - right,
    h = height - top - bottom,
    max = Math.max(...r.daily.map((v) => Math.abs(v.return)), 0.0001) * 1.08,
    x = (i) => left + (i / 179) * w,
    y = (v) => top + h / 2 - ((v / max) * h) / 2,
    split = x(119.5);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  svg.append(
    svgNode("rect", {
      x: split,
      y: top,
      width: left + w - split,
      height: h,
      fill: "#f4e6d8",
    }),
  );
  for (const v of [-max, 0, max]) {
    svg.append(
      svgNode("line", {
        x1: left,
        x2: left + w,
        y1: y(v),
        y2: y(v),
        class: "grid",
      }),
    );
    svg.append(
      svgNode(
        "text",
        { x: left - 7, y: y(v) + 3, "text-anchor": "end" },
        (v * 100).toFixed(max < 0.001 ? 3 : 1) + "%",
      ),
    );
  }
  svg.append(
    svgNode("path", {
      d: r.daily
        .map(
          (v, i) =>
            (i ? "L" : "M") + x(i).toFixed(2) + " " + y(v.return).toFixed(2),
        )
        .join(" "),
      class: "return-line",
    }),
  );
  svg.append(
    svgNode("line", {
      x1: split,
      x2: split,
      y1: top,
      y2: top + h,
      class: "split",
    }),
  );
  svg.append(
    svgNode(
      "text",
      { x: left, y: 14 },
      width < 500 ? "EARLIER" : "EARLIER · 120 FITTED DAYS",
    ),
  );
  svg.append(
    svgNode(
      "text",
      { x: width - right, y: 14, "text-anchor": "end" },
      width < 500 ? "LATER" : "LATER · 60 HELD-OUT DAYS",
    ),
  );
  for (const i of [0, 59, 119, 179])
    svg.append(
      svgNode(
        "text",
        {
          x: x(i),
          y: height - 8,
          "text-anchor": i === 0 ? "start" : i === 179 ? "end" : "middle",
        },
        String(i + 1),
      ),
    );
}
function render(r) {
  current = r;
  text("scenario-name", r.name);
  text("scenario-desc", r.description);
  text(
    "book",
    `Net ${pct(r.net_weight)} · Gross ${pct(r.gross_weight)} · reference-capital book`,
  );
  text("model-vol", pct(r.risk.annual_vol));
  text("train-vol", pct(r.train_risk.annual_vol));
  text("holdout-vol", pct(r.holdout_risk.annual_vol));
  text(
    "model-mode",
    `${r.settings.residual} Ψ · factor shrink ${r.settings.shrink}%`,
  );
  bars("asset-bars", ASSETS, r.risk.asset_contributions);
  const components = r.factor_contributions
    ? [...r.factor_contributions, ...r.residual_contributions]
    : null;
  bars(
    "factor-bars",
    [...FACTORS, ...ASSETS.map((name) => "ε " + name)],
    components,
  );
  const sum = (a) => a.reduce((a, b) => a + b, 0);
  text(
    "asset-sum",
    r.risk.asset_contributions
      ? `Signed sum ${pct(sum(r.risk.asset_contributions) * Math.sqrt(252))} = total annualized vol`
      : "Euler contributions undefined when σ = 0",
  );
  text(
    "factor-sum",
    components
      ? `Signed sum ${pct(sum(components) * Math.sqrt(252))} = total annualized vol`
      : "Euler contributions undefined when σ = 0",
  );
  text(
    "factor-share",
    r.risk.variance ? pct(r.factor_variance / r.risk.variance) : "Undefined",
  );
  text(
    "residual-share",
    r.risk.variance ? pct(r.residual_variance / r.risk.variance) : "Undefined",
  );
  text(
    "variance-gap",
    (r.risk.variance - r.train_risk.variance).toExponential(5),
  );
  matrix("covariance", r.covariance, ASSETS);
  matrix("sample", r.model.sample_covariance, ASSETS);
  matrix("residual-matrix", r.residual_covariance, ASSETS);
  matrix("factors", r.factor_covariance, FACTORS);
  text(
    "exposure",
    FACTORS.map((f, i) => `${f}: ${num(r.exposure[i], 5)}`).join(" · "),
  );
  $("loadings").replaceChildren();
  ASSETS.forEach((name, i) => {
    const row = document.createElement("tr"),
      th = cell("th", name);
    th.scope = "row";
    row.append(
      th,
      ...[
        r.model.intercept[i].toFixed(7),
        ...r.model.beta[i].map((v) => v.toFixed(5)),
        r.risk.asset_contributions
          ? pct(r.risk.asset_contributions[i] * Math.sqrt(252))
          : "Undefined",
        r.residual_contributions
          ? pct(r.residual_contributions[i] * Math.sqrt(252))
          : "Undefined",
      ].map((v) => cell("td", v)),
    );
    $("loadings").append(row);
  });
  text(
    "identity",
    `Full Ψ, unshrunk F reconstruction max |Σ − sample|: ${r.model.reconstruction_max_error.toExponential(3)} daily return². Factor-residual max |sample covariance|: ${r.model.orthogonality_max.toExponential(3)}. Floating-point zero, not an accuracy score.`,
  );
  $("data").replaceChildren();
  r.model.data.forEach((v, i) => {
    const row = document.createElement("tr");
    row.append(
      ...[
        v.day,
        v.partition,
        ...v.factors.map((x) => x.toFixed(7)),
        ...v.returns.map((x) => x.toFixed(7)),
        r.daily[i].return.toFixed(7),
      ].map((v) => cell("td", v)),
    );
    $("data").append(row);
  });
  chart(r);
  document.body.classList.remove("stale");
  text("status", "APPLIED");
  text("error", "");
}
function read() {
  const integer = (id) => {
    const s = $(id).value;
    if (!/^-?\d+$/.test(s))
      throw new Error("Enter whole numbers in every field.");
    return Number(s);
  };
  return {
    scenario: integer("scenario"),
    residual: $("residual").value,
    shrink: integer("shrink"),
    weights: ASSETS.map((_, i) => integer("w" + i)),
  };
}
function setControls(s) {
  $("scenario").value = s.scenario;
  $("residual").value = s.residual;
  $("shrink").value = s.shrink;
  s.weights.forEach((w, i) => ($("w" + i).value = w));
}
$("controls").addEventListener("input", () => {
  document.body.classList.add("stale");
  text("status", "UNAPPLIED");
});
$("controls").addEventListener("change", () => {
  document.body.classList.add("stale");
  text("status", "UNAPPLIED");
});
$("controls").addEventListener("submit", (e) => {
  e.preventDefault();
  try {
    render(analyze(read()));
  } catch (error) {
    text("error", error.message);
    text("status", "CHECK INPUT");
    document.body.classList.add("stale");
  }
});
$("reset").addEventListener("click", () => {
  const cfg = settings();
  setControls(cfg);
  render(analyze(cfg));
});
let resize;
window.addEventListener("resize", () => {
  clearTimeout(resize);
  resize = setTimeout(() => {
    if (current) chart(current);
  }, 100);
});
setControls(settings());
render(analyze());
