import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { analyze, settings, risk, covariance, fit } from "../docs/core.js";
let assertions = 0;
function check(condition, message) {
  assertions++;
  assert(condition, message);
}
function near(a, b, path = "") {
  if (typeof a === "number") {
    check(Number.isFinite(a) && Number.isFinite(b), path + " finite");
    check(
      Math.abs(a - b) <= 2e-15 + Math.abs(a) * 2e-12,
      `${path}: ${a} != ${b}`,
    );
  } else if (a === null || typeof a !== "object") check(a === b, path);
  else {
    check(
      JSON.stringify(Object.keys(a)) === JSON.stringify(Object.keys(b)),
      path + " keys",
    );
    for (const k of Object.keys(a)) near(a[k], b[k], path + "." + k);
  }
}
const configs = [];
for (const scenario of [0, 1, 2])
  for (const residual of ["diagonal", "full"])
    for (const shrink of [0, 37, 100])
      for (const weights of [
        [25, 25, 20, 15, 15],
        [90, -70, 30, 10, -45],
        [0, 0, 0, 0, 0],
        [-100, 100, -100, 100, -100],
      ])
        configs.push({ scenario, residual, shrink, weights });
const py = spawnSync(
  "python3",
  [
    "-c",
    "import json,sys;from coveshift.core import analyze;print(json.dumps([analyze(c) for c in json.load(sys.stdin)],allow_nan=False))",
  ],
  {
    input: JSON.stringify(configs),
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
  },
);
assert.equal(py.status, 0, py.stderr);
const oracle = JSON.parse(py.stdout);
for (let n = 0; n < configs.length; n++)
  near(analyze(configs[n]), oracle[n], "case" + n);
const bad = [
  null,
  [],
  {},
  { ...settings(), extra: 1 },
  { ...settings(), scenario: true },
  { ...settings(), scenario: 3 },
  { ...settings(), shrink: -1 },
  { ...settings(), shrink: NaN },
  { ...settings(), weights: new Array(5) },
  { ...settings(), weights: [1, 2, 3, 4, 101] },
  { ...settings(), weights: [1, 2, 3, 4, 1.5] },
  { ...settings(), residual: [] },
];
for (const v of bad) {
  assert.throws(() => analyze(v));
  assertions++;
}
const cfg = settings(),
  original = JSON.stringify(cfg);
analyze(cfg);
check(JSON.stringify(cfg) === original, "no mutation");
assert.deepEqual(
  covariance([
    [1, 3],
    [2, 2],
    [3, 1],
  ]),
  [
    [1, -1],
    [-1, 1],
  ],
);
assertions++;
const analytic = risk(
  [
    [4, -1, 0, 0, 0],
    [-1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
  ],
  [1, 0.5, 0, 0, 0],
);
near(analytic.variance, 3.25);
near(analytic.asset_contributions[1], -0.25 / Math.sqrt(3.25));
for (let n = 0; n < 72; n++) {
  const r = analyze(configs[n]),
    w = r.weights,
    C = r.covariance,
    quadratic = (values) =>
      Math.sqrt(
        values.reduce(
          (s, a, i) => s + values.reduce((u, b, j) => u + a * b * C[i][j], 0),
          0,
        ),
      );
  if (!r.risk.daily_vol) {
    check(r.factor_contributions === null, "zero factor");
    check(r.risk.asset_contributions === null, "zero asset");
    continue;
  }
  for (let i = 0; i < 5; i++) {
    const a = [...w],
      b = [...w];
    a[i] += 1e-5;
    b[i] -= 1e-5;
    check(
      Math.abs(
        (w[i] * (quadratic(a) - quadratic(b))) / 2e-5 -
          r.risk.asset_contributions[i],
      ) < 3e-11,
      "finite difference " + n + "/" + i,
    );
  }
  const total =
    r.factor_contributions.reduce((s, v) => s + v, 0) +
    r.residual_contributions.reduce((s, v) => s + v, 0);
  check(Math.abs(total - r.risk.daily_vol) < 1e-15, "Euler adds");
  if (configs[n].residual === "full" && configs[n].shrink === 0)
    check(
      Math.abs(r.risk.variance - r.train_risk.variance) < 1e-18,
      "full accounting",
    );
}
for (const s of [0, 1, 2]) {
  const m = fit(s);
  check(m.reconstruction_max_error < 1e-18, "reconstruction");
  check(m.orthogonality_max < 1e-18, "orthogonality");
}
console.log(
  `${assertions} JS assertions; ${configs.length} complete Python/JS states; analytic, finite-difference, zero and validation checks PASS`,
);
