export const ASSETS = ["Aster", "Birch", "Cove", "Dune", "Elm"],
  FACTORS = ["Broad", "Tilt"],
  TRAIN = 120,
  TOTAL = 180;
export const LOADINGS = [
  [1.1, 0.3],
  [0.8, -0.6],
  [-0.4, 1.2],
  [0.3, -0.9],
  [1.3, 0.8],
];
export const SCENARIOS = [
  [
    "A stable teaching control",
    "Frozen loadings and shock scales across the split. Finite samples still disagree.",
  ],
  [
    "The residuals move together",
    "A common omitted shock drives correlated asset residuals. Diagonal residual covariance can miss it.",
  ],
  [
    "The later period changes",
    "Both factor shock scales double after day 120. Earlier estimates are frozen, not refitted to the change.",
  ],
];
export function settings() {
  return {
    scenario: 0,
    weights: [25, 25, 20, 15, 15],
    residual: "diagonal",
    shrink: 0,
  };
}
export function validate(v) {
  if (
    v === null ||
    typeof v !== "object" ||
    Object.getPrototypeOf(v) !== Object.prototype ||
    Object.keys(v).sort().join(",") !== Object.keys(settings()).sort().join(",")
  )
    throw new Error("settings require exactly the documented four keys");
  for (const [key, lo, hi] of [
    ["scenario", 0, 2],
    ["shrink", 0, 100],
  ])
    if (!Number.isSafeInteger(v[key]) || v[key] < lo || v[key] > hi)
      throw new Error(`${key} must be an integer in ${lo}..${hi}`);
  if (v.residual !== "diagonal" && v.residual !== "full")
    throw new Error("residual must be diagonal or full");
  if (
    !Array.isArray(v.weights) ||
    v.weights.length !== 5 ||
    [0, 1, 2, 3, 4].some(
      (i) =>
        !Object.hasOwn(v.weights, i) ||
        !Number.isSafeInteger(v.weights[i]) ||
        v.weights[i] < -100 ||
        v.weights[i] > 100,
    )
  )
    throw new Error("weights must be five integer percentages in -100..100");
}
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0),
  mv = (a, x) => a.map((r) => dot(r, x));
export function covariance(rows) {
  const n = rows.length,
    p = rows[0].length,
    mean = Array.from(
      { length: p },
      (_, j) => rows.reduce((s, r) => s + r[j], 0) / n,
    );
  return Array.from({ length: p }, (_, i) =>
    Array.from(
      { length: p },
      (_, j) =>
        rows.reduce((s, r) => s + (r[i] - mean[i]) * (r[j] - mean[j]), 0) /
        (n - 1),
    ),
  );
}
export function dataset(scenario) {
  let state = 9173;
  const rows = [];
  for (let t = 0; t < TOTAL; t++) {
    const u = [];
    for (let k = 0; k < 8; k++) {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      u.push((state / 2 ** 32) * 2 - 1);
    }
    const scale = scenario === 2 && t >= TRAIN ? 2 : 1,
      f = [u[0] * 0.012 * scale, (0.55 * u[0] * 0.012 + u[1] * 0.008) * scale],
      common = scenario === 1 ? u[7] * 0.009 : 0,
      r = LOADINGS.map(
        (b, i) =>
          0.0001 * (i - 2) +
          dot(b, f) +
          u[i + 2] * (0.003 + i * 0.0003) +
          common * (i < 4 ? 1 : -0.5),
      );
    rows.push({
      day: t + 1,
      partition: t < TRAIN ? "train" : "holdout",
      factors: f,
      returns: r,
    });
  }
  return rows;
}
export function fit(scenario = 0) {
  const data = dataset(scenario),
    frows = data.slice(0, TRAIN).map((r) => r.factors),
    yrows = data.slice(0, TRAIN).map((r) => r.returns),
    fm = [0, 1].map((j) => frows.reduce((s, r) => s + r[j], 0) / TRAIN),
    ym = [0, 1, 2, 3, 4].map(
      (j) => yrows.reduce((s, r) => s + r[j], 0) / TRAIN,
    ),
    fc = covariance(frows),
    det = fc[0][0] * fc[1][1] - fc[0][1] ** 2;
  if (!Number.isFinite(det) || det <= 0)
    throw new Error("fixed factor covariance must be nonsingular");
  const beta = [],
    intercept = [];
  for (let i = 0; i < 5; i++) {
    const c = [0, 1].map(
        (j) =>
          frows.reduce(
            (s, r, t) => s + (r[j] - fm[j]) * (yrows[t][i] - ym[i]),
            0,
          ) /
          (TRAIN - 1),
      ),
      b = [
        (c[0] * fc[1][1] - c[1] * fc[0][1]) / det,
        (c[1] * fc[0][0] - c[0] * fc[0][1]) / det,
      ];
    beta.push(b);
    intercept.push(ym[i] - dot(b, fm));
  }
  const residuals = data
      .slice(0, TRAIN)
      .map((r) =>
        beta.map((b, i) => r.returns[i] - intercept[i] - dot(b, r.factors)),
      ),
    ec = covariance(residuals),
    sample = covariance(yrows),
    full = beta.map((bi, i) =>
      beta.map((bj, j) => dot(bi, mv(fc, bj)) + ec[i][j]),
    ),
    orthogonality = beta.map((_, i) =>
      [0, 1].map(
        (j) =>
          frows.reduce((s, r, t) => s + (r[j] - fm[j]) * residuals[t][i], 0) /
          (TRAIN - 1),
      ),
    );
  return {
    data,
    beta,
    intercept,
    factor_covariance: fc,
    residual_covariance: ec,
    sample_covariance: sample,
    holdout_covariance: covariance(data.slice(TRAIN).map((r) => r.returns)),
    reconstruction_max_error: Math.max(
      ...full.flatMap((r, i) => r.map((v, j) => Math.abs(v - sample[i][j]))),
    ),
    orthogonality_max: Math.max(...orthogonality.flat().map(Math.abs)),
  };
}
export function risk(cov, weights) {
  const product = mv(cov, weights);
  let variance = dot(weights, product);
  if (variance < -1e-15 || !Number.isFinite(variance))
    throw new Error("invalid covariance quadratic form");
  variance = Math.max(0, variance);
  const volatility = Math.sqrt(variance);
  return {
    variance,
    daily_vol: volatility,
    annual_vol: volatility * Math.sqrt(252),
    asset_contributions:
      volatility > 0
        ? weights.map((w, i) => (w * product[i]) / volatility)
        : null,
  };
}
export function analyze(value = settings()) {
  validate(value);
  const cfg = { ...value, weights: [...value.weights] },
    model = fit(cfg.scenario),
    weights = cfg.weights.map((w) => w / 100),
    fc = model.factor_covariance.map((r, i) =>
      r.map((v, j) => v * (i !== j ? 1 - cfg.shrink / 100 : 1)),
    ),
    ec = model.residual_covariance.map((r, i) =>
      r.map((v, j) => (cfg.residual === "full" || i === j ? v : 0)),
    ),
    b = model.beta,
    systematic = b.map((bi) => b.map((bj) => dot(bi, mv(fc, bj)))),
    total = systematic.map((r, i) => r.map((v, j) => v + ec[i][j])),
    exposure = [0, 1].map((j) =>
      weights.reduce((s, w, i) => s + w * b[i][j], 0),
    ),
    result = risk(total, weights),
    sigma = result.daily_vol,
    fx = mv(fc, exposure),
    ew = mv(ec, weights),
    factor_variance = dot(exposure, fx),
    residual_variance = dot(weights, ew),
    factor_contributions = sigma
      ? exposure.map((v, i) => (v * fx[i]) / sigma)
      : null,
    residual_contributions = sigma
      ? weights.map((w, i) => (w * ew[i]) / sigma)
      : null,
    daily = model.data.map((r) => ({
      day: r.day,
      partition: r.partition,
      return: dot(weights, r.returns),
    }));
  return {
    settings: cfg,
    name: SCENARIOS[cfg.scenario][0],
    description: SCENARIOS[cfg.scenario][1],
    model,
    weights,
    net_weight: weights.reduce((a, b) => a + b, 0),
    gross_weight: weights.reduce((a, b) => a + Math.abs(b), 0),
    factor_covariance: fc,
    residual_covariance: ec,
    systematic_covariance: systematic,
    covariance: total,
    exposure,
    risk: result,
    factor_variance,
    residual_variance,
    factor_contributions,
    residual_contributions,
    train_risk: risk(model.sample_covariance, weights),
    holdout_risk: risk(model.holdout_covariance, weights),
    daily,
  };
}
