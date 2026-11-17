"""Synthetic fitted factor covariance and signed Euler risk accounting."""

import math
from copy import deepcopy

ASSETS = ("Aster", "Birch", "Cove", "Dune", "Elm")
FACTORS = ("Broad", "Tilt")
LOADINGS = ((1.1, 0.3), (0.8, -0.6), (-0.4, 1.2), (0.3, -0.9), (1.3, 0.8))
SCENARIOS = (
    (
        "A stable teaching control",
        "Frozen loadings and shock scales across the split. Finite samples still disagree.",
    ),
    (
        "The residuals move together",
        "A common omitted shock drives correlated asset residuals. Diagonal residual covariance can miss it.",
    ),
    (
        "The later period changes",
        "Both factor shock scales double after day 120. Earlier estimates are frozen, not refitted to the change.",
    ),
)
DEFAULT = {
    "scenario": 0,
    "weights": [25, 25, 20, 15, 15],
    "residual": "diagonal",
    "shrink": 0,
}
TRAIN = 120
TOTAL = 180


def settings():
    return deepcopy(DEFAULT)


def validate(v):
    if type(v) is not dict or set(v) != set(DEFAULT):
        raise ValueError("settings require exactly the documented four keys")
    for key, lo, hi in (("scenario", 0, 2), ("shrink", 0, 100)):
        if type(v[key]) is not int or not lo <= v[key] <= hi:
            raise ValueError(key + " must be an integer in " + str(lo) + ".." + str(hi))
    if type(v["residual"]) is not str or v["residual"] not in ("diagonal", "full"):
        raise ValueError("residual must be diagonal or full")
    if (
        type(v["weights"]) is not list
        or len(v["weights"]) != 5
        or any(type(w) is not int or not -100 <= w <= 100 for w in v["weights"])
    ):
        raise ValueError("weights must be five integer percentages in -100..100")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(a, x):
    return [dot(row, x) for row in a]


def covariance(rows):
    """Internal trusted nonempty rectangular matrix, at least two rows."""
    n, p = len(rows), len(rows[0])
    mean = [sum(row[j] for row in rows) / n for j in range(p)]
    return [
        [
            sum((row[i] - mean[i]) * (row[j] - mean[j]) for row in rows) / (n - 1)
            for j in range(p)
        ]
        for i in range(p)
    ]


def dataset(scenario):
    """Fixed 32-bit LCG; eight uniform draws each day; no market observations."""
    state = 9173
    rows = []
    for t in range(TOTAL):
        u = []
        for _ in range(8):
            state = (1664525 * state + 1013904223) % 2**32
            u.append(state / 2**32 * 2 - 1)
        scale = 2 if scenario == 2 and t >= TRAIN else 1
        f = [u[0] * 0.012 * scale, (0.55 * u[0] * 0.012 + u[1] * 0.008) * scale]
        common = u[7] * 0.009 if scenario == 1 else 0
        r = [
            0.0001 * (i - 2)
            + dot(LOADINGS[i], f)
            + u[i + 2] * (0.003 + i * 0.0003)
            + common * (1 if i < 4 else -0.5)
            for i in range(5)
        ]
        rows.append(
            {
                "day": t + 1,
                "partition": "train" if t < TRAIN else "holdout",
                "factors": f,
                "returns": r,
            }
        )
    return rows


def fit(scenario=0):
    """Internal: fit only the fixed first 120 synthetic rows."""
    data = dataset(scenario)
    frows = [r["factors"] for r in data[:TRAIN]]
    yrows = [r["returns"] for r in data[:TRAIN]]
    fm = [sum(row[j] for row in frows) / TRAIN for j in range(2)]
    ym = [sum(row[j] for row in yrows) / TRAIN for j in range(5)]
    fc = covariance(frows)
    determinant = fc[0][0] * fc[1][1] - fc[0][1] ** 2
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError("fixed factor covariance must be nonsingular")
    beta, intercept = [], []
    for i in range(5):
        cross = [
            sum((frows[t][j] - fm[j]) * (yrows[t][i] - ym[i]) for t in range(TRAIN))
            / (TRAIN - 1)
            for j in range(2)
        ]
        b = [
            (cross[0] * fc[1][1] - cross[1] * fc[0][1]) / determinant,
            (cross[1] * fc[0][0] - cross[0] * fc[0][1]) / determinant,
        ]
        beta.append(b)
        intercept.append(ym[i] - dot(b, fm))
    residuals = [
        [
            row["returns"][i] - intercept[i] - dot(beta[i], row["factors"])
            for i in range(5)
        ]
        for row in data[:TRAIN]
    ]
    ec = covariance(residuals)
    sample = covariance(yrows)
    full = [
        [dot(beta[i], mv(fc, beta[j])) + ec[i][j] for j in range(5)] for i in range(5)
    ]
    residual_orthogonality = [
        [
            sum((frows[t][j] - fm[j]) * residuals[t][i] for t in range(TRAIN))
            / (TRAIN - 1)
            for j in range(2)
        ]
        for i in range(5)
    ]
    return {
        "data": data,
        "beta": beta,
        "intercept": intercept,
        "factor_covariance": fc,
        "residual_covariance": ec,
        "sample_covariance": sample,
        "holdout_covariance": covariance([r["returns"] for r in data[TRAIN:]]),
        "reconstruction_max_error": max(
            abs(full[i][j] - sample[i][j]) for i in range(5) for j in range(5)
        ),
        "orthogonality_max": max(abs(x) for row in residual_orthogonality for x in row),
    }


def risk(cov, weights):
    """Internal trusted positive-semidefinite covariance and finite weights."""
    product = mv(cov, weights)
    variance = dot(weights, product)
    if variance < -1e-15 or not math.isfinite(variance):
        raise ValueError("invalid covariance quadratic form")
    variance = max(0, variance)  # only floating roundoff, not an estimator repair
    volatility = math.sqrt(variance)
    return {
        "variance": variance,
        "daily_vol": volatility,
        "annual_vol": volatility * math.sqrt(252),
        "asset_contributions": (
            [weights[i] * product[i] / volatility for i in range(5)]
            if volatility > 0
            else None
        ),
    }


def analyze(value=None):
    cfg = settings() if value is None else value
    validate(cfg)
    cfg = deepcopy(cfg)
    model = fit(cfg["scenario"])
    weights = [w / 100 for w in cfg["weights"]]
    fc = [
        [
            model["factor_covariance"][i][j]
            * (1 - cfg["shrink"] / 100 if i != j else 1)
            for j in range(2)
        ]
        for i in range(2)
    ]
    ec = [
        [
            (
                model["residual_covariance"][i][j]
                if cfg["residual"] == "full" or i == j
                else 0
            )
            for j in range(5)
        ]
        for i in range(5)
    ]
    b = model["beta"]
    systematic = [[dot(b[i], mv(fc, b[j])) for j in range(5)] for i in range(5)]
    total = [[systematic[i][j] + ec[i][j] for j in range(5)] for i in range(5)]
    exposure = [sum(weights[i] * b[i][j] for i in range(5)) for j in range(2)]
    result = risk(total, weights)
    sigma = result["daily_vol"]
    fx, ew = mv(fc, exposure), mv(ec, weights)
    factor_variance = dot(exposure, fx)
    residual_variance = dot(weights, ew)
    factor_contributions = (
        [exposure[i] * fx[i] / sigma for i in range(2)] if sigma else None
    )
    residual_contributions = (
        [weights[i] * ew[i] / sigma for i in range(5)] if sigma else None
    )
    daily = [
        {
            "day": row["day"],
            "partition": row["partition"],
            "return": dot(weights, row["returns"]),
        }
        for row in model["data"]
    ]
    return {
        "settings": cfg,
        "name": SCENARIOS[cfg["scenario"]][0],
        "description": SCENARIOS[cfg["scenario"]][1],
        "model": model,
        "weights": weights,
        "net_weight": sum(weights),
        "gross_weight": sum(abs(w) for w in weights),
        "factor_covariance": fc,
        "residual_covariance": ec,
        "systematic_covariance": systematic,
        "covariance": total,
        "exposure": exposure,
        "risk": result,
        "factor_variance": factor_variance,
        "residual_variance": residual_variance,
        "factor_contributions": factor_contributions,
        "residual_contributions": residual_contributions,
        "train_risk": risk(model["sample_covariance"], weights),
        "holdout_risk": risk(model["holdout_covariance"], weights),
        "daily": daily,
    }
