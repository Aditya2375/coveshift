import json
import math
import random
import subprocess
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch
from coveshift import core as c


def qr_solve(x, y):
    """Independent modified Gram-Schmidt QR, no production covariance/solver."""
    columns = [list(v) for v in zip(*x)]
    q, r = [], [[0.0] * len(columns) for _ in columns]
    for j, column in enumerate(columns):
        v = column[:]
        for i in range(j):
            r[i][j] = math.fsum(a * b for a, b in zip(q[i], v))
            v = [a - r[i][j] * b for a, b in zip(v, q[i])]
        r[j][j] = math.sqrt(math.fsum(a * a for a in v))
        q.append([a / r[j][j] for a in v])
    rhs = [math.fsum(a * b for a, b in zip(qi, y)) for qi in q]
    b = [0.0] * len(q)
    for i in reversed(range(len(q))):
        b[i] = (rhs[i] - math.fsum(r[i][j] * b[j] for j in range(i + 1, len(q)))) / r[
            i
        ][i]
    return b


def pair_cov(a, b):
    n = len(a)
    return math.fsum(
        (a[i] - a[j]) * (b[i] - b[j]) for i in range(n) for j in range(i)
    ) / (n * (n - 1))


class RiskTests(unittest.TestCase):
    def test_01_covariance_known_fixture_and_pairwise_oracle(self):
        self.assertEqual(c.covariance([[1, 3], [2, 2], [3, 1]]), [[1, -1], [-1, 1]])
        for scenario in range(3):
            m = c.fit(scenario)
            for key, field, start, stop in (
                ("factor_covariance", "factors", 0, 120),
                ("sample_covariance", "returns", 0, 120),
                ("holdout_covariance", "returns", 120, 180),
            ):
                columns = list(zip(*(r[field] for r in m["data"][start:stop])))
                for i, a in enumerate(columns):
                    for j, b in enumerate(columns):
                        self.assertAlmostEqual(
                            m[key][i][j], pair_cov(a, b), delta=2e-18
                        )

    def test_02_ols_independent_qr_and_orthogonality(self):
        for scenario in range(3):
            m = c.fit(scenario)
            x = [[1, *r["factors"]] for r in m["data"][:120]]
            for i in range(5):
                expected = qr_solve(x, [r["returns"][i] for r in m["data"][:120]])
                for a, b in zip(expected, [m["intercept"][i], *m["beta"][i]]):
                    self.assertAlmostEqual(a, b, delta=1e-13)
            self.assertLess(m["orthogonality_max"], 1e-18)
            self.assertLess(m["reconstruction_max_error"], 2e-18)

    def test_03_generator_recurrence_partition_and_later_jump(self):
        stable, omitted, shifted = [c.dataset(i) for i in range(3)]
        self.assertEqual(len(stable), 180)
        self.assertEqual([r["day"] for r in stable], list(range(1, 181)))
        self.assertEqual(shifted[:120], stable[:120])
        self.assertEqual(
            shifted[120]["factors"], [v * 2 for v in stable[120]["factors"]]
        )
        state, u = 9173, []
        for _ in range(8):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            u.append(state / 2147483648 - 1)
        self.assertEqual(
            stable[0]["factors"], [u[0] * 0.012, (0.55 * u[0] * 0.012 + u[1] * 0.008)]
        )
        for row0, row1 in zip(stable, omitted):
            self.assertEqual(row0["factors"], row1["factors"])
        self.assertEqual(stable[119]["partition"], "train")
        self.assertEqual(stable[120]["partition"], "holdout")

    def test_04_holdout_cannot_leak_into_fit(self):
        for scenario in range(3):
            expected = c.fit(scenario)
            changed = deepcopy(expected["data"])
            for row in changed[120:]:
                row["factors"] = [9e3, -1e4]
                row["returns"] = [42.0, -19.0, 17.0, 7.0, -12.0]
            with patch.object(c, "dataset", return_value=changed):
                actual = c.fit(scenario)
            for key in (
                "beta",
                "intercept",
                "factor_covariance",
                "residual_covariance",
                "sample_covariance",
                "orthogonality_max",
                "reconstruction_max_error",
            ):
                self.assertEqual(expected[key], actual[key])
            self.assertNotEqual(
                expected["holdout_covariance"], actual["holdout_covariance"]
            )

    def test_05_analytic_signed_euler_and_zero(self):
        m = [
            [4.0, -1.0, 0, 0, 0],
            [-1.0, 1.0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
        r = c.risk(m, [1, 0.5, 0, 0, 0])
        self.assertEqual(r["variance"], 3.25)
        self.assertAlmostEqual(r["asset_contributions"][0], 3.5 / math.sqrt(3.25))
        self.assertAlmostEqual(r["asset_contributions"][1], -0.25 / math.sqrt(3.25))
        z = c.analyze({**c.settings(), "weights": [0] * 5})
        for key in ("risk", "train_risk", "holdout_risk"):
            self.assertEqual(z[key]["variance"], 0)
            self.assertIsNone(z[key]["asset_contributions"])
        self.assertIsNone(z["factor_contributions"])
        self.assertIsNone(z["residual_contributions"])
        self.assertTrue(any(x < 0 for x in c.analyze()["risk"]["asset_contributions"]))

    def test_06_all_scenarios_modes_shrink_signed_weights_accounting(self):
        rng = random.Random(20261117)
        for scenario in range(3):
            for mode in ("diagonal", "full"):
                for shrink in (0, 37, 100):
                    for _ in range(20):
                        weights = [rng.randint(-100, 100) for _ in range(5)]
                        r = c.analyze(
                            {
                                "scenario": scenario,
                                "residual": mode,
                                "shrink": shrink,
                                "weights": weights,
                            }
                        )
                        variance = r["risk"]["variance"]
                        expected = math.fsum(
                            r["weights"][i] * r["weights"][j] * r["covariance"][i][j]
                            for i in range(5)
                            for j in range(5)
                        )
                        self.assertAlmostEqual(variance, expected, delta=1e-18)
                        self.assertGreaterEqual(variance, 0)
                        self.assertAlmostEqual(
                            variance,
                            r["factor_variance"] + r["residual_variance"],
                            delta=1e-18,
                        )
                        self.assertAlmostEqual(
                            sum(r["risk"]["asset_contributions"]),
                            r["risk"]["daily_vol"],
                            delta=1e-15,
                        )
                        self.assertAlmostEqual(
                            sum(r["factor_contributions"])
                            + sum(r["residual_contributions"]),
                            r["risk"]["daily_vol"],
                            delta=1e-15,
                        )
                        self.assertAlmostEqual(
                            r["risk"]["annual_vol"] ** 2, 252 * variance, delta=1e-15
                        )
                        self.assertEqual(r["weights"], [w / 100 for w in weights])
                        if mode == "full" and shrink == 0:
                            self.assertAlmostEqual(
                                variance, r["train_risk"]["variance"], delta=2e-18
                            )
                        for i in range(5):
                            for j in range(5):
                                self.assertAlmostEqual(
                                    r["covariance"][i][j],
                                    r["covariance"][j][i],
                                    delta=1e-18,
                                )

    def test_07_euler_finite_difference_independent_of_formula(self):
        for scenario in range(3):
            for mode in ("diagonal", "full"):
                r = c.analyze(
                    {
                        "scenario": scenario,
                        "residual": mode,
                        "shrink": 37,
                        "weights": [90, -70, 30, 10, -45],
                    }
                )
                w, cov = r["weights"], r["covariance"]

                def sigma(values):
                    return math.sqrt(
                        math.fsum(
                            values[i] * values[j] * cov[i][j]
                            for i in range(5)
                            for j in range(5)
                        )
                    )

                for i in range(5):
                    plus, minus = w[:], w[:]
                    plus[i] += 1e-5
                    minus[i] -= 1e-5
                    numeric = w[i] * (sigma(plus) - sigma(minus)) / 2e-5
                    self.assertAlmostEqual(
                        numeric, r["risk"]["asset_contributions"][i], delta=2e-12
                    )

    def test_08_shrink_diagonal_and_frozen_estimates(self):
        for scenario in range(3):
            baseline = c.analyze({**c.settings(), "scenario": scenario})
            diagonal = c.analyze(
                {
                    **c.settings(),
                    "scenario": scenario,
                    "shrink": 100,
                    "weights": [-100] * 5,
                }
            )
            self.assertEqual(baseline["model"], diagonal["model"])
            self.assertEqual(diagonal["factor_covariance"][0][1], 0)
            self.assertEqual(diagonal["factor_covariance"][1][0], 0)
            for i in range(2):
                self.assertEqual(
                    baseline["factor_covariance"][i][i],
                    diagonal["factor_covariance"][i][i],
                )
            for i in range(5):
                for j in range(5):
                    if i != j:
                        self.assertEqual(diagonal["residual_covariance"][i][j], 0)

    def test_09_covariance_choice_is_not_causality_or_guarantee(self):
        for scenario in (0, 1):
            full = c.analyze({**c.settings(), "scenario": scenario, "residual": "full"})
            diag = c.analyze({**c.settings(), "scenario": scenario})
            self.assertNotEqual(full["covariance"], diag["covariance"])
        self.assertLess(
            c.analyze({**c.settings(), "scenario": 1})["risk"]["variance"],
            c.analyze({**c.settings(), "scenario": 1, "residual": "full"})["risk"][
                "variance"
            ],
        )
        self.assertGreater(
            c.analyze({**c.settings(), "scenario": 2})["holdout_risk"]["variance"],
            c.analyze({**c.settings(), "scenario": 2})["train_risk"]["variance"],
        )

    def test_10_validation_no_mutation_or_nonfinite_json(self):
        defaults = c.settings()
        before = deepcopy(defaults)
        c.analyze(defaults)
        self.assertEqual(before, defaults)
        self.assertEqual(defaults, c.settings())
        invalid = [
            None,
            [],
            {},
            {**defaults, "extra": 1},
            {**defaults, "scenario": True},
            {**defaults, "scenario": -1},
            {**defaults, "scenario": 3},
            {**defaults, "shrink": 101},
            {**defaults, "shrink": float("nan")},
            {**defaults, "residual": []},
            {**defaults, "weights": [1, 2]},
            {**defaults, "weights": [100, 100, 100, 100, 101]},
            {**defaults, "weights": [True, 0, 0, 0, 0]},
            {**defaults, "weights": [float("inf"), 0, 0, 0, 0]},
        ]
        for cfg in invalid:
            if cfg is None:
                continue  # None intentionally means defaults for the Python API
            with self.assertRaises(ValueError):
                c.analyze(cfg)
        json.dumps(c.analyze(), allow_nan=False)
        for scenario in range(3):
            json.dumps(c.analyze({**defaults, "scenario": scenario}), allow_nan=False)

    def test_11_cli_is_strict_and_matches_api(self):
        command = [sys.executable, "-m", "coveshift"]
        valid = subprocess.run(
            command
            + [
                "--scenario",
                "1",
                "--residual",
                "full",
                "--shrink",
                "37",
                "--weights",
                "90",
                "-70",
                "30",
                "10",
                "-45",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(valid.stdout),
            c.analyze(
                {
                    "scenario": 1,
                    "residual": "full",
                    "shrink": 37,
                    "weights": [90, -70, 30, 10, -45],
                }
            ),
        )
        for args in [
            ["--scenario", "3"],
            ["--weights", "1", "2"],
            ["--shrink", "0.5"],
            ["--residual", "whatever"],
        ]:
            invalid = subprocess.run(command + args, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(invalid.stdout, "")
            self.assertIn("error:", invalid.stderr)

    def test_12_scaling_reversal_and_signed_residuals(self):
        cfg = {
            **c.settings(),
            "scenario": 1,
            "residual": "full",
            "weights": [40, -30, 10, 5, -20],
        }
        r = c.analyze(cfg)
        negative = c.analyze({**cfg, "weights": [-w for w in cfg["weights"]]})
        double = c.analyze({**cfg, "weights": [2 * w for w in cfg["weights"]]})
        self.assertTrue(any(v < 0 for v in r["factor_contributions"]))
        self.assertTrue(any(v < 0 for v in r["residual_contributions"]))
        self.assertEqual(r["risk"], negative["risk"])
        self.assertAlmostEqual(
            double["risk"]["variance"], 4 * r["risk"]["variance"], delta=1e-18
        )
        self.assertAlmostEqual(
            double["risk"]["daily_vol"], 2 * r["risk"]["daily_vol"], delta=1e-15
        )
        for key in ("factor_contributions", "residual_contributions"):
            for a, b, d in zip(r[key], negative[key], double[key]):
                self.assertAlmostEqual(a, b, delta=1e-15)
                self.assertAlmostEqual(2 * a, d, delta=1e-15)


if __name__ == "__main__":
    unittest.main()
