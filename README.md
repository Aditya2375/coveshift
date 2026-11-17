# Coveshift

**Same holdings. Different risk.** A small, fitted factor-covariance lab with signed risk attribution and its assumptions on the page.

[Open the lab](https://aditya2375.github.io/coveshift/) · [Source](https://github.com/Aditya2375/coveshift)

Coveshift generates five fictional asset returns and two fictional factor returns. It fits an intercept and two loadings for each asset on the first 120 daily observations, then freezes those estimates. The next 60 observations are held out. There are no market prices, personal portfolios, accounts, uploads, recommendations or trades.

The question is not which allocation wins. It is how a covariance assumption changes the risk assigned to the same linear book.

## Try it

Choose a data world, full or diagonal residual covariance, and a factor-correlation shrink percentage. Enter signed integer holdings from -100% to 100% per asset, then apply. Holdings are not normalized. Net need not be 100%; gross can reach 500%. This is a reference-capital linear book, not a financing, cash, leverage or short-sale simulator.

- **Stable control:** the generating loadings and shock scales stay fixed. Finite earlier and later samples still disagree.
- **Correlated residuals:** an omitted common shock introduces residual co-movement. Keeping only the residual diagonal can miss it.
- **Later volatility jump:** both factor shock scales double after the fit period. Earlier estimates do not see the change.

The lab shows earlier-sample, selected-model and later-realized volatility; signed asset, factor and asset-residual Euler contributions; all selected covariance matrices; fitted coefficients; a daily book-return chart; and all 180 raw observations. Covariance table entries are daily decimal return squared, multiplied by 10^6 for display.

Edited inputs are marked unapplied. Invalid inputs leave the previous results in place and marked stale. Reset restores the original experiment. Narrow screens scroll tables locally; chart labels are drawn at the actual display width.

## What is fitted

For each of the five assets, ordinary least squares with an intercept fits:

```text
r[t,i] = alpha[i] + B[i,:] f[t] + residual[t,i]
```

Only observations 1 through 120 enter the fit. A centered 2-by-2 solve computes the loadings. The intercept is the asset mean minus its fitted loadings times the factor mean. Broad and Tilt are synthetic factor names, not observed economic factors, and the generating loadings are not passed into the estimator.

Let F be the centered sample factor covariance and E the centered sample covariance of the fitted residuals. Both use denominator 119. This convention preserves the covariance accounting identity; E is not the unbiased regression error-variance estimate RSS/(n-3).

The selected covariance is:

```text
F_selected[i,j] = F[i,j]                        if i == j
                  F[i,j] * (1 - shrink/100)     otherwise
Psi[i,j] = E[i,j]                              in full mode
           E[i,i] on the diagonal; 0 elsewhere in diagonal mode
Sigma = B F_selected B^T + Psi
```

The two-factor shrink control moves off-diagonal covariance toward zero while retaining factor variances. This also shrinks factor correlation. It is a manual assumption, not a fitted shrinkage estimator, not Ledoit-Wolf, and not a parameter chosen using the later sample. Shrink and residual mode do not refit loadings or residuals.

With full E and zero shrink, the selected covariance equals the earlier sample asset covariance up to floating-point roundoff. OLS residuals are orthogonal to the fitted factors, so the sample cross terms vanish. The lab reports the maximum entrywise reconstruction error and factor-residual covariance. A diagonal residual matrix or nonzero factor shrink intentionally breaks that identity. Neither change is claimed to improve a forecast.

## Risk accounting

For decimal holdings w and book factor exposure x = B^T w:

```text
variance       = w^T Sigma w = x^T F_selected x + w^T Psi w
volatility     = sqrt(variance)
asset Euler[i] = w[i] * (Sigma w)[i] / volatility
factor Euler[k]= x[k] * (F_selected x)[k] / volatility
resid Euler[i] = w[i] * (Psi w)[i] / volatility
```

Asset contributions sum to total volatility. Separately, factor contributions plus asset-residual contributions sum to the same total volatility. They are two views of the same risk, not quantities to add together. Contributions can be negative, including residual contributions in full mode. A negative contribution offsets modeled risk in the selected book; it is not a claim of safety or a causal effect.

Variances add across the factor and residual blocks. Standalone block volatilities do not add. The two displayed variance shares use the model's total variance denominator. At zero risk, contributions and shares are undefined and shown as such, rather than invented zero percentages. JSON uses `null` for undefined Euler vectors.

The page annualizes standard deviation and contributions with sqrt(252). It assumes uncorrelated daily returns and 252 hypothetical trading days per year. Later realized covariance uses only the later 60 observations with denominator 59. This single synthetic later comparison is not validation of forecast accuracy. No expected return, probability of loss, tail risk, confidence interval or optimal allocation is supplied.

## Reproduce locally

Python 3.10 or newer. Runtime uses only the standard library.

```sh
python3 -m coveshift
python3 -m coveshift --scenario 1 --residual full --shrink 37 --weights 90 -70 30 10 -45
make test
```

Each CLI run emits a complete finite JSON result. Malformed or out-of-range parameters fail with an argparse error and no JSON. The import API is:

```python
from coveshift import analyze, settings

cfg = settings()
cfg["scenario"] = 1
cfg["residual"] = "full"
report = analyze(cfg)
print(report["risk"]["annual_vol"])
```

`analyze()` or `analyze(None)` uses defaults in Python. Otherwise its input must be a plain built-in dictionary containing exactly `scenario`, `residual`, `shrink`, and `weights`. Scenario is integer 0..2; shrink is integer 0..100; residual is `full` or `diagonal`; weights is a built-in list of five integers from -100 to 100. Booleans, float-valued integers, strings, extra keys and nonfinite numbers are rejected. JavaScript `analyze()` uses defaults; explicit `null` is invalid there. JavaScript inputs must be plain objects with the same keys, dense arrays and safe integers. Caller-supplied proxies, getters, subclasses and concurrent mutation are outside the API contract. Internal numerical helpers assume trusted finite rectangular data from this fixed generator and are not arbitrary-matrix APIs.

Serve the browser version:

```sh
python3 -m http.server 8366 --directory docs --bind 127.0.0.1
# Open http://127.0.0.1:8366/
```

JavaScript has no runtime packages. Development checks need Node 22 and the locked browser-test package:

```sh
npm ci --ignore-scripts
npm test
npx playwright install chromium
npm run test:browser
```

Tests include an independent Gram-Schmidt QR regression oracle, a pairwise-difference covariance oracle, an analytic signed-Euler fixture, finite-difference Euler checks, holdout contamination checks, covariance reconstruction, 360 signed-weight accounting cases, strict-input and CLI checks, and 72 full-result Python/JavaScript comparisons. Browser checks exercise five widths, every data world and residual mode, both shrink endpoints, zero holdings, stale and invalid inputs, all matrix cells and Euler labels, raw observations, page containment and failed or third-party requests. Chromium is exercised; other browsers and assistive technologies are not certified.

## Synthetic data and scope

The fixed 32-bit LCG starts at 9173, uses multiplier 1664525 and increment 1013904223 modulo 2^32, and consumes eight draws per day. Each draw is mapped uniformly into [-1,1). No Gaussian, independence, return-distribution or stationarity claim is made about this teaching generator. Core sources define every scale and loading. The same seed and draw order are used in all worlds; the later-jump world's earlier data are byte-for-byte equal to the stable world's earlier data.

Five assets, two factors, 180 observations, bounded weights and finite integer controls keep each calculation small. There is no arbitrary CSV loader, date alignment, missing-data logic, rolling estimation or user-selected sample. Floating point is used throughout; the estimator is not an exact-arithmetic or numerically general regression library. Synthetic factors are well-conditioned enough for the documented direct solve. Tiny negative variance caused solely by roundoff is clamped at zero; materially negative or nonfinite quadratic forms fail.

The two-factor structure is smaller than the five-asset universe, but this tiny fitted example is not evidence of parameter savings or better estimation. Fitted loadings themselves consume parameters. No claim of independence, causality, real-world factor discovery, machine-learning forecasting or risk reduction is made.

## Privacy and security

The static app runs in the browser after same-origin assets load. It has no runtime fetches, analytics, cookies, local storage, service worker, persistence, file upload, remote model, API key, credential field or execution endpoint. External reading and source links navigate away only when clicked. Ordinary hosting access logs may exist. Browser session restore can retain form state; reset is not a secure-erasure feature.

The document includes a restrictive meta content-security policy with runtime connections, objects, base changes and form actions disabled. Hosting response headers, framing controls and cache policy are controlled by the host and are not promised by the page's meta policy. Python runtime performs bounded local arithmetic; test harnesses start a loopback server and invoke Python as documented. These checks are engineering safeguards, not a security certification.

## Mathematical references

Conceptual references only. No external market data, assets, model weights or implementation code are bundled.

- [MOSEK Portfolio Optimization Cookbook: factor models](https://docs.mosek.com/portfolio-cookbook/factormodels.html): covariance assumptions and the factor/residual variance split.
- [William Sharpe, Stanford: factor-based expected returns, risks and correlations](https://web.stanford.edu/~wfsharpe/mia/fac/mia_fac3.htm): covariance identities. Coveshift does not use an expected-return allocation model.
- [qis documentation: factor risk models](https://quantinveststrats.readthedocs.io/en/stable/factor_risk_models.html): signed Euler contributions and the distinction between variance, volatility and contributions.
- [NumPy covariance reference](https://numpy.org/doc/stable/reference/generated/numpy.cov.html): sample-covariance orientation and N-1 convention. NumPy is not a runtime dependency.
- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html): OLS and intercept terminology. scikit-learn is not used.
- [StatLect: linear regression model](https://www.statlect.com/fundamentals-of-statistics/linear-regression): regression assumptions, coefficients and fitted residuals.

Built by Aditya Kulkarni. MIT licensed. Education only, not financial advice.
