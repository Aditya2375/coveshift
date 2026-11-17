import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import assert from "node:assert/strict";
import { analyze, settings, ASSETS } from "../docs/core.js";
const base = process.env.BASE_URL || "http://127.0.0.1:8366/";
let server;
if (!process.env.BASE_URL) {
  server = spawn(
    "python3",
    ["-m", "http.server", "8366", "--directory", "docs", "--bind", "127.0.0.1"],
    { stdio: "ignore" },
  );
  for (let i = 0; i < 50; i++) {
    try {
      if ((await fetch(base)).ok) break;
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
}
const browser = await chromium.launch({ headless: true });
let assertions = 0;
const check = (v, m) => {
    assertions++;
    assert(v, m);
  },
  pct = (x) => (x * 100).toFixed(3) + "%";
await mkdir("/tmp/coveshift-shots", { recursive: true });
try {
  for (const width of [1440, 1024, 768, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } }),
      errors = [],
      failed = [],
      third = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("requestfailed", (r) => failed.push(r.url()));
    page.on("request", (r) => {
      if (!r.url().startsWith(base) && !r.url().startsWith("data:"))
        third.push(r.url());
    });
    await page.goto(base);
    await page.locator("#status").filter({ hasText: "APPLIED" }).waitFor();
    async function apply(cfg) {
      await page.locator("#scenario").selectOption(String(cfg.scenario));
      await page.locator("#residual").selectOption(cfg.residual);
      await page.locator("#shrink").fill(String(cfg.shrink));
      for (let i = 0; i < 5; i++)
        await page.locator("#w" + i).fill(String(cfg.weights[i]));
      await page.locator("#apply").click();
      await page.waitForFunction(
        () => !document.body.classList.contains("stale"),
      );
    }
    async function verify(cfg) {
      const r = analyze(cfg);
      for (const [id, v] of [
        ["model-vol", r.risk.annual_vol],
        ["train-vol", r.train_risk.annual_vol],
        ["holdout-vol", r.holdout_risk.annual_vol],
      ])
        check(
          (await page.locator("#" + id).textContent()) === pct(v),
          width + ": " + id,
        );
      check(
        (await page.locator("#scenario-name").textContent()) === r.name,
        "scenario label",
      );
      for (const [id, mat] of [
        ["covariance", r.covariance],
        ["sample", r.model.sample_covariance],
        ["residual-matrix", r.residual_covariance],
        ["factors", r.factor_covariance],
      ]) {
        const values = await page
          .locator("#" + id + " tbody tr")
          .evaluateAll((rs) =>
            rs.map((row) =>
              [...row.querySelectorAll("td")].map((td) => td.textContent),
            ),
          );
        check(
          JSON.stringify(values) ===
            JSON.stringify(
              mat.map((row) => row.map((v) => (v * 1e6).toFixed(3))),
            ),
          width + ": " + id + " every cell",
        );
      }
      check(
        (await page.locator("#data tr").count()) === 180,
        "all observations",
      );
      const allRows = await page
        .locator("#data tr")
        .evaluateAll((rows) =>
          rows.map((row) =>
            [...row.querySelectorAll("td")].map((td) => td.textContent),
          ),
        );
      check(
        JSON.stringify(allRows) ===
          JSON.stringify(
            r.model.data.map((row, i) => [
              String(row.day),
              row.partition,
              ...row.factors.map((v) => v.toFixed(7)),
              ...row.returns.map((v) => v.toFixed(7)),
              r.daily[i].return.toFixed(7),
            ]),
          ),
        "all 180 audit rows exact",
      );
      const loadingRows = await page
        .locator("#loadings tr")
        .evaluateAll((rows) =>
          rows.map((row) =>
            [...row.querySelectorAll("td")].map((td) => td.textContent),
          ),
        );
      check(
        JSON.stringify(loadingRows) ===
          JSON.stringify(
            ASSETS.map((_, i) => [
              r.model.intercept[i].toFixed(7),
              ...r.model.beta[i].map((v) => v.toFixed(5)),
              r.risk.asset_contributions
                ? pct(r.risk.asset_contributions[i] * Math.sqrt(252))
                : "Undefined",
              r.residual_contributions
                ? pct(r.residual_contributions[i] * Math.sqrt(252))
                : "Undefined",
            ]),
          ),
        "all fitted coefficients and per-asset residuals exact",
      );

      if (r.risk.asset_contributions) {
        const labels = await page
          .locator("#asset-bars .bar-number")
          .allTextContents();
        check(
          JSON.stringify(labels) ===
            JSON.stringify(
              r.risk.asset_contributions.map((v) => pct(v * Math.sqrt(252))),
            ),
          "all asset Euler labels",
        );
        const factors = await page
          .locator("#factor-bars .bar-number")
          .allTextContents();
        check(
          JSON.stringify(factors) ===
            JSON.stringify(
              [...r.factor_contributions, ...r.residual_contributions].map(
                (v) => pct(v * Math.sqrt(252)),
              ),
            ),
          "all factor/residual labels",
        );
      } else {
        check(
          (await page.locator("#asset-bars").textContent()).includes(
            "Undefined",
          ),
          "zero bars",
        );
        check(
          (await page.locator("#factor-share").textContent()) === "Undefined",
          "zero share",
        );
      }
      check(
        (await page.locator("#returns-chart path").getAttribute("d")) !== null,
        "chart present",
      );
      check(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        "no page overflow",
      );
    }
    await verify(settings());
    if (width === 1440 || width === 320) {
      await page.screenshot({ path: `/tmp/coveshift-shots/${width}-hero.png` });
      await page.locator(".panels").scrollIntoViewIfNeeded();
      await page.screenshot({ path: `/tmp/coveshift-shots/${width}-risk.png` });
      await page.locator("#matrices").scrollIntoViewIfNeeded();
      await page.screenshot({
        path: `/tmp/coveshift-shots/${width}-matrix.png`,
      });
      await page.locator("#observations").scrollIntoViewIfNeeded();
      await page.screenshot({
        path: `/tmp/coveshift-shots/${width}-chart.png`,
      });
    }
    for (const scenario of [0, 1, 2])
      for (const residual of ["diagonal", "full"])
        for (const shrink of [0, 100]) {
          const cfg = {
            scenario,
            residual,
            shrink,
            weights:
              scenario === 1 ? [90, -70, 30, 10, -45] : [25, 25, 20, 15, 15],
          };
          await apply(cfg);
          await verify(cfg);
          if (
            (width === 1440 || width === 320) &&
            scenario === 1 &&
            residual === "full" &&
            shrink === 0
          ) {
            await page.locator(".panels").scrollIntoViewIfNeeded();
            await page.screenshot({
              path: `/tmp/coveshift-shots/${width}-negative-residuals.png`,
            });
          }
        }
    const zero = { ...settings(), weights: [0, 0, 0, 0, 0] };
    await apply(zero);
    await verify(zero);
    if (width === 320) {
      await page.locator(".panels").scrollIntoViewIfNeeded();
      await page.screenshot({ path: "/tmp/coveshift-shots/320-zero-risk.png" });
      await page.locator("#observations").scrollIntoViewIfNeeded();
      await page.screenshot({
        path: "/tmp/coveshift-shots/320-zero-chart.png",
      });
    }
    await page.locator("#shrink").fill("101");
    check(
      (await page.locator("#status").textContent()) === "UNAPPLIED",
      "stale",
    );
    await page.locator("#apply").click();
    check(
      (await page.locator("#status").textContent()) === "CHECK INPUT",
      "invalid state",
    );
    check(
      (await page.locator("#error").textContent()).includes("integer"),
      "error visible",
    );
    check(
      (await page.locator("#model-vol").textContent()) === "0.000%",
      "invalid preserves prior result",
    );
    await page.locator("#reset").click();
    await verify(settings());
    await page.locator("#w0").fill("1.5");
    await page.locator("#apply").click();
    check(
      (await page.locator("#error").textContent()).includes("whole numbers"),
      "fraction rejected",
    );
    await page.locator("#reset").click();
    await page.locator("summary").click();
    check(
      await page.locator("#data tr").first().isVisible(),
      "audit disclosed",
    );
    check(
      (await page.locator("#data tr").last().count()) === 1,
      "last row exists",
    );
    const cells = await page
        .locator("#data tr")
        .first()
        .locator("td")
        .allTextContents(),
      first = analyze().model.data[0];
    check(
      JSON.stringify(cells) ===
        JSON.stringify([
          "1",
          "train",
          ...first.factors.map((v) => v.toFixed(7)),
          ...first.returns.map((v) => v.toFixed(7)),
          analyze().daily[0].return.toFixed(7),
        ]),
      "first full audit row",
    );
    check(errors.length === 0, "page errors " + errors.join(","));
    check(failed.length === 0, "failed requests " + failed.join(","));
    check(third.length === 0, "third-party requests");
    check(
      await page.evaluate(
        () => localStorage.length === 0 && sessionStorage.length === 0,
      ),
      "no local storage",
    );
    check((await page.context().cookies()).length === 0, "no cookies");
    check(
      await page.evaluate(
        async () =>
          !("serviceWorker" in navigator) ||
          (await navigator.serviceWorker.getRegistrations()).length === 0,
      ),
      "no service workers",
    );
    check(
      (await page.locator("html").getAttribute("lang")) === "en",
      "language",
    );
    check((await page.locator("h1").count()) === 1, "one h1");
    for (const id of [
      "scenario",
      "residual",
      "shrink",
      ...ASSETS.map((_, i) => "w" + i),
    ])
      check(
        await page.locator("#" + id).evaluate((e) => e.labels?.length > 0),
        "associated label " + id,
      );
    await page.close();
  }
  console.log(
    `${assertions} browser assertions across 5 widths; scenarios/modes/shrink/zero/input guards/all matrices/Euler labels/no network PASS`,
  );
} finally {
  await browser.close();
  if (server) server.kill();
}
