# Multi-leg options payoff simulator

Status: design approved in conversation on 2026-10-19; implementation awaits plan approval.

## Purpose and reference

Upgrade the existing `/options?tab=chain` payoff simulator into a multi-leg strategy builder with a price-by-date P&L heatgrid, following the interaction demonstrated at https://optionstrat.com/build/long-call/SPY/.SPY261120C757 and the user's screenshots. Adapt the interaction to the application's light, dark, and Excel themes.

The user approved: editable option legs, common strategy presets, a heatgrid with dates across columns and underlying prices down rows, adjustable range and IV, dollar P&L/contract value/percent-of-max-risk metrics, combined risk statistics, a graph toggle, and full-width placement below the Greeks visualizer. All legs share the selected expiration in this release. Pre-expiration values use Black–Scholes with an adjustable multiplier applied to each leg's IV; expiration values use exact intrinsic payoff.

## Scope and constraints

- Use the existing Flask/Jinja page, plain JavaScript, and TerminalCharts/ECharts; no new dependencies or frontend build step.
- Use the existing `/api/options-greeks/<ticker>` endpoint; no new endpoint or schema change.
- Options only, one ticker, one shared expiration, standard 100-share contracts.
- Preserve the other options tabs and the Greeks visualizer's behavior.
- Do not fabricate quotes or IV, silently substitute another strike/option side, or overwrite entered premiums during polling.
- The scenario values are conditional modeled P&L, not probability-weighted expected returns.
- Calculations exclude dividends, early exercise/assignment, commissions, and slippage; state the modeling assumptions near the controls.
- No live-flow feed, optimizer, persistence service, stock legs, calendar/diagonal spreads, chance-of-profit estimate, or historical performance tracking.

## Architecture

`static/options-payoff.js` owns pure option valuation, multi-leg aggregation, expiry risk analysis, scenario sampling, and metric conversion. It exports `window.OptionsPayoff` for the page and CommonJS exports for Node's built-in test runner.

`static/options-payoff-simulator.js` owns leg state, quote/IV selection, presets, validated editing, table rendering, chart construction, view controls, and chain lifecycle handling. It exports `window.OptionsPayoffSimulator.create(config)` and accepts the existing chart/reference helpers as injected dependencies, keeping the model independent of the DOM.

`templates/options.html` supplies accessible controls and lightweight lifecycle adapters. The existing `initPayoffSimulator`, `drawPayoffDiagram`, and `selectOptionForSimulator` entry points delegate to the controller. Chain polling updates source context and model inputs without resetting user entry prices. `static/options-payoff.css` scopes layout and heatgrid styling. `glossary.py` updates the payoff definition to match the combined payoff and modeled scenario display.

## Leg editing and presets

Each leg has a stable ID, buy/sell side, call/put type, strike, premium per share, positive integer contract quantity, and IV percentage. Premium and IV track their source (quote midpoint, positive last, custom, or unavailable). The quantity multiplier is always 100; the UI shows both units.

Initial load creates a single long call at the loaded strike nearest spot. Add Call/Add Put creates a leg using that same selection rule. Chain clicks populate the explicitly selected leg; when no leg is selected, they append a leg. Visibly identify the selected leg so the action is predictable. A standalone Add action and each row's Remove action support custom combinations without a leg-count restriction.

Changing a leg's strike or type looks up only the exact loaded strike and that side's quote/IV. Use a finite nonnegative bid/ask midpoint only when the existing `<side>_quote_valid` flag is true; otherwise use a finite strictly positive last price, clearly labeled as Last. A missing quote stays blank until entered. A user-entered zero premium is valid. A chain IV must be finite and positive; missing IV remains blank. A manually entered zero IV is valid and uses the deterministic zero-volatility limit.

Presets: long call, long put, bull call debit spread, bear put debit spread, bull put credit spread, bear call credit spread, long straddle, long strangle, long call butterfly, and short iron condor. Presets replace the existing legs and choose actual loaded strikes around spot. Butterfly uses an equally spaced triple and a 1:-2:1 ratio. Disable a preset with an explanatory message when the loaded strike set cannot construct it; never invent strikes or premiums.

On a same-expiration refresh preserve leg identity, side/type/strike/quantity, entry premiums, selected leg, scenario controls, and manually overridden IV. Auto IV and provenance update from the matching fresh observation. Changing expiration retains the structure but deliberately resets premiums/IV to the new expiration's matching quotes, clears custom quote/IV overrides, and announces the change. Missing strikes remain editable with missing-data labels. Failed or stale requests cannot relabel the strategy as the requested expiration before data arrives. An empty response cannot retain stale calculated results without labeling the state.

## Exact expiry calculations

For direction `d` (+1 buy, -1 sell), premium `p`, contracts `n`, strike `K`, and scenario price `S`, leg P&L is `d * (intrinsic(S, K) - p) * n * 100`. Total P&L is the sum of every valid leg. Initial debit is `sum(d * p * n * 100)`; negative debit is displayed as a net credit.

Evaluate the combined payoff at zero and every distinct positive strike. Between those knots, payoff is linear. Solve each sign-changing interval for its exact root; collect zero-valued knots, merge contiguous zero intervals, and include a zero tail as a breakeven range. Inspect the total call slope above the highest strike: positive slope means unlimited maximum profit, negative slope means unlimited maximum loss, and zero slope means bounded extrema determined by the knots. Do not infer global risk from the visible chart/grid range. Preserve all breakevens, including multiple roots and identically flat strategies.

Keep numeric results unrounded until presentation. Inputs require a finite strike greater than zero, finite premium at least zero, and finite positive integer contracts. Empty, fractional, negative, and nonfinite inputs cannot silently default to a valid position.

## Pre-expiration model and timing

Use the standard European Black–Scholes call/put value with the page's risk-free rate (including an explicit zero rate), no dividend yield, and per-leg volatility `ivPct / 100 * ivMultiplier`. Use an accurate normal CDF approximation and explicit limits at time zero, spot zero, and volatility zero. At time zero return exact intrinsic value. For nonzero time and zero volatility, use discounted deterministic exercise value.

Anchor the expiry deadline using the response's `time_to_expiration_years` measured from receipt time, which already accounts for 16:00 America/New_York and remaining intraday time. Do not replace a valid zero with a 30-day fallback. Missing timing metadata leaves dated projections unavailable while the exact expiration result remains usable. Column date labels use America/New_York; elapsed times remain numeric timestamps. Sample Now through the expiry deadline with at most 15 distinct columns, always including the exact expiration. An already expired chain has only the expiration column. Same-day options retain intraday Now/expiry projections when time remains.

Missing premium or invalid position inputs invalidate the combined outputs, with leg-specific correction messages. Missing IV affects pre-expiration cells and the modeled graph only; exact expiration results remain available. Never sum a subset of legs as if it were the whole strategy.

## Heatgrid and graph

Place the simulator below the Greeks card at full available width. Keep a compact leg editor and summary strip above a Table/Graph switch. The table is the default view.

Rows descend by underlying price; show dollar price and percent move from spot in sticky row headers. Use 21 evenly spaced rows over a configurable symmetric percent range, with the current spot included exactly. The default range derives from a one-standard-deviation move when IV/time is available, clamped to 2–50%; otherwise use 10%. The range control spans 1–100%. The IV multiplier spans 0–3x and resets to 1x; the leg editor retains the original per-leg IV values.

Columns run from Now to Expiration. Color each cell using its unrounded dollar P&L, with a stable symmetric scale across the displayed matrix, green for positive, red for negative, and neutral for zero/missing. Use readable text colors in light/dark/Excel themes. Each cell has its numeric value plus a focusable descriptive tooltip containing date/time, underlying price/move, dollar P&L, and leg contribution details. Add a visible zero/positive/negative legend. Sticky headers and local horizontal scrolling keep the page itself within the viewport on mobile.

Metric modes are P&L dollars, P&L as percent of absolute net opening debit/credit, signed total contract value (sum of directional modeled option values times contracts and 100), and P&L as percent of maximum risk. Percent-of-premium is unavailable for a zero net opening cash amount. Percent-of-max-risk is available only when maximum expiry loss is finite and greater than zero; otherwise display N/A with the reason. Do not invent a denominator for naked shorts or zero-risk positions.

Graph view displays total expiry P&L and total modeled P&L at a selected scenario date. Reuse TerminalCharts, retain Reset/Export, and show the zero baseline, spot, strikes, and every finite nonnegative breakeven that lies in the visible price domain. Use exact knots/root prices in the expiry series so corners and breakevens remain exact. Expand the chart domain to include all finite breakevens even when outside the table range, and describe that difference. Switching back from Table to Graph triggers chart resize. No shared chart-library refactor is necessary.

## Verification and completion

Use independent, hand-calculated risk fixtures: single call/put long and short; bull call spread; straddle with two roots; iron condor; butterfly quantities; offsetting legs; ratio spreads with an unbounded tail; negative/unreachable roots; and zero-payoff intervals. Verify Black–Scholes against the published standard values C=10.45058357 and P=5.57352602 for S=K=100, T=1, volatility=.2, rate=.05, and verify put-call parity and boundary limits.

Controller tests cover chain selection, preset strike validity, invalid/missing data, quote provenance, entry-price preservation, expiration switching, stale requests, same-day timing, zero risk-free rate, unavailable max-risk percentages, and table/graph consistency. Preserve existing Greeks/chart regression coverage and make the new Node suites reachable from pytest/CI. Verify the rendered route with mocked data and use the collaborative browser for desktop/mobile and light/dark/Excel inspection.

Record actual verification results and file changes in `.scratch/options-multi-leg/walkthrough.md`. Source implementation begins only after approval of `.scratch/options-multi-leg/implementation_plan.md`.
