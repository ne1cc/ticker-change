"""AI analyst report — a written take on a stock, via an LLM.

Feeds the dashboard's already-computed numbers (price action, volatility, VaR,
Monte-Carlo odds, valuation multiples, dealer GEX positioning, and the ML
signal) to an LLM with a seasoned-analyst system prompt, and returns a concise
HTML report for the analytics page. Results are cached in the shared SQLite
api_cache table so a page refresh doesn't re-bill the API.

Works with any configured AI provider — Anthropic (Claude), OpenAI, Google
Gemini, or OpenRouter — resolved from the /settings page or environment (see
providers.ai_providers). They're tried in order, so the report falls back to the
next provider if one is missing, errored, or out of quota. Degrades gracefully:
with no key configured (or every provider failing) it returns None and the
report section simply doesn't render.
"""
from __future__ import annotations

import json

import requests

import db
import providers

CACHE_TTL_HOURS = 12
HTTP_TIMEOUT = 90

SYSTEM = """You are a seasoned sell-side equity research analyst writing a concise \
desk note for an experienced investor. You are given a structured snapshot of one \
stock — price action, volatility and tail-risk statistics, a Monte-Carlo path \
forecast, valuation multiples, dealer gamma-exposure (GEX) positioning, and a \
machine-learning signal. Write a balanced, professional, and scientifically rigorous briefing.

Rules:
- Ground every claim in the numbers provided. Do not invent figures, news, \
earnings results, or price targets that aren't in the data. If something isn't \
given, say it's not available rather than guessing.
- Be even-handed: state the bull case and the bear case. Never cheerlead.
- Synthesize the indicators into clear "Buyer Signals" using statistical rigor:
  * For GEX: Assess the market micro-structure. Is the GEX regime positive (vol-dampening, supportive of mean reversion/support at walls) or negative (vol-expanding, trend-amplifying)? Analyze the Call Wall, Put Wall, and Zero-Gamma flip relative to the Spot price to explain where dealer hedging could accelerate or cap price action.
  * For Valuation & Fundamentals: Analyze how multiple metrics (P/E, PEG, EV/EBITDA) compare to growth rates and estimates to see if valuation is supported.
  * For Technical & Volatility: Connect realized volatility, Sharpe, Beta, and the ML signal.
  * For Key Risks: Weigh the 1-day Value-at-Risk (VaR) and Expected Shortfall (ES) at 95%/99% confidence to detail potential tail risk.
  * For Monte Carlo: Detail the path projections and the probability of upward movement over time.
- If an `ml_signal` is present, work it into the Technical & Volatility Picture and \
the Bottom Line: state its action and confidence, name the specific feature drivers \
it cites (e.g. momentum, RSI, distance to moving averages, volatility), and say \
whether they agree or conflict with the other evidence. Weight it lightly — it is a \
weak signal that has not beaten buy-and-hold out of sample.
- Calibrate confidence to the data quality; flag where the inputs are thin.
- Output GitHub-flavoured Markdown with these sections, in order, using `###` \
headings: Snapshot, Valuation & Fundamentals, Technical & Volatility Picture, \
Options & Dealer Positioning, Key Risks & Catalysts, Bottom Line. Keep it tight \
(~400-600 words). Use short paragraphs and bullets where they help.
- End with one italic line: *Educational analysis generated from delayed data — \
not financial advice.*"""


def _context(ticker: str, data: dict) -> str:
    """Compact JSON of the fields worth reasoning over (skip chart HTML)."""
    payload = {
        "ticker": ticker,
        "current_price": data.get("current_price"),
        "statistics": data.get("stats"),
        "forward_estimates": data.get("forward_estimates"),
        "fundamentals": data.get("fundamentals"),
        "price_ranges": data.get("price_ranges"),
        "dealer_gex": data.get("gex"),
        "ml_signal": data.get("ml"),
    }
    payload = {k: v for k, v in payload.items() if v}
    return json.dumps(payload, default=str, indent=2)


def _preprocess_markdown_tables(md: str) -> str:
    import re
    if not md:
        return md
    # 1. Separate table rows compressed with double pipes || or | |
    md = re.sub(r'\|\s*\|', '|\n|', md)
    # 2. Separate table title/caption from the start of table headers on the same line
    md = re.sub(r'(?:\n|^)([^\n|]*?Table\s+[A-Z]:[^\n|]+?)\s*(\|.*)', r'\n\n#### \1\n\n\2', md, flags=re.IGNORECASE)
    # 3. Clean up any duplicated header markers or leading symbols in title lines
    md = re.sub(r'####\s*[\*\#]*\s*(Table\s+[A-Z]:[^\n]+?)\s*[\*\#]*\s*$', r'#### \1', md, flags=re.MULTILINE | re.IGNORECASE)
    return md


def _preprocess_options_markdown(md: str) -> str:
    """Clean, structure, and space AI options market report markdown."""
    import re
    if not md:
        return md

    # 1. Run standard table preprocessor
    md = _preprocess_markdown_tables(md)

    # 2. Fix malformed heading prefixes like "1. ###" -> "### 1."
    md = re.sub(r'(?m)^\s*(\d+)[\.\)]\s*###\s*', r'### \1. ', md)

    # 3. Standardize main section headers (with or without numbers or hashes)
    main_sections = [
        (r'Options Market Regime\s*(?:&|and)\s*Volatility Analysis', '1. Options Market Regime & Volatility Analysis'),
        (r'Expected Move\s*(?:&|and)\s*(?:Key\s*)?Boundary Levels', '2. Expected Move & Boundary Levels'),
        (r'Quantitative Strategy Recommendations', '3. Quantitative Strategy Recommendations'),
        (r'Tail Risk\s*(?:&|and)\s*(?:Risk Management|Defensive Protocols)', '4. Tail Risk & Risk Management'),
    ]
    for pat, canonical in main_sections:
        md = re.sub(
            rf'(?m)^(?:\d+[\.\)]\s*)?(?:###?\s*)?({pat})[:\s]*$',
            rf'\n\n### {canonical}\n\n',
            md,
            flags=re.IGNORECASE
        )

    # 4. Standardize subsection headers
    subsections = [
        r'Expected Move Analysis',
        r'Structural Boundaries',
        r'Defensive Adjustments(?:\s*&?\s*Protocols)?',
        r'Bull Call Debit Spread\s*(?:\([^)]+\))?',
        r'Bear Put (?:Debit )?Spread\s*(?:\([^)]+\))?',
        r'Long Calendar Spread\s*(?:\([^)]+\))?',
        r'Long Diagonal Spread\s*(?:\([^)]+\))?',
        r'Iron Condor\s*(?:\([^)]+\))?',
        r'Strategy\s+\d+[:\s].*',
    ]
    for sub in subsections:
        md = re.sub(
            rf'(?m)^(?:\d+[\.\)]\s*)?(?:####?\s*)?({sub})[:\s]*$',
            r'\n\n#### \1\n\n',
            md,
            flags=re.IGNORECASE
        )

    # 5. Bulletize labeled key-value items if not already bulleted
    labeled_items = [
        r'Volatility Risk Premium\s*(?:\(VRP\))?',
        r'IV Rank\s*/\s*Percentile',
        r'IV Rank',
        r'IV Percentile',
        r'Gamma Flip Point',
        r'Call Wall\s*(?:\(Resistance\))?',
        r'Put Wall\s*(?:\(Support\))?',
        r'Max Pain',
        r'Worst-Case Scenario',
        r'Stop-Loss',
        r'Dynamic Delta Hedging',
        r'Gamma Risk',
        r'Structure',
        r'Rationale',
        r'Target',
        r'PoP',
        r'Trade Management',
        r'Execution & Sizing',
        r'Execution',
        r'Profit Target',
    ]
    for lbl in labeled_items:
        md = re.sub(
            rf'(?m)^(?!\s*[-*]\s*)({lbl}):\s*(.+)$',
            r'\n- **\1:** \2\n',
            md,
            flags=re.IGNORECASE
        )

    # 6. Ensure headings have proper vertical breathing room
    md = re.sub(r'(?m)([^\n])\n(#{2,4}\s+)', r'\1\n\n\2', md)
    md = re.sub(r'(?m)(#{2,4}\s+[^\n]+)\n([^\n#])', r'\1\n\n\2', md)

    # 7. Normalize excessive blank lines (more than 2) to 2
    md = re.sub(r'\n{3,}', '\n\n', md)

    return md.strip()



def _to_html(md: str) -> str:
    try:
        import markdown
        md = _preprocess_markdown_tables(md)
        return markdown.markdown(md, extensions=["extra", "sane_lists"])
    except Exception:
        # No markdown lib installed — render as readable preformatted text.
        from html import escape
        return f'<div class="prose-fallback whitespace-pre-wrap">{escape(md)}</div>'


def parse_html_sections(html: str) -> dict[str, str]:
    """Parse a compiled HTML report into sections based on <h2/3/4> headers."""
    import re
    sections = {}
    
    # Match headers <h2/3/4>...</h2/3/4> and capture content up to the next header or end of string.
    pattern = re.compile(r'<h[234]>(.*?)</h[234]>(.*?)(?=(?:<h[234]>|$))', re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(html)
    
    header_mapping = {
        "snapshot": "snapshot",
        "valuation": "valuation",
        "fundamentals": "valuation",
        "technical": "technical",
        "volatility": "technical",
        "options": "options",
        "dealer": "options",
        "gex": "options",
        "risks": "risks",
        "catalysts": "risks",
        "bottom line": "bottom_line",
        "verdict": "bottom_line"
    }
    
    for header, content in matches:
        clean_header = header.strip().lower()
        content = content.strip()
        
        found_key = None
        for phrase, key in header_mapping.items():
            if phrase in clean_header:
                found_key = key
                break
                
        if found_key:
            sections[found_key] = content
            
    return sections


def _anthropic_report(key: str, model: str, user_msg: str, system_prompt: str = SYSTEM) -> str | None:
    """Generate the report via the Anthropic SDK (native Claude API)."""
    try:
        import anthropic
    except Exception:
        return None
    client = anthropic.Anthropic(api_key=key)
    # Streaming + get_final_message keeps us safe from HTTP timeouts on the
    # longer reports; adaptive thinking lets Claude reason over the numbers.
    with client.messages.stream(
        model=model,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},  # stable prefix → cache across tickers
        }],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()
    if getattr(msg, "stop_reason", None) == "refusal":
        return None
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    return text or None


def _openai_compatible_report(base_url: str, key: str, model: str, user_msg: str, system_prompt: str = SYSTEM) -> str | None:
    """Generate the report via an OpenAI-compatible /chat/completions endpoint.

    Shared by OpenAI, Google Gemini (OpenAI-compat base), and OpenRouter — they
    all speak the same request/response shape, so one path covers all three.
    """
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 4000,
            "temperature": 0.4,
        },
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")
    choices = (resp.json() or {}).get("choices") or []
    if not choices:
        raise Exception(f"No choices returned by model API. Response: {resp.text[:300]}")
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise Exception(f"Model returned empty content choice. Response: {resp.text[:300]}")
    return content.strip()


def generate_report(ticker: str, data: dict) -> str | None:
    """Return an HTML analyst report for `ticker`, or None if unavailable.

    Tries each configured AI provider in order and uses the first that returns a
    report, so a missing/exhausted provider transparently rolls over to the next.
    """
    ticker = ticker.upper()
    configured = providers.ai_providers()
    if not configured:
        return None

    cached = db.cache_get("ai_report", ticker, CACHE_TTL_HOURS)
    if cached is not None:
        return cached.get("html")

    user_msg = (
        f"Write the desk note for {ticker} from this snapshot:\n\n{_context(ticker, data)}"
    )

    text, used = None, None
    for prov in configured:
        try:
            if prov["id"] == "anthropic":
                text = _anthropic_report(prov["key"], prov["model"], user_msg, SYSTEM)
            else:
                text = _openai_compatible_report(
                    prov["base_url"], prov["key"], prov["model"], user_msg, SYSTEM
                )
        except Exception as e:
            print(f"AI report via {prov['id']} failed for {ticker}: {e}")
            text = None
        if text:
            used = prov["id"]
            break

    if not text:
        return None

    html = _to_html(text)
    db.cache_set("ai_report", ticker, {"html": html, "provider": used})
    return html


COMPREHENSIVE_SYSTEM = """You are a world-class quantitative hedge fund strategist and financial research director. You are given a comprehensive, multi-dimensional dataset of a single stock ticker containing daily returns stats, tail risk calculations (VaR, Expected Shortfall), valuation multiples, dealer GEX walls/ regime flip details, insider trading MSP sentiment, top institutional 13F holders, seasonality history, momentum indices, and machine learning model signal/feature weights.

Your task is to write a highly rigorous, comprehensive quantitative analysis and trading strategy report.

Format your response in GitHub-Flavoured Markdown. Use the following structured sections:
1. ### Executive Summary: High-level overview of the name.
2. ### Comprehensive Data Summary (Tables):
   You must compile the input numbers into structured GitHub-Flavored Markdown tables summarizing EVERY single category of data provided.
   CRITICAL TABLE FORMATTING RULE: Each table MUST be formatted as a standard multi-line Markdown table with its own header row, delimiter row (`|---|---|`), and data rows on separate lines. Do NOT combine table cells onto a single line or use double pipes `||`.
   Example standard table format:
   #### Table A: Price, Realized Volatility & Risk Metrics
   | Metric | Value | Metric | Value |
   | --- | --- | --- | --- |
   | Current Price | $308.91 | Annualized Vol | 28.03% |

   Create separate standard Markdown tables for:
   - Table A: Price, Realized Volatility & Risk Metrics (drawdown, Sharpe, Beta, VaR, Expected Shortfall, etc.)
   - Table B: Valuation & Consensus Growth Guidance (PE, PEG, revenue growth, sector/industry, etc.)
   - Table C: Options Market & Dealer GEX Profile (flip point, walls, HVL, net regime, etc.)
   - Table D: Insider Transactions & Institutional Ownership (sentiment, recent purchases, top holders, etc.)
   - Table E: Momentum Performance & Backtest Statistics (CAGR, win-rate, profit factor, Jensen's alpha, etc.)
   - Table F: Machine Learning Signal & Features (action, confidence, drivers, etc.)
3. ### Scientific Analysis of Signal Convergence:
   Critically evaluate how these metrics confirm or contradict each other. For example: does positive/negative GEX dealer positioning support or counter the ML signal? Does the risk-adjusted momentum score align with institutional accumulation? Is the tail risk (VaR/ES) justified by growth estimates?
4. ### Tail Risk & Forward Scenario Analysis:
   Analyze the downside risk and Monte Carlo path probabilities. Discuss tail scenarios.
5. ### Trading Strategy Synthesis & Recommendations:
   Synthesize all the above data points into a clear, actionable trading strategy recommendation. Define:
   - Position sizing recommendation (based on volatility, realized ATR, and tail risk)
   - Entry thresholds and catalyst parameters
   - Exit parameters (stop-loss, profit targets, or options hedging overlay)
   - Execution timeframe (short-term options play, medium-term momentum follow, long-term fundamentals build)
6. ### Concluding Verdict: A single clear-cut operational summary.

Rules:
- Be extremely thorough and precise. Cover EVERY data point provided.
- Do not invent any numbers. If a data point is missing or empty, mark it as 'N/A' in the tables and explain that it is not available.
- Treat every indicator with scientific skepticism, detailing the limits of the models (e.g. backtest overfitting, delayed yfinance data, option model approximations).
"""


def generate_comprehensive_report(ticker: str, data: dict) -> tuple[str | None, str | None]:
    """Return an HTML comprehensive quantitative strategy report for `ticker` and any error message as a tuple (html, error)."""
    ticker = ticker.upper()
    configured = providers.ai_providers()
    if not configured:
        return None, "No AI providers configured. Please go to Settings to add an API key."

    cached = db.cache_get("ai_comprehensive_report", ticker, CACHE_TTL_HOURS)
    if cached is not None:
        return cached.get("html"), None

    user_msg = (
        f"Write the comprehensive quantitative strategy report for {ticker} from this dataset:\n\n"
        f"{json.dumps(data, default=str, indent=2)}"
    )

    errors = []
    text, used = None, None
    for prov in configured:
        try:
            if prov["id"] == "anthropic":
                text = _anthropic_report(prov["key"], prov["model"], user_msg, COMPREHENSIVE_SYSTEM)
            else:
                text = _openai_compatible_report(
                    prov["base_url"], prov["key"], prov["model"], user_msg, COMPREHENSIVE_SYSTEM
                )
            if not text:
                errors.append(f"{prov['label']} ({prov['model']}) returned empty content.")
        except Exception as e:
            err_msg = f"{prov['label']} ({prov['model']}) failed: {str(e)}"
            print(f"AI comprehensive report via {prov['id']} failed for {ticker}: {e}")
            errors.append(err_msg)
        if text:
            used = prov["id"]
            break

    if not text:
        err_report = "All configured AI providers failed to generate the report:\n- " + "\n- ".join(errors)
        return None, err_report

    html = _to_html(text)
    db.cache_set("ai_comprehensive_report", ticker, {"html": html, "provider": used})
    return html, None


OPTIONS_SYSTEM = """You are a premier quantitative derivatives strategist with a measured, reserved analytical style. You are given a detailed dataset of an option chain snapshot for a single stock ticker, including current spot price, days to expiration (DTE), implied volatility (IV), historical realized volatilities (HV30/HV90), IV Rank, expected moves, Put-Call ratios, Max Pain, and dealer Gamma Exposure (GEX) walls.

Your task is to write a highly rigorous, actionable Option Chain Analysis and Strategy Report.

Format your response in GitHub-Flavoured Markdown. Use the following structured sections, in order, using `###` headings:
### 1. Options Market Regime & Volatility Analysis
- Compile the volatility profile into a clean Markdown table:
  | Metric | Value | Comparison / Benchmark | Regime Interpretation |
  | :--- | :---: | :---: | :--- |
  Include rows for ATM Implied Volatility (IV), 30-day Realized Volatility (HV30), 90-day Realized Volatility (HV90), IV Rank, and IV Percentile.
- Provide 2-3 concise, spaced bullet points analyzing the Volatility Risk Premium (VRP) and whether option premium is cheap or rich relative to historical distribution.

### 2. Expected Move & Boundary Levels
- Analyze the Expected Move calculated via Black-Scholes vs Straddle pricing. What does the market imply about potential trading range by expiration?
- Compile the dealer structural boundaries into a clean Markdown table:
  | Key Level | Strike / Price | Dealer Positioning | Market Impact |
  | :--- | :---: | :--- | :--- |
  Include rows for Call Wall, Put Wall, Max Pain, Gamma Flip Point, and Current Spot.
- Explain dealer hedging dynamics around the Gamma Flip and walls (positive gamma dampening vs negative gamma trend acceleration).

### 3. Quantitative Strategy Recommendations
- State the optimal volatility posture (long gamma/vega vs premium selling).
- For each recommended strategy (2 setups), use a distinct `#### Strategy [N]: [Strategy Name]` sub-heading, followed by spaced bullet points:
  * - **Structure:** exact strikes, expirations, and leg types.
  * - **Rationale:** why this structure matches the current volatility regime and dealer boundaries.
  * - **Execution & Greeks:** target long-leg delta (e.g. 40-45 delta), capital efficiency, and theta/vega exposure.
  * - **Trade Management:** profit target (e.g. 50% max profit) and defined invalidation/exit levels.

### 4. Tail Risk & Risk Management
- Format with clear sub-headings and bullet points:
  * - **Worst-Case Scenario:** market breakdown or melt-up implications if dealer walls fail.
  #### Defensive Adjustments
  * - **Stop-Loss Protocol:** spot price or technical trigger to close the position.
  * - **Dynamic Delta Hedging:** adjustments if spot moves significantly from target strikes.
  * - **Gamma & Expiration Risk:** managing short gamma into final trading hours.

CRITICAL FORMATTING RULES:
- Always leave a blank line before and after EVERY heading, table, paragraph, list, and bullet item.
- Use GitHub-Flavored Markdown bullet points (`- `) with bold labels (`- **Label:** Details`).
- Never combine separate items or sections into single run-on paragraphs.
- Be highly precise and quantitative. Use the exact data points provided.
- Do not invent any numbers. Mark missing metrics as 'N/A'.
- Keep your tone analytical, professional, and strategic.
"""


def generate_options_report(ticker: str, data: dict, force: bool = False) -> tuple[str | None, str | None]:
    """Return an HTML options analysis report for `ticker` and any error message as a tuple (html, error)."""
    ticker = ticker.upper()
    configured = providers.ai_providers()
    if not configured:
        return None, "No AI providers configured. Please go to Settings to add an API key."

    exp_date = data.get("selected_expiration", "default")
    cache_key = f"options_report:{ticker}:{exp_date}"
    if not force:
        cached = db.cache_get("ai_options_report", cache_key, CACHE_TTL_HOURS)
        if cached is not None:
            if isinstance(cached, dict):
                if cached.get("markdown"):
                    html = _to_html(_preprocess_options_markdown(cached["markdown"]))
                    return html, None
                elif cached.get("html"):
                    return cached.get("html"), None

    user_msg = (
        f"Write the options strategy analysis report for {ticker} from this option chain dataset:\n\n"
        f"{json.dumps(data, default=str, indent=2)}"
    )

    errors = []
    text, used = None, None
    for prov in configured:
        try:
            if prov["id"] == "anthropic":
                text = _anthropic_report(prov["key"], prov["model"], user_msg, OPTIONS_SYSTEM)
            else:
                text = _openai_compatible_report(
                    prov["base_url"], prov["key"], prov["model"], user_msg, OPTIONS_SYSTEM
                )
            if not text:
                errors.append(f"{prov['label']} ({prov['model']}) returned empty content.")
        except Exception as e:
            err_msg = f"{prov['label']} ({prov['model']}) failed: {str(e)}"
            print(f"AI options report via {prov['id']} failed for {ticker}: {e}")
            errors.append(err_msg)
        if text:
            used = prov["id"]
            break

    if not text:
        err_report = "All configured AI providers failed to generate the report:\n- " + "\n- ".join(errors)
        return None, err_report

    cleaned_md = _preprocess_options_markdown(text)
    html = _to_html(cleaned_md)
    db.cache_set("ai_options_report", cache_key, {"html": html, "markdown": text, "provider": used})
    return html, None
