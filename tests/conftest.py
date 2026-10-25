"""Shared test environment -- imported by pytest before any test module.

Keeps app.py's import-time background warmers offline: the S&P heatmap
warmer would otherwise start a real ~500-ticker yfinance download the
moment any test imports app. (The options warmer is already benign because
it no-ops on an empty database; the heatmap warmer's whole job is filling
an empty one.)

Also pops host AI provider keys so no test's page render can reach a real
LLM: AI-backed views must be exercised through stubs only.
"""
import os

os.environ["HEATMAP_WARM_ON_BOOT"] = "0"
os.environ["RADAR_WARM_ON_BOOT"] = "0"
os.environ["BRIEF_WARM_ON_BOOT"] = "0"
os.environ["FACTORS_WARM_ON_BOOT"] = "0"

for _ai_key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
    os.environ.pop(_ai_key, None)
