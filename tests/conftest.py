"""Shared test environment -- imported by pytest before any test module.

Keeps app.py's import-time background warmers offline: the S&P heatmap
warmer would otherwise start a real ~500-ticker yfinance download the
moment any test imports app. (The options warmer is already benign because
it no-ops on an empty database; the heatmap warmer's whole job is filling
an empty one.)
"""
import os

os.environ["HEATMAP_WARM_ON_BOOT"] = "0"
