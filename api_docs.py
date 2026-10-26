"""OpenAPI 3.0 Specification Generator and Interactive Swagger UI.

Generates complete OpenAPI 3.0.3 compliant schemas for all backend endpoints
including institutional quantitative suite, corporate actions, options greeks,
analytics, positioning, chart data, and SEC filings.
"""
from typing import Any, Dict


def get_openapi_spec() -> Dict[str, Any]:
    """Return the complete OpenAPI 3.0.3 dictionary schema."""
    listing_schema = {
        "type": "object", "nullable": True,
        "properties": {k: {"type": "string"} for k in ("id", "symbol", "exchange", "country", "name")},
    }
    feed_schema = {
        "type": "object",
        "required": ["ticker", "status", "listing", "candidates", "articles", "fetched_at", "stale", "partial", "message", "limit"],
        "properties": {
            "ticker": {"type": "string", "example": "AAPL"},
            "status": {"type": "string", "enum": ["ok", "empty", "needs_selection", "unsupported", "unconfigured", "quota_exhausted", "unavailable"]},
            "listing": listing_schema,
            "candidates": {"type": "array", "items": listing_schema},
            "articles": {"type": "array", "maxItems": 500, "items": {
                "type": "object", "required": ["id", "title", "url", "source", "published_at", "language", "entities"],
                "properties": {
                    "id": {"type": "string"}, "title": {"type": "string"},
                    "url": {"type": "string", "format": "uri"}, "source": {"type": "string"},
                    "published_at": {"type": "string", "format": "date-time"},
                    "language": {"type": "string", "enum": ["en"]},
                    "entities": {"type": "array", "items": {"type": "object", "properties": {
                        k: {"type": "string"} for k in ("symbol", "exchange", "country")
                    }}},
                },
            }},
            "fetched_at": {"type": "string", "format": "date-time", "nullable": True},
            "stale": {"type": "boolean"}, "partial": {"type": "boolean"},
            "message": {"type": "string"}, "limit": {"type": "integer", "enum": [500]},
        },
    }
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Ticker-change Quantitative Engine API",
            "version": "1.0.0",
            "description": (
                "Production-grade Quantitative & Institutional Analytics REST API. "
                "Provides point-in-time corporate action resolution, market microstructure (VPIN, Roll/Corwin-Schultz spreads), "
                "macro regime conditional betas, walk-forward algorithmic backtesting, SEC 8-K event classification, "
                "and Black-Scholes higher-order derivatives Greeks."
            ),
            "contact": {
                "name": "Quantitative Systems Team"
            },
            "license": {
                "name": "MIT"
            }
        },
        "servers": [
            {
                "url": "/",
                "description": "Current Environment Server"
            }
        ],
        "tags": [
            {
                "name": "Institutional & Quantitative Analytics",
                "description": "Microstructure, macro conditioning, CAR event studies, and walk-forward simulations"
            },
            {
                "name": "Corporate Actions & Identity",
                "description": "Point-in-time CIK/FIGI/CUSIP resolution, splits, symbol changes, and mergers"
            },
            {
                "name": "Options & Derivatives",
                "description": "Black-Scholes Greeks, IV Smile/Skew, GEX Gamma Exposure, and AI Options Reports"
            },
            {
                "name": "Market Data & Analytics",
                "description": "OHLCV historical charts, fundamental stats, forward estimates, and momentum metrics"
            },
            {
                "name": "Market Positioning & SEC Filings",
                "description": "Finnhub recommendations, insider sentiment (MSPR), institutional 13F holders, raw SEC filings"
            },
            {
                "name": "System & Cache Operations",
                "description": "Health checks, configuration, active ticker discovery, and EOD options cache warming"
            }
        ],
        "paths": {
            "/api/news": {"get": {
                "tags": ["Market Data & Analytics"], "summary": "English-language ticker headlines",
                "description": "A same-origin feed used by the News page. No visitor authentication. "
                               "Requires a server-side Marketaux key. Fetches up to five pages at the provider plan's "
                               "maximum page size, returning at most 500 headlines from seven days "
                               "(up to 15 on the free plan), "
                               "with hourly shared caching and a daily upstream budget. No visitor pagination. "
                               "A 200 may contain empty, unsupported, needs_selection, partial, or stale results; "
                               "503 returns an explanatory feed envelope when no usable snapshot exists.",
                "parameters": [
                    {"name": "ticker", "in": "query", "required": True,
                     "schema": {"type": "string", "maxLength": 32}, "example": "VOD.L"},
                    {"name": "listing", "in": "query", "required": False,
                     "description": "Candidate ID returned by needs_selection; validated for the requested ticker.",
                     "schema": {"type": "string", "maxLength": 200}},
                ],
                "responses": {
                    "200": {"description": "Feed, coverage result, or listing choices", "content": {"application/json": {"schema": feed_schema}}},
                    "400": {"description": "Invalid ticker or listing choice", "content": {"application/problem+json": {"schema": {
                        "type": "object", "properties": {
                            "type": {"type": "string", "example": "/api/docs#news-invalid-request"},
                            "title": {"type": "string"}, "status": {"type": "integer", "enum": [400]},
                            "detail": {"type": "string"},
                        },
                    }}}},
                    "503": {"description": "Unconfigured, quota exhausted, or temporarily unavailable", "content": {"application/json": {
                        "schema": feed_schema,
                        "example": {"ticker": "AAPL", "status": "unconfigured", "listing": None,
                                    "candidates": [], "articles": [], "fetched_at": None, "stale": False,
                                    "partial": False, "message": "News is not configured yet. Please check back later.", "limit": 500},
                    }}},
                },
            }},
            "/api/institutional/{ticker}": {
                "get": {
                    "tags": ["Institutional & Quantitative Analytics"],
                    "summary": "Institutional Quantitative Suite",
                    "description": (
                        "Computes high-frequency microstructure metrics (VPIN, Corwin-Schultz, Roll spread, Squeeze risk), "
                        "macro-financial conditioning (Bull/Bear dual betas, upside/downside capture ratios, yield curve spread), "
                        "walk-forward momentum/mean-reversion signals backtest, and parsed SEC 8-K material event classifications."
                    ),
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "description": "Equity ticker symbol (e.g. AAPL, NVDA, SPY)",
                            "schema": {"type": "string", "example": "AAPL"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Comprehensive institutional analytics report",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/InstitutionalAnalyticsResponse"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Ticker not found or insufficient pricing data",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/corporate-actions/{ticker}": {
                "get": {
                    "tags": ["Corporate Actions & Identity"],
                    "summary": "Point-in-Time Corporate Actions & Entity Timeline",
                    "description": "Resolves entity identification (CIK, FIGI, CUSIP) as-of a specific date and returns chronological corporate actions (splits, symbol changes, mergers).",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "description": "Ticker symbol to look up",
                            "schema": {"type": "string", "example": "AAPL"}
                        },
                        {
                            "name": "as_of",
                            "in": "query",
                            "required": False,
                            "description": "Point-in-time date (YYYY-MM-DD) for historical entity resolution",
                            "schema": {"type": "string", "format": "date", "example": "2020-08-30"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Entity details, symbol history, and corporate action log",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/CorporateActionsResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/options-greeks/{ticker}": {
                "get": {
                    "tags": ["Options & Derivatives"],
                    "summary": "Options Chain & Black-Scholes Greeks",
                    "description": "Returns full options chain strikes with calculated First and Second order Greeks (Delta, Gamma, Theta, Vega, Rho) and implied volatility.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "description": "Underlying equity ticker symbol",
                            "schema": {"type": "string", "example": "AAPL"}
                        },
                        {
                            "name": "expiration",
                            "in": "query",
                            "required": False,
                            "description": "Expiration date (YYYY-MM-DD). Defaults to nearest expiration.",
                            "schema": {"type": "string", "format": "date", "example": "2026-09-18"}
                        },
                        {
                            "name": "rf_rate",
                            "in": "query",
                            "required": False,
                            "description": "Annualized risk-free interest rate",
                            "schema": {"type": "number", "default": 0.045, "example": 0.045}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Options chain greeks grid",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/OptionsGreeksResponse"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Options data unavailable for ticker",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/options-analysis/{ticker}": {
                "get": {
                    "tags": ["Options & Derivatives"],
                    "summary": "Options Posture, GEX & Volatility Analysis",
                    "description": "Computes Put-Call Ratio (PCR), Max Pain, ATM Implied Volatility, IV Rank, IV Percentile, Expected Move, and Dealer Gamma Exposure (GEX). GEX stats come from the options terminal engine (naive convention, live spot, first 8 expiries) so they match /options; gex_stats is null when the terminal errors. hv30/hv90 are null when there are fewer than 30/90 daily returns.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "description": "Underlying ticker symbol",
                            "schema": {"type": "string", "example": "AAPL"}
                        },
                        {
                            "name": "expiration",
                            "in": "query",
                            "required": False,
                            "description": "Expiration date (YYYY-MM-DD)",
                            "schema": {"type": "string", "format": "date"}
                        },
                        {
                            "name": "rf_rate",
                            "in": "query",
                            "required": False,
                            "description": "Risk-free interest rate",
                            "schema": {"type": "number", "default": 0.045}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Quantitative options analysis synthesis",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/OptionsAnalysisResponse"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Failed to compute options analysis",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/options-ai-report/{ticker}": {
                "get": {
                    "tags": ["Options & Derivatives"],
                    "summary": "LLM Generated Options Volatility Synthesis",
                    "description": "Synthesizes dealer positioning, GEX profile, and volatility term structure into an actionable market report.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "AAPL"}
                        },
                        {
                            "name": "expiration",
                            "in": "query",
                            "schema": {"type": "string"}
                        },
                        {
                            "name": "rf_rate",
                            "in": "query",
                            "schema": {"type": "number", "default": 0.045}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "HTML or markdown formatted AI narrative",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "html": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "AI synthesis generation failed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/stock/{ticker}": {
                "get": {
                    "tags": ["Market Data & Analytics"],
                    "summary": "Consolidated Stock Fundamentals & Pricing",
                    "description": "Fetches current market quotes, multi-period percentage returns, and technical indicators.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "AAPL"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Current quote and metrics",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/chart-data/{ticker}": {
                "get": {
                    "tags": ["Market Data & Analytics"],
                    "summary": "Historical OHLCV Chart Series",
                    "description": "Fetches structured historical OHLCV data array with optional start_date and end_date filtering.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "AAPL"}
                        },
                        {
                            "name": "start_date",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "format": "date", "example": "2025-01-01"}
                        },
                        {
                            "name": "end_date",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "format": "date", "example": "2026-01-01"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "OHLCV array for charting",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "dates": {"type": "array", "items": {"type": "string"}},
                                            "open": {"type": "array", "items": {"type": "number"}},
                                            "high": {"type": "array", "items": {"type": "number"}},
                                            "low": {"type": "array", "items": {"type": "number"}},
                                            "close": {"type": "array", "items": {"type": "number"}},
                                            "volume": {"type": "array", "items": {"type": "number"}}
                                        }
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Chart data unavailable",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/analytics/{ticker}": {
                "get": {
                    "tags": ["Market Data & Analytics"],
                    "summary": "Comprehensive Fundamental & Statistical Analytics",
                    "description": "Returns distribution statistics, historical drawdown, valuation multiples, and (under `factors`) the FF5+momentum factor decomposition with Newey-West t-stats, annualised alpha, residual volatility, beta hygiene (OLS/Dimson/Blume/Vasicek) and declared unavailable/insufficient states. Chart HTML is stripped.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "AAPL"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Full analytics payload",
                            "content": {
                                "application/json": {
                                    "schema": {"": "#/components/schemas/AnalyticsResponse"}
                                }
                            }
                        },
                        "404": {
                            "description": "Analytics unavailable",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/positioning/{ticker}": {
                "get": {
                    "tags": ["Market Positioning & SEC Filings"],
                    "summary": "Market Positioning & Institutional Sentiment",
                    "description": "Assembles institutional 13F holders, insider trading transactions, Finnhub monthly MSPR sentiment, and consensus analyst targets.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "AAPL"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Positioning breakdown",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/raw-sec-filings/{ticker}": {
                "get": {
                    "tags": ["Market Positioning & SEC Filings"],
                    "summary": "Raw SEC EDGAR Filings",
                    "description": "Retrieves recent Form 3, 4, 8-K, 10-Q, 10-K, and 13F filings from SEC EDGAR API.",
                    "parameters": [
                        {
                            "name": "ticker",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "AAPL"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of recent filings",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/SECFiling"
                                        }
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Error fetching filings from EDGAR",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/heatmap": {
                "get": {
                    "tags": ["Market Data & Analytics"],
                    "summary": "S&P 500 Heatmap",
                    "description": "Sector-grouped S&P 500 treemap payload: per-constituent daily % change (colour), trailing average dollar volume (tile size), plus breadth counts and universe coverage. Served from cache; returns status 'warming' with empty tiles while the universe is still downloading.",
                    "responses": {
                        "200": {
                            "description": "Heatmap payload",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/config": {
                "get": {
                    "tags": ["System & Cache Operations"],
                    "summary": "Returns whether a Finnhub key is configured",
                    "description": "Returns a boolean flag indicating whether a Finnhub API key is configured server-side. No key material is exposed.",
                    "responses": {
                        "200": {
                            "description": "Configuration object",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "has_finnhub": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/active-tickers": {
                "get": {
                    "tags": ["System & Cache Operations"],
                    "summary": "List Tracked Database Tickers",
                    "description": "Returns distinct equity symbols stored in the SQLite daily_prices table.",
                    "responses": {
                        "200": {
                            "description": "Array of active symbols",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"type": "string", "example": "AAPL"}
                                    }
                                }
                            }
                        },
                        "403": {
                            "description": "Unauthorized access when WARM_CACHE_TOKEN is enforced"
                        }
                    }
                }
            },
            "/api/warm-cache": {
                "post": {
                    "tags": ["System & Cache Operations"],
                    "summary": "Trigger EOD Options Chain Warmer",
                    "description": "Triggers asynchronous background refresh of options chains for all tracked tickers.",
                    "security": [
                        {"BearerAuth": []}
                    ],
                    "responses": {
                        "202": {
                            "description": "Warming job accepted and dispatched in background"
                        },
                        "403": {
                            "description": "Unauthorized invalid bearer token"
                        },
                        "503": {
                            "description": "WARM_CACHE_TOKEN not configured on server"
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "tags": ["System & Cache Operations"],
                    "summary": "Health Check Probe",
                    "description": "Returns 200 OK if the web server is healthy.",
                    "responses": {
                        "200": {
                            "description": "Service is operational",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "ok"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "Token"
                }
            },
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "required": ["error"],
                    "properties": {
                        "error": {"type": "string", "example": "Could not retrieve data"}
                    }
                },
                "SECFiling": {
                    "type": "object",
                    "properties": {
                        "form": {"type": "string", "example": "8-K"},
                        "filingDate": {"type": "string", "format": "date", "example": "2026-02-14"},
                        "reportDate": {"type": "string", "format": "date"},
                        "acceptanceDateTime": {"type": "string"},
                        "act": {"type": "string"},
                        "fileNumber": {"type": "string"},
                        "accessionNumber": {"type": "string"},
                        "items": {"type": "string", "example": "2.02,7.01"},
                        "size": {"type": "integer"},
                        "isXBRL": {"type": "integer"},
                        "isInlineXBRL": {"type": "integer"},
                        "primaryDocument": {"type": "string"},
                        "primaryDocDescription": {"type": "string"},
                        "url": {"type": "string", "format": "uri"}
                    }
                },
                "InstitutionalAnalyticsResponse": {
                    "type": "object",
                    "required": ["ticker", "microstructure", "macro_conditioning", "signals_backtest", "sec_8k_events"],
                    "properties": {
                        "ticker": {"type": "string", "example": "AAPL"},
                        "microstructure": {
                            "type": "object",
                            "properties": {
                                "corwin_schultz_spread_pct": {"type": "number", "example": 0.42},
                                "roll_effective_spread_pct": {"type": "number", "example": 0.38},
                                "amihud_illiquidity": {"type": "number"},
                                "squeeze_risk_score": {"type": "number", "example": 45.0},
                                "squeeze_risk_level": {"type": "string", "example": "MODERATE"},
                                "squeeze_inputs_assumed": {
                                    "type": "array", "items": {"type": "string"},
                                    "example": ["SI % float", "DTC"],
                                    "description": "Squeeze inputs that fell back to defaults (no short-interest feed). Cost-to-borrow is excluded from the score unless reported."
                                }
                            }
                        },
                        "macro_conditioning": {
                            "type": "object",
                            "properties": {
                                "regime": {"type": "string", "nullable": True, "example": "NEUTRAL_EXPANSION"},
                                "fed_funds_rate": {"type": "number", "nullable": True, "example": 4.50},
                                "yield_curve_2s10s": {"type": "number", "nullable": True, "example": 0.35},
                                "is_inverted": {"type": "boolean", "nullable": True, "example": False},
                                "cpi_yoy": {"type": "number", "nullable": True, "example": 2.8},
                                "betas": {
                                    "type": "object",
                                    "description": "Up/down-market betas (split by the sign of SPY's daily return).",
                                    "properties": {
                                        "overall": {"type": "number", "example": 1.15},
                                        "bull_beta": {"type": "number", "example": 1.25},
                                        "bear_beta": {"type": "number", "example": 0.95},
                                        "bull_days": {"type": "integer", "example": 310},
                                        "bear_days": {"type": "integer", "example": 254}
                                    }
                                },
                                "regime_betas": {
                                    "type": "object", "nullable": True,
                                    "description": "Betas conditional on the FRED regime at each date (curve sign, 3m fed-funds direction); each bucket {beta, n}, beta withheld below 60 obs.",
                                    "properties": {
                                        "curve_inverted": {"type": "object"},
                                        "curve_steep": {"type": "object"},
                                        "rate_tightening": {"type": "object"},
                                        "rate_pause": {"type": "object"},
                                        "rate_easing": {"type": "object"}
                                    }
                                },
                                "upside_capture": {"type": "number", "nullable": True, "example": 112.0,
                                                    "description": "Geometric monthly convention; null below 6 up-market months."},
                                "downside_capture": {"type": "number", "nullable": True, "example": 88.0,
                                                      "description": "Geometric monthly convention; null below 6 down-market months."}
                            }
                        },
                        "short_interest": {
                            "type": "object",
                            "properties": {
                                "shares_short": {"type": "integer", "nullable": True},
                                "short_pct_float": {"type": "number", "nullable": True, "example": 0.024},
                                "days_to_cover": {"type": "number", "nullable": True, "example": 1.8},
                                "shares_short_prior_month": {"type": "integer", "nullable": True},
                                "si_mom_change": {"type": "number", "nullable": True},
                                "as_of": {"type": "string", "nullable": True},
                                "source": {"type": "string", "nullable": True, "example": "reported"},
                                "si_source": {"type": "string", "enum": ["reported", "assumed"], "example": "reported"}
                            }
                        },
                        "signals_backtest": {
                            "type": "object",
                            "properties": {
                                "total_return_pct": {"type": "number", "example": 28.5},
                                "cagr_pct": {"type": "number", "example": 14.2},
                                "benchmark_return_pct": {"type": "number", "example": 15.2},
                                "alpha_pct": {"type": "number", "example": 13.3},
                                "annualized_sharpe": {"type": "number", "example": 1.65},
                                "annualized_sortino": {"type": "number", "example": 2.15},
                                "max_drawdown_pct": {"type": "number", "example": -8.4},
                                "calmar_ratio": {"type": "number", "example": 1.69},
                                "win_rate_pct": {"type": "number", "example": 58.2},
                                "profit_factor": {"type": "number", "example": 1.45},
                                "total_trades": {"type": "integer", "example": 42},
                                "factors_used": {"type": "array", "items": {"type": "string"}, "example": ["MOM_12_1", "VOL_60D"]},
                                "n_folds": {"type": "integer", "example": 5},
                                "oos_sharpe_mean": {"type": "number", "example": 1.25},
                                "oos_sharpe_std": {"type": "number", "example": 0.35},
                                "fold_returns": {"type": "array", "items": {"type": "number"}},
                                "deflated_sharpe": {"type": "number", "example": 0.92},
                                "n_trials_assumed": {"type": "integer", "example": 10}
                            }
                        },
                        "permutation_test": {
                            "type": "object",
                            "description": (
                                "Block-bootstrap permutation null test: the same strategy rerun on "
                                "block-shuffled copies of the return series. p_value is the "
                                "bias-corrected fraction of null Sharpes >= the real Sharpe, so its "
                                "floor is 1/(n_permutations+1), never exactly 0."
                            ),
                            "properties": {
                                "real_sharpe": {"type": "number", "example": 1.65},
                                "null_mean": {"type": "number", "example": 0.12},
                                "null_std": {"type": "number", "example": 0.78},
                                "p_value": {"type": "number", "example": 0.0396},
                                "n_permutations": {"type": "integer", "example": 100}
                            }
                        },
                        "sec_8k_events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "accession_number": {"type": "string"},
                                    "filing_date": {"type": "string"},
                                    "items": {"type": "array", "items": {"type": "string"}},
                                    "primary_category": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "CorporateActionsResponse": {
                    "type": "object",
                    "required": ["ticker", "actions"],
                    "properties": {
                        "ticker": {"type": "string", "example": "AAPL"},
                        "entity": {
                            "type": "object",
                            "nullable": True,
                            "properties": {
                                "entity_id": {"type": "string", "example": "AAPL_INC"},
                                "cik": {"type": "string", "example": "0000320193"},
                                "figi": {"type": "string", "example": "BBG000B9XRY4"},
                                "cusip": {"type": "string", "example": "037833100"},
                                "legal_name": {"type": "string", "example": "Apple Inc."}
                            }
                        },
                        "symbol_timeline": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "valid_from": {"type": "string"},
                                    "valid_to": {"type": "string", "nullable": True}
                                }
                            }
                        },
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action_id": {"type": "string"},
                                    "action_type": {"type": "string", "example": "SPLIT"},
                                    "effective_date": {"type": "string", "example": "2020-08-31"},
                                    "announcement_date": {"type": "string", "nullable": True},
                                    "ratio": {"type": "number", "example": 4.0},
                                    "old_value": {"type": "string", "nullable": True},
                                    "new_value": {"type": "string", "nullable": True},
                                    "status": {"type": "string", "example": "EFFECTIVE"}
                                }
                            }
                        }
                    }
                },
                "OptionsGreeksResponse": {
                    "type": "object",
                    "required": ["ticker", "expirations", "selected_expiration", "spot_price", "days_to_expiration", "options"],
                    "properties": {
                        "ticker": {"type": "string", "example": "AAPL"},
                        "expirations": {"type": "array", "items": {"type": "string"}},
                        "selected_expiration": {"type": "string", "example": "2026-09-18"},
                        "spot_price": {"type": "number", "example": 225.50},
                        "days_to_expiration": {"type": "integer", "example": 33},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "strike": {"type": "number", "example": 225.0},
                                    "call_bid": {"type": "number"},
                                    "call_ask": {"type": "number"},
                                    "call_last": {"type": "number"},
                                    "call_iv": {"type": "number"},
                                    "call_delta": {"type": "number", "example": 0.52},
                                    "call_gamma": {"type": "number", "example": 0.035},
                                    "call_theta": {"type": "number", "example": -0.082},
                                    "call_vega": {"type": "number", "example": 0.24},
                                    "put_bid": {"type": "number"},
                                    "put_ask": {"type": "number"},
                                    "put_last": {"type": "number"},
                                    "put_iv": {"type": "number"},
                                    "put_delta": {"type": "number", "example": -0.48}
                                }
                            }
                        }
                    }
                },
                "AnalyticsResponse": {
                    "type": "object",
                    "required": ["ticker", "current_price", "stats"],
                    "properties": {
                        "ticker": {"type": "string", "example": "AAPL"},
                        "current_price": {"type": "number", "example": 225.50},
                        "stats": {
                            "type": "object",
                            "properties": {
                                "mean_daily_return": {"type": "number", "example": 0.082},
                                "daily_std": {"type": "number", "example": 1.25},
                                "annualised_vol": {"type": "number", "example": 19.84},
                                "skewness": {"type": "number", "example": -0.15},
                                "kurtosis": {"type": "number", "example": 3.42},
                                "max_drawdown": {"type": "number", "example": -14.2},
                                "beta": {"type": "number", "nullable": True, "example": 1.12},
                                "data_points": {"type": "integer", "example": 504},
                                "var_95": {"type": "number", "nullable": True, "example": -1.85},
                                "var_99": {"type": "number", "nullable": True, "example": -2.95},
                                "poc_price": {"type": "number", "nullable": True, "example": 220.00},
                                "sharpe_stats": {
                                    "type": "object",
                                    "nullable": True,
                                    "description": "Rolling Sharpe ratio with Lo (2002) SE and PSR"
                                }
                            }
                        },
                        "gics_sector": {
                            "type": "string",
                            "nullable": True,
                            "example": "Information Technology"
                        },
                        "percentiles": {
                            "type": "object",
                            "nullable": True,
                            "description": "Cross-sectional peer percentiles against S&P 500 and sector"
                        },
                        "signals_lab": {
                            "type": "object",
                            "nullable": True,
                            "description": "Signal ranks and validation status"
                        },
                        "seasonality_obs": {
                            "type": "array",
                            "nullable": True,
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "integer"},
                            "description": "[min_observations, max_observations] per calendar month",
                            "example": [10, 10]
                        }
                    }
                },
                "OptionsAnalysisResponse": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "example": "AAPL"},
                        "spot_price": {"type": "number", "example": 225.50},
                        "selected_expiration": {"type": "string"},
                        "days_to_expiration": {"type": "integer"},
                        "hv30": {"type": "number", "nullable": True, "example": 18.5},
                        "hv90": {"type": "number", "nullable": True, "example": 21.2},
                        "atm_iv": {"type": "number", "example": 23.4},
                        "iv_rank": {"type": "number", "example": 62.5},
                        "iv_percentile": {"type": "number", "example": 68.0},
                        "expected_move_bs": {"type": "number", "example": 12.8},
                        "expected_move_straddle": {"type": "number", "example": 13.1},
                        "vol_pcr": {"type": "number", "example": 0.85},
                        "oi_pcr": {"type": "number", "example": 0.92},
                        "max_pain": {"type": "number", "example": 220.0},
                        "strategy_recommendation": {"type": "string", "example": "Sell Premium (Strangle / Iron Condor)"},
                        "strategy_rationale": {"type": "string"}
                    }
                }
            }
        }
    }


def get_swagger_ui_html(openapi_json_url: str = "/api/openapi.json", title: str = "Ticker-change Quantitative Engine API") -> str:
    """Return an interactive standalone Swagger UI HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <link rel="icon" type="image/png" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/favicon-32x32.png" sizes="32x32" />
  <link rel="icon" type="image/png" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/favicon-16x16.png" sizes="16x16" />
  <style>
    html {{
      box-sizing: border-box;
      overflow: -moz-scrollbars-vertical;
      overflow-y: scroll;
    }}
    *, *:before, *:after {{
      box-sizing: inherit;
    }}
    body {{
      margin:0;
      background: #0f172a;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .topbar-header {{
      background: #1e293b;
      padding: 16px 24px;
      border-bottom: 1px solid #334155;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .topbar-header h1 {{
      margin: 0;
      font-size: 1.25rem;
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .topbar-links a {{
      color: #94a3b8;
      text-decoration: none;
      margin-left: 16px;
      font-size: 0.875rem;
      font-weight: 500;
      transition: color 0.15s ease;
    }}
    .topbar-links a:hover {{
      color: #38bdf8;
    }}
    /* Swagger UI Overrides for clean readability */
    .swagger-ui {{
      background: #ffffff;
      padding-bottom: 40px;
    }}
  </style>
</head>
<body>
  <div class="topbar-header">
    <h1>
      <span>⚡</span> {title}
    </h1>
    <div class="topbar-links">
      <a href="/" target="_self">Dashboard</a>
      <a href="/glossary" target="_self">Glossary</a>
      <a href="/api/openapi.json" target="_blank">OpenAPI JSON</a>
    </div>
  </div>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
  <script>
    window.onload = function() {{
      window.ui = SwaggerUIBundle({{
        url: "{openapi_json_url}",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        plugins: [
          SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "BaseLayout",
        defaultModelsExpandDepth: 2,
        defaultModelExpandDepth: 2,
        docExpansion: "list",
        filter: true
      }});
    }};
  </script>
</body>
</html>
"""
