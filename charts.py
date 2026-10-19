"""Native ECharts data and safe, inert chart mounts for Jinja pages."""
import html
import json
import math
import uuid
from datetime import date, datetime

import numpy as np
import pandas as pd


def normalize(value):
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [normalize(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def render_chart(option, *, chart_id=None, height=320):
    chart_id = chart_id or 'chart-' + uuid.uuid4().hex
    safe_id = html.escape(chart_id, quote=True)
    body = json.dumps(normalize(option), separators=(',', ':'), allow_nan=False)
    for char, escaped in [('&', r'\u0026'), ('<', r'\u003c'), ('>', r'\u003e'), ('\u2028', r'\u2028'), ('\u2029', r'\u2029')]:
        body = body.replace(char, escaped)
    return (f'<div class="terminal-chart" id="{safe_id}" style="height:{int(height)}px" role="img" aria-label="Interactive financial chart"></div>'
            f'<script type="application/json" data-chart="{safe_id}">{body}</script>')


def pairs(x, y):
    return [[v.strftime('%Y-%m-%d') if isinstance(v, (datetime, date)) else v, w] for v, w in zip(x, y)]


def line(name, x, y, color='#f59e0b', *, width=1.8, area=None, dash=False, **extra):
    result = {'name': name, 'type': 'line', 'showSymbol': False, 'connectNulls': False,
              'data': pairs(x, y), 'lineStyle': {'color': color, 'width': width, 'type': 'dashed' if dash else 'solid'},
              'itemStyle': {'color': color}, **extra}
    if area:
        result['areaStyle'] = {'color': area}
    return result


def bar(name, x, y, colors='#6366f1'):
    data = pairs(x, y)
    if not isinstance(colors, str):
        data = [{'value': p, 'itemStyle': {'color': c}} for p, c in zip(data, colors)]
    return {'name': name, 'type': 'bar', 'data': data, 'itemStyle': {'color': colors} if isinstance(colors, str) else {}, 'barMaxWidth': 22}


def reference(axis, value, name='', color='#71717a'):
    return {axis + 'Axis': value, 'name': name, 'label': {'formatter': name, 'position': 'insideEnd'}, 'lineStyle': {'color': color, 'type': 'dashed'}}


def cartesian(series, *, xname='', yname='', xtype='time', categories=None, title='', refs=None):
    option = {'series': series, 'xAxis': {'type': xtype, 'name': xname}, 'yAxis': {'type': 'value', 'name': yname, 'scale': True}}
    if xtype == 'value':
        option['xAxis']['scale'] = True
    if categories is not None:
        option['xAxis']['data'] = list(categories)
    if title:
        option['title'] = {'text': title}
    if refs and series:
        series[0]['markLine'] = {'symbol': 'none', 'silent': True, 'data': refs}
    return option
