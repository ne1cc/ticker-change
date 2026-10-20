(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.OptionsPayoff = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const CONTRACT_SIZE = 100;
  const VALID_SIDES = new Set(['long', 'short']);
  const VALID_TYPES = new Set(['call', 'put']);
  const sign = side => side === 'long' ? 1 : -1;
  const finite = value => typeof value === 'number' && Number.isFinite(value);

  function validateLeg(leg, {requireIv = false} = {}) {
    const errors = [];
    if (!leg || !VALID_SIDES.has(leg.side)) errors.push('Choose buy or sell.');
    if (!leg || !VALID_TYPES.has(leg.type)) errors.push('Choose call or put.');
    if (!leg || !finite(leg.strike) || leg.strike <= 0) errors.push('Strike must be greater than zero.');
    if (!leg || !finite(leg.premium) || leg.premium < 0) errors.push('Enter a premium of zero or more.');
    if (!leg || !Number.isInteger(leg.quantity) || leg.quantity <= 0) errors.push('Contracts must be a positive whole number.');
    if (requireIv && (!leg || !finite(leg.ivPct) || leg.ivPct < 0)) errors.push('Enter a valid implied volatility.');
    return errors;
  }

  function intrinsic(type, spot, strike) {
    if (!VALID_TYPES.has(type) || !finite(spot) || spot < 0 || !finite(strike) || strike <= 0) return null;
    return type === 'call' ? Math.max(spot - strike, 0) : Math.max(strike - spot, 0);
  }

  function expiryPnl(legs, spot) {
    if (!Array.isArray(legs) || !legs.length || !finite(spot) || spot < 0 || legs.some(leg => validateLeg(leg).length)) return null;
    return legs.reduce((total, leg) => total + sign(leg.side) * (intrinsic(leg.type, spot, leg.strike) - leg.premium) * leg.quantity * CONTRACT_SIZE, 0);
  }

  function nearlyEqual(a, b) {
    return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
  }

  function analyzeExpiry(legs) {
    if (!Array.isArray(legs) || !legs.length || legs.some(leg => validateLeg(leg).length)) return null;
    const strikes = [...new Set(legs.map(leg => leg.strike))].sort((a, b) => a - b);
    const knots = [...new Set([0, ...strikes])];
    const values = knots.map(spot => expiryPnl(legs, spot));
    const tailSlope = legs.reduce((sum, leg) => sum + (leg.type === 'call' ? sign(leg.side) * leg.quantity * CONTRACT_SIZE : 0), 0);
    const breakevens = [];
    const breakevenRanges = [];
    let segmentStart = null;

    function addRoot(root) {
      if (!Number.isFinite(root) || root < 0) return;
      if (!breakevens.some(existing => nearlyEqual(existing, root))) breakevens.push(root);
    }
    function closeZeroRange(end) {
      if (segmentStart == null) return;
      breakevenRanges.push([segmentStart, end]);
      segmentStart = null;
    }

    knots.forEach((spot, index) => {
      const value = values[index];
      const zero = nearlyEqual(value, 0);
      if (zero) {
        if (segmentStart == null) segmentStart = spot;
      } else if (segmentStart != null) {
        closeZeroRange(spot);
      }
      if (index === knots.length - 1) return;
      const nextSpot = knots[index + 1];
      const nextValue = values[index + 1];
      if (!zero && !nearlyEqual(nextValue, 0) && Math.sign(value) !== Math.sign(nextValue)) {
        addRoot(spot + (nextSpot - spot) * (-value / (nextValue - value)));
      }
      if (zero && !nearlyEqual(nextValue, 0)) closeZeroRange(spot);
    });

    const tailAt = values[values.length - 1];
    const lastStrike = knots[knots.length - 1];
    if (nearlyEqual(tailAt, 0) && nearlyEqual(tailSlope, 0)) {
      if (segmentStart == null) segmentStart = lastStrike;
      closeZeroRange(Infinity);
    } else {
      if (nearlyEqual(tailAt, 0)) addRoot(lastStrike);
      if (!nearlyEqual(tailSlope, 0)) {
        const tailRoot = lastStrike - tailAt / tailSlope;
        if (tailRoot > lastStrike && Number.isFinite(tailRoot)) addRoot(tailRoot);
      }
    }

    const minAtKnots = Math.min(...values);
    const maxAtKnots = Math.max(...values);
    const tailUnbounded = tailSlope > 0 ? 'profit' : tailSlope < 0 ? 'loss' : null;
    return {
      netDebit: legs.reduce((sum, leg) => sum + sign(leg.side) * leg.premium * leg.quantity * CONTRACT_SIZE, 0),
      maxProfit: tailUnbounded === 'profit' ? Infinity : Math.max(0, maxAtKnots),
      maxLoss: tailUnbounded === 'loss' ? Infinity : Math.max(0, -minAtKnots),
      breakevens: breakevens.sort((a, b) => a - b),
      breakevenRanges,
      knots,
      tailSlope
    };
  }

  function normalCdf(x) {
    const absX = Math.abs(x);
    const t = 1 / (1 + 0.2316419 * absX);
    const density = 0.3989422804014327 * Math.exp(-absX * absX / 2);
    const tail = density * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1 - tail : tail;
  }

  function optionValue({type, spot, strike, tYears, iv, rate}) {
    if (!VALID_TYPES.has(type) || !finite(spot) || spot < 0 || !finite(strike) || strike <= 0 ||
        !finite(tYears) || tYears < 0 || !finite(iv) || iv < 0 || !finite(rate)) return null;
    if (tYears === 0) return intrinsic(type, spot, strike);
    const discountedStrike = strike * Math.exp(-rate * tYears);
    if (iv === 0 || spot === 0) {
      if (spot === 0) return type === 'call' ? 0 : discountedStrike;
      return type === 'call' ? Math.max(spot - discountedStrike, 0) : Math.max(discountedStrike - spot, 0);
    }
    const sqrtT = Math.sqrt(tYears);
    const d1 = (Math.log(spot / strike) + (rate + 0.5 * iv * iv) * tYears) / (iv * sqrtT);
    const d2 = d1 - iv * sqrtT;
    return type === 'call'
      ? spot * normalCdf(d1) - discountedStrike * normalCdf(d2)
      : discountedStrike * normalCdf(-d2) - spot * normalCdf(-d1);
  }

  function valuePosition(legs, {spot, tYears, rate, ivMultiplier = 1}) {
    if (!Array.isArray(legs) || !legs.length || !finite(spot) || spot < 0 || !finite(tYears) || tYears < 0 ||
        !finite(rate) || !finite(ivMultiplier) || ivMultiplier < 0 || legs.some(leg => validateLeg(leg, {requireIv:tYears > 0}).length)) return null;
    if (tYears === 0) {
      const pnl = expiryPnl(legs, spot);
      const legValues = legs.map(leg => sign(leg.side) * intrinsic(leg.type, spot, leg.strike) * leg.quantity * CONTRACT_SIZE);
      return {pnl, contractValue:legValues.reduce((sum,value) => sum+value,0), legValues};
    }
    const legValues = [];
    let pnl = 0;
    let contractValue = 0;
    for (const leg of legs) {
      const value = optionValue({type:leg.type,spot,strike:leg.strike,tYears,iv:leg.ivPct / 100 * ivMultiplier,rate});
      if (value == null) return null;
      const signedValue = sign(leg.side) * value * leg.quantity * CONTRACT_SIZE;
      legValues.push(signedValue);
      contractValue += signedValue;
      pnl += sign(leg.side) * (value - leg.premium) * leg.quantity * CONTRACT_SIZE;
    }
    return {pnl,contractValue,legValues};
  }

  function scenarioPrices(spot, rangePct) {
    if (!finite(spot) || spot <= 0 || !finite(rangePct) || rangePct < 0) return [];
    const low = Math.max(0, spot * (1 - rangePct / 100));
    const high = spot * (1 + rangePct / 100);
    const prices = Array.from({length:21}, (_,index) => high - (high-low)*index/20);
    prices[10] = spot;
    return [...new Set(prices.map(value => Number(value.toFixed(10))))].sort((a,b) => b-a);
  }

  function scenarioTimes(nowMs, expiryMs) {
    if (!finite(nowMs) || !finite(expiryMs)) return [];
    if (expiryMs <= nowMs) return [expiryMs];
    const count = Math.min(14, Math.max(1, Math.ceil((expiryMs-nowMs)/86400000)));
    return Array.from({length:count+1}, (_,index) => nowMs+(expiryMs-nowMs)*index/count);
  }

  function metricValue(value, mode, analysis) {
    if (!value || !analysis) return null;
    if (mode === 'pnl') return value.pnl;
    if (mode === 'contractValue') return value.contractValue;
    if (mode === 'pnlPercent') {
      const basis = Math.abs(analysis.netDebit);
      return basis > 0 ? value.pnl / basis * 100 : null;
    }
    if (mode === 'maxRiskPercent') return finite(analysis.maxLoss) && analysis.maxLoss > 0 ? value.pnl / analysis.maxLoss * 100 : null;
    return null;
  }

  return {CONTRACT_SIZE,validateLeg,intrinsic,expiryPnl,analyzeExpiry,optionValue,valuePosition,scenarioPrices,scenarioTimes,metricValue};
});
