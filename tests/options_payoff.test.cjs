const {test} = require('node:test');
const assert = require('node:assert/strict');
const payoff = require('../static/options-payoff.js');

test('bull call spread has exact risk and breakeven', () => {
  const legs = [
    {side:'long',type:'call',strike:100,premium:5,quantity:1,ivPct:20},
    {side:'short',type:'call',strike:110,premium:2,quantity:1,ivPct:20}
  ];
  const result = payoff.analyzeExpiry(legs);
  assert.equal(result.netDebit, 300);
  assert.equal(result.maxLoss, 300);
  assert.equal(result.maxProfit, 700);
  assert.deepEqual(result.breakevens, [103]);
  assert.equal(payoff.expiryPnl(legs, 103), 0);
  assert.equal(payoff.expiryPnl(legs, 200), 700);
});

test('long straddle returns both breakevens and an unbounded profit side', () => {
  const result = payoff.analyzeExpiry([
    {side:'long',type:'call',strike:100,premium:5,quantity:1},
    {side:'long',type:'put',strike:100,premium:4,quantity:1}
  ]);
  assert.equal(result.netDebit, 900);
  assert.equal(result.maxLoss, 900);
  assert.equal(result.maxProfit, Infinity);
  assert.deepEqual(result.breakevens, [91, 109]);
});

test('iron condor has bounded risk and exact two breakevens', () => {
  const result = payoff.analyzeExpiry([
    {side:'long',type:'put',strike:90,premium:1,quantity:1},
    {side:'short',type:'put',strike:95,premium:2,quantity:1},
    {side:'short',type:'call',strike:105,premium:2,quantity:1},
    {side:'long',type:'call',strike:110,premium:1,quantity:1}
  ]);
  assert.equal(result.netDebit, -200);
  assert.equal(result.maxProfit, 200);
  assert.equal(result.maxLoss, 300);
  assert.deepEqual(result.breakevens, [93, 107]);
});

test('call butterfly accounts for its two contracts on the middle leg', () => {
  const legs = [
    {side:'long',type:'call',strike:95,premium:7,quantity:1},
    {side:'short',type:'call',strike:100,premium:4,quantity:2},
    {side:'long',type:'call',strike:105,premium:2,quantity:1}
  ];
  const result = payoff.analyzeExpiry(legs);
  assert.equal(result.netDebit, 100);
  assert.equal(result.maxLoss, 100);
  assert.equal(result.maxProfit, 400);
  assert.deepEqual(result.breakevens, [96, 104]);
});

test('naked single options and an unequal ratio spread use the combined tail slope', () => {
  for (const type of ['call', 'put']) {
    for (const side of ['long', 'short']) {
      const result = payoff.analyzeExpiry([{side,type,strike:100,premium:5,quantity:1}]);
      assert.ok(result);
      if (type === 'call' && side === 'long') assert.equal(result.maxProfit, Infinity);
      if (type === 'call' && side === 'short') assert.equal(result.maxLoss, Infinity);
      if (type === 'put' && side === 'long') assert.equal(result.maxProfit, 9500);
      if (type === 'put' && side === 'short') assert.equal(result.maxLoss, 9500);
    }
  }
  const ratio = payoff.analyzeExpiry([
    {side:'long',type:'call',strike:100,premium:4,quantity:1},
    {side:'short',type:'call',strike:110,premium:2,quantity:2}
  ]);
  assert.equal(ratio.maxProfit, 1000);
  assert.equal(ratio.maxLoss, Infinity);
  assert.deepEqual(ratio.breakevens, [120]);
});

test('exactly offsetting identical legs describe a zero breakeven interval', () => {
  const result = payoff.analyzeExpiry([
    {side:'long',type:'call',strike:100,premium:3,quantity:2},
    {side:'short',type:'call',strike:100,premium:3,quantity:2}
  ]);
  assert.equal(result.maxProfit, 0);
  assert.equal(result.maxLoss, 0);
  assert.deepEqual(result.breakevens, []);
  assert.deepEqual(result.breakevenRanges, [[0, Infinity]]);
  assert.equal(result.tailSlope, 0);
});

test('negative roots are excluded and invalid leg values do not become valid trades', () => {
  const deepItmPut = payoff.analyzeExpiry([{side:'long',type:'put',strike:100,premium:150,quantity:1}]);
  assert.deepEqual(deepItmPut.breakevens, []);
  for (const patch of [{strike:NaN},{strike:0},{premium:-1},{quantity:1.5},{quantity:0},{side:'buy'},{type:'stock'}]) {
    const leg = {side:'long',type:'call',strike:100,premium:5,quantity:1,...patch};
    assert.ok(payoff.validateLeg(leg).length > 0);
    assert.equal(payoff.expiryPnl([leg], 110), null);
    assert.equal(payoff.analyzeExpiry([leg]), null);
  }
});

test('Black–Scholes matches standard reference prices and put-call parity', () => {
  const args = {spot:100,strike:100,tYears:1,iv:.2,rate:.05};
  const call = payoff.optionValue({...args,type:'call'});
  const put = payoff.optionValue({...args,type:'put'});
  assert.ok(Math.abs(call - 10.45058357) < 1e-5);
  assert.ok(Math.abs(put - 5.57352602) < 1e-5);
  assert.ok(Math.abs(call-put-(100-100*Math.exp(-.05))) < 1e-5);
});

test('option values have explicit expiry, zero-volatility, zero-spot, and zero-rate limits', () => {
  assert.equal(payoff.optionValue({type:'call',spot:110,strike:100,tYears:0,iv:.3,rate:.05}),10);
  assert.equal(payoff.optionValue({type:'put',spot:90,strike:100,tYears:0,iv:.3,rate:.05}),10);
  assert.ok(Math.abs(payoff.optionValue({type:'call',spot:100,strike:105,tYears:1,iv:0,rate:.05}) - (100-105*Math.exp(-.05)))<1e-10);
  assert.equal(payoff.optionValue({type:'put',spot:100,strike:105,tYears:1,iv:0,rate:.05}),0);
  assert.equal(payoff.optionValue({type:'call',spot:0,strike:105,tYears:1,iv:.2,rate:.05}),0);
  assert.ok(Math.abs(payoff.optionValue({type:'call',spot:100,strike:100,tYears:1,iv:.2,rate:0}) - 7.96556745)<2e-5);
});

test('position scenario uses intrinsic value at expiry and requires IV only before expiry', () => {
  const legs = [{side:'long',type:'call',strike:100,premium:5,quantity:1,ivPct:null}];
  assert.equal(payoff.valuePosition(legs,{spot:100,tYears:0,rate:0,ivMultiplier:1}).pnl,-500);
  assert.equal(payoff.valuePosition(legs,{spot:100,tYears:1,rate:0,ivMultiplier:1}),null);
  assert.ok(payoff.valuePosition([{...legs[0],ivPct:0}],{spot:100,tYears:1,rate:0,ivMultiplier:1}));
  assert.equal(payoff.metricValue({pnl:10,contractValue:200},'pnlPercent',{netDebit:0,maxLoss:20}),null);
  assert.equal(payoff.metricValue({pnl:10,contractValue:200},'maxRiskPercent',{netDebit:0,maxLoss:Infinity}),null);
  assert.equal(payoff.metricValue({pnl:10,contractValue:200},'maxRiskPercent',{netDebit:0,maxLoss:0}),null);
});

test('scenario grids include spot, stay ordered, and cap dates at expiry', () => {
  const prices = payoff.scenarioPrices(100,20);
  assert.equal(prices.length,21);
  assert.equal(prices[10],100);
  assert.equal(prices[0],120);
  assert.equal(prices.at(-1),80);
  const zeroBound = payoff.scenarioPrices(100,100);
  assert.ok(zeroBound.includes(0));
  assert.equal(new Set(zeroBound).size,zeroBound.length);
  const dates = payoff.scenarioTimes(0,32*86400000);
  assert.ok(dates.length<=15);
  assert.equal(dates[0],0);
  assert.equal(dates.at(-1),32*86400000);
  assert.deepEqual(payoff.scenarioTimes(10,5),[5]);
});

test('P&L metric modes use the opening cash amount or finite maximum risk', () => {
  const position = {pnl:250,contractValue:750};
  const analysis = {netDebit:500,maxLoss:1000};
  assert.equal(payoff.metricValue(position,'pnl',analysis),250);
  assert.equal(payoff.metricValue(position,'pnlPercent',analysis),50);
  assert.equal(payoff.metricValue(position,'contractValue',analysis),750);
  assert.equal(payoff.metricValue(position,'maxRiskPercent',analysis),25);
});
