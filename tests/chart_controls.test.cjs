const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function library(file, name) {
 const context={window:{},module:{exports:{}},console};
 vm.runInNewContext(fs.readFileSync(file,'utf8'),context);
 return context.window[name];
}
test('calendar month range clamps leap-day and YTD begins January 1',()=>{
 const c=library('static/stock-chart.js','StockChart');
 assert.equal(c.rangeStart('1M','2024-03-31'),'2024-02-29');
 assert.equal(c.rangeStart('1Y','2024-02-29'),'2023-02-28');
 assert.equal(c.rangeStart('YTD','2024-08-15'),'2024-01-01');
});
test('option surface pairs each row with DTE and each column with moneyness',()=>{
 const c=library('static/charts.js','TerminalCharts');
 const points=c.surfaceData({moneyness:[.9,1.1],dtes:[7,30],iv_matrix:[[20,21],[22,null]]});
 assert.equal(JSON.stringify(points),'[[0.9,7,20],[1.1,7,21],[0.9,30,22]]');
});
test('numeric coordinates preserve gaps rather than manufacturing zeroes',()=>{
 const c=library('static/charts.js','TerminalCharts');
 assert.equal(JSON.stringify(c.points([90,100,110],[1,'N/A',null])),'[[90,1],[100,null],[110,null]]');
});
