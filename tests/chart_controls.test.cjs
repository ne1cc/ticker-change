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

function rendererHarness() {
 let visibility, domReady, instance;
 const element={clientWidth:800,clientHeight:460,style:{},attributes:{},classList:{add(){}},
   setAttribute(k,v){this.attributes[k]=v;},before(controls){this.controls=controls;}};
 const document={documentElement:{classList:{contains(){return false;}}},head:{append(script){queueMicrotask(()=>script.onload());}},
   createElement(tag){return {tag,style:{},children:[],setAttribute(){},append(...items){this.children.push(...items);},click(){}};},
   getElementById(){return element;},querySelectorAll(){return [];},addEventListener(name,fn){if(name==='DOMContentLoaded')domReady=fn;}};
 const context={window:{},document,console,location:{},
   IntersectionObserver:class{constructor(fn){visibility=fn;}observe(target){this.target=target;}},
   ResizeObserver:class{observe(){}},MutationObserver:class{observe(){}},
   echarts:{init(){instance={option:null,events:{},on(name,fn){this.events[name]=fn;},
     setOption(option){this.option=option;},getOption(){return {dataZoom:this.option?.dataZoom||[],legend:[{selected:this.option?.legend?.selected||{}}]};},resize(){},
     dispatchAction(){},getDataURL(){return 'data:image/png;base64,';}};return instance;}}};
 vm.runInNewContext(fs.readFileSync('static/charts.js','utf8'),context);
 domReady();
 return {context,element,get instance(){return instance;},show(){visibility([{target:element,isIntersecting:true}]);}};
}
const flush=()=>new Promise(resolve=>setTimeout(resolve,0));
const chartOption=(key, extra={})=>({series:[{name:'Calls',type:'bar',data:[[100,1]],itemStyle:{color:'#2563eb'}}],
 terminalComparison:true,terminalLegendFontSize:12,terminalLegendSelected:{Calls:true,Puts:true,'Net GEX':true},terminalViewKey:key,...extra});

test('comparison view key resets zoom and legend on view changes but preserves same-view refresh state',async()=>{
 const h=rendererHarness(); h.context.window.TerminalCharts.mount(h.element,chartOption('gex:near:exp:100'));
 h.show(); await flush();
 assert.equal(h.instance.option.legend.textStyle.fontSize,12);
 h.instance.option.legend.selected.Calls=false; h.instance.option.dataZoom=[{start:30,end:60}];
 h.context.window.TerminalCharts.mount(h.element,chartOption('gex:near:exp:100',{series:[{name:'Calls',type:'bar',data:[[100,2]]}]}));
 assert.equal(h.instance.option.legend.selected.Calls,false);
 assert.equal(h.instance.option.dataZoom[0].start,30);
 h.context.window.TerminalCharts.mount(h.element,chartOption('pctchg:near:exp:100'));
 assert.equal(h.instance.option.legend.selected.Calls,true);
 assert.notEqual(h.instance.option.dataZoom[0].start,30);
 h.instance.option.legend.selected.Calls=false; h.instance.option.dataZoom=[{start:10,end:40}];
 h.element.controls.children.find(button=>button.textContent==='Reset').onclick();
 assert.equal(h.instance.option.legend.selected.Calls,true);
 assert.notEqual(h.instance.option.dataZoom[0].start,10);
});

test('charts without opt-in view fields preserve their existing legend and zoom behavior',async()=>{
 const h=rendererHarness(); const first={series:[{name:'Legacy',type:'line',data:[[1,2]]}]};
 h.context.window.TerminalCharts.mount(h.element,first); h.show(); await flush();
 h.instance.option.legend.selected.Legacy=false; h.instance.option.dataZoom=[{start:15,end:85}];
 h.context.window.TerminalCharts.mount(h.element,{series:[{name:'Legacy',type:'line',data:[[1,3]]}]});
 assert.equal(h.instance.option.legend.selected.Legacy,false);
 assert.equal(h.instance.option.dataZoom[0].start,15);
});
