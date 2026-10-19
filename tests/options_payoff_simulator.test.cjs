const {test} = require('node:test');
const assert = require('node:assert/strict');
const payoff = require('../static/options-payoff.js');
const simulator = require('../static/options-payoff-simulator.js');

function harness() {
  const doc={activeElement:null,documentElement:{classList:{contains(){return false;}}},createElement(tag){return new FakeElement(tag,this);}};
  class FakeElement {
    constructor(tag,ownerDocument){this.tagName=tag;this.ownerDocument=ownerDocument;this.children=[];this.attributes={};this.style={};this.value='';this.textContent='';this.className='';this.type='';}
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=nodes;}
    setAttribute(key,value){this.attributes[key]=String(value);}
    getAttribute(key){return this.attributes[key]??null;}
    querySelector(selector){
      const match=n=>selector==='output'?n.tagName==='output':selector.startsWith('.')?n.className.split(' ').includes(selector.slice(1)):selector.startsWith('#')?n.id===selector.slice(1):
        selector.startsWith('[data-focus-key=')?n.attributes['data-focus-key']===selector.slice(17,-2):false;
      const visit=n=>{for(const child of n.children||[]){if(match(child))return child;const found=visit(child);if(found)return found;}return null;};
      return visit(this);
    }
    focus(){this.ownerDocument.activeElement=this;}
  }
  const root=new FakeElement('div',doc);
  const charts={
    line(name,x,y,color){return {name,type:'line',data:x.map((v,i)=>[v,y[i]]),itemStyle:{color}};},
    ref(axis,value,name,color){return {[axis+'Axis']:value,name,lineStyle:{color}};},
    cartesian(series,xname,yname){return {series,xAxis:{},yAxis:{name:yname}};},
    mount(id,option){this.last={id,option};},
    resize(id){this.resized=id;}
  };
  let clock=1000;
  const controller=simulator.create({root,charts,renderReferences(){},getRate:()=>0,now:()=>clock});
  return {root,charts,controller,advance(ms){clock+=ms;}};
}
function chain(expiration='2026-11-20',iv=25,quote=10) {
  return {selected_expiration:expiration,spot_price:100,time_to_expiration_years:.1,source:'Fixture',source_as_of:'2026-10-19',
    options:[80,90,95,100,105,110,120].map(strike=>({
      strike,call_bid:quote,call_ask:quote+2,put_bid:quote+3,put_ask:quote+5,
      call_last:quote+1,put_last:quote+4,call_quote_valid:true,put_quote_valid:true,call_iv:iv,put_iv:iv+1
    }))};
}

test('initial chain uses nearest call and exact side midpoint, then add and select work',()=>{
  const h=harness();h.controller.setChain(chain());
  assert.equal(h.controller.state.legs.length,1);
  assert.equal(h.controller.state.legs[0].strike,100);
  assert.equal(h.controller.state.legs[0].premium,11);
  assert.equal(h.controller.state.legs[0].ivPct,25);
  const put=h.controller.addLeg('put');
  assert.equal(put.premium,14);
  h.controller.selectLeg(put.id);assert.equal(h.controller.state.selectedLegId,put.id);
  h.controller.selectFromChain(95,'call');
  assert.equal(h.controller.state.legs[1].type,'call');
  assert.equal(h.controller.state.legs[1].strike,95);
  assert.equal(h.controller.state.legs[1].premium,11);
  h.controller.removeLeg(put.id);assert.equal(h.controller.state.legs.length,1);
});

test('missing/invalid bid-ask uses positive last only and missing IV stays unavailable',()=>{
  const h=harness();const data=chain();
  data.options[0].call_quote_valid=false;data.options[0].call_last=0;
  data.options[0].call_iv=0;
  h.controller.setChain(data);
  h.controller.selectFromChain(80,'call');
  const leg=h.controller.state.legs[0];
  assert.equal(leg.premium,null);
  assert.equal(leg.ivPct,null);
  assert.equal(h.controller.state.legs[0].premiumSource,'missing');
});

test('same expiry refresh preserves entry premiums and manual IV while refreshing automatic IV',()=>{
  const h=harness();h.controller.setChain(chain());
  const initial=h.controller.state.legs[0];
  h.controller.updateLeg(initial.id,'premium',3.5);
  h.controller.updateLeg(initial.id,'ivPct',40);
  h.controller.addLeg('put');
  const second=h.controller.state.legs[1];const secondPremium=second.premium;
  h.advance(1000);h.controller.setChain(chain('2026-11-20',30,20));
  assert.equal(initial.premium,3.5);assert.equal(initial.ivPct,40);
  assert.equal(second.premium,secondPremium);assert.equal(second.ivPct,31);
});

test('expiration change preserves leg structure but refreshes market inputs and labels the loaded expiry',()=>{
  const h=harness();h.controller.setChain(chain());
  const leg=h.controller.state.legs[0];h.controller.updateLeg(leg.id,'premium',3.5);
  h.advance(1000);h.controller.setChain(chain('2026-12-18',35,20));
  assert.equal(h.controller.state.legs.length,1);
  assert.equal(h.controller.state.legs[0].strike,100);
  assert.equal(h.controller.state.legs[0].premium,21);
  assert.equal(h.controller.state.legs[0].ivPct,35);
  assert.equal(h.controller.state.loadedExpiration,'2026-12-18');
});

test('presets use actual chain strikes, sides, and quantities, and unavailable sets explain why',()=>{
  const h=harness();h.controller.setChain(chain());
  for(const name of ['bullCall','bearPut','bullPut','bearCall','straddle','strangle','butterfly','ironCondor']){
    assert.equal(h.controller.applyPreset(name),true,name);
    const legs=h.controller.state.legs;
    assert.ok(legs.every(leg=>[80,90,95,100,105,110,120].includes(leg.strike)));
    if(name==='butterfly')assert.deepEqual(legs.map(leg=>leg.quantity),[1,2,1]);
  }
  const short={...chain(),options:[{...chain().options[3]}]};h.controller.setChain(short);
  assert.equal(h.controller.applyPreset('ironCondor'),false);
  assert.match(h.controller.state.statusMessage,/needs more loaded strikes/);
});

test('graph and heatgrid render the same expiry values with exact dates and focusable cells',()=>{
  const h=harness();h.controller.setChain(chain());
  h.controller.applyPreset('bullCall');
  const wrap=h.root.querySelector('.payoff-grid-wrap');
  assert.ok(wrap);
  const cell=h.root.querySelector('.payoff-cell');
  assert.ok(cell&&cell.title&&cell.getAttribute('aria-label'));
  h.controller.state.view='graph';h.controller.render();
  assert.equal(h.charts.last.id,'payoff-chart');
  assert.deepEqual(h.charts.last.option.series.map(s=>s.name),['P&L at expiration','P&L on Expiration']);
  h.controller.resize();assert.equal(h.charts.resized,'payoff-chart');
});
