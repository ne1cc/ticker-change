(function (root, factory) {
  const api = factory(root.OptionsPayoff);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.OptionsPayoffSimulator = api;
})(typeof window !== 'undefined' ? window : globalThis, function (model) {
  'use strict';
  const DAY = 86400000, YEAR = 365 * DAY;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const labels = {pnl:'Profit / Loss $',pnlPercent:'Profit / Loss %',contractValue:'Contract Value',maxRiskPercent:'% of Max Risk'};
  const presets = [['longCall','Long Call'],['longPut','Long Put'],['bullCall','Bull Call Spread'],['bearPut','Bear Put Spread'],
    ['bullPut','Bull Put Spread'],['bearCall','Bear Call Spread'],['straddle','Long Straddle'],['strangle','Long Strangle'],
    ['butterfly','Long Call Butterfly'],['ironCondor','Short Iron Condor']];
  const sign = side => side === 'long' ? 1 : -1;

  function create({root,charts,renderReferences,getRate,now=()=>Date.now()}) {
    const doc=root.ownerDocument||document;
    let nextId=1;
    const state={chain:null,status:'empty',requestedExpiration:null,loadedExpiration:null,expiryMs:null,receivedAt:null,
      legs:[],selectedLegId:null,view:'table',metric:'pnl',rangePct:10,rangeInitialized:false,ivMultiplier:1,
      selectedTimeMs:null,statusMessage:''};
    function el(tag,cls='',text=null) {const n=doc.createElement(tag);if(cls)n.className=cls;if(text!=null)n.textContent=text;return n;}
    function add(parent,...children){parent.append(...children.filter(Boolean));return parent;}
    function button(text,action,cls='payoff-button'){const n=el('button',cls,text);n.type='button';n.onclick=action;return n;}
    function labeled(text,control){const l=el('label','payoff-label',text);l.htmlFor=control.id;return add(l,control);}
    function select(value,options,action,cls=''){const s=el('select',cls);options.forEach(([name,key])=>{const o=el('option','',name);o.value=key;s.append(o);});s.value=value;s.onchange=action;return s;}
    function observation(strike,type){
      const row=state.chain?.options?.find(item=>Math.abs(Number(item.strike)-strike)<0.01);
      if(!row)return {premium:null,premiumSource:'missing',ivPct:null,ivSource:'missing'};
      const bid=Number(row[type+'_bid']),ask=Number(row[type+'_ask']);let premium=null,premiumSource='missing';
      if(row[type+'_quote_valid']===true&&finite(bid)&&finite(ask)&&bid>=0&&ask>=bid){premium=(bid+ask)/2;premiumSource='mid';}
      else {const last=Number(row[type+'_last']);if(finite(last)&&last>0){premium=last;premiumSource='last';}}
      const iv=Number(row[type+'_iv']);
      return {premium,premiumSource,ivPct:finite(iv)&&iv>0?iv:null,ivSource:finite(iv)&&iv>0?'chain':'missing'};
    }
    function makeLeg(type,strike,side='long',quantity=1){
      const q=observation(strike,type);
      return {id:'leg-'+nextId++,side,type,strike,premium:q.premium,premiumSource:q.premiumSource,quantity,ivPct:q.ivPct,ivSource:q.ivSource};
    }
    function find(id){return state.legs.find(leg=>leg.id===id);}
    function nearestIndex(){
      const rows=state.chain?.options||[];if(!rows.length)return -1;
      const spot=Number(state.chain.spot_price);
      return rows.reduce((best,row,index)=>Math.abs(Number(row.strike)-spot)<Math.abs(Number(rows[best].strike)-spot)?index:best,0);
    }
    function strikeAt(index){const row=state.chain?.options?.[index];return row&&Number(row.strike);}
    function nearestStrike(){return strikeAt(nearestIndex())||null;}
    function initLegs(){const strike=nearestStrike();if(strike){const leg=makeLeg('call',strike);state.legs=[leg];state.selectedLegId=leg.id;}}
    function refreshQuote(leg,keepPremium=false){
      const q=observation(leg.strike,leg.type);
      if(!keepPremium){leg.premium=q.premium;leg.premiumSource=q.premiumSource;}
      leg.ivPct=q.ivPct;leg.ivSource=q.ivSource;
    }
    function setStatus(status,requestedExpiration){state.status=status;if(requestedExpiration)state.requestedExpiration=requestedExpiration;render();}
    function setChain(data,status='ready'){
      if(!data||!Array.isArray(data.options)){setStatus(status==='ready'?'empty':status,state.requestedExpiration);return;}
      const expiration=data.selected_expiration||null,changed=state.loadedExpiration!=null&&expiration!==state.loadedExpiration,received=now();
      state.chain=data;state.status=data.options.length?'ready':'empty';
      if(!state.legs.length&&data.options.length)initLegs();
      if(changed)state.legs.forEach(leg=>refreshQuote(leg));
      else if(state.loadedExpiration!=null)state.legs.forEach(leg=>{if(leg.ivSource==='chain')refreshQuote(leg,true);});
      state.loadedExpiration=expiration;state.requestedExpiration=null;state.receivedAt=received;
      const years=Number(data.time_to_expiration_years);
      state.expiryMs=finite(years)&&years>=0?received+years*YEAR:null;
      if(!state.rangeInitialized){
        const iv=state.legs.map(leg=>leg.ivPct).find(value=>finite(value)&&value>0);
        state.rangePct=finite(iv)&&years>0?Math.max(2,Math.min(50,iv*Math.sqrt(years))):10;
        state.rangeInitialized=true;
      }
      render();
    }
    function addLeg(type){const strike=nearestStrike();if(!strike)return null;const leg=makeLeg(type,strike);state.legs.push(leg);state.selectedLegId=leg.id;render();return leg;}
    function selectLeg(id){if(find(id)){state.selectedLegId=id;render();}}
    function updateLeg(id,field,value){
      const leg=find(id);if(!leg)return;
      if(field==='strike'&&finite(value)&&value>0&&value!==leg.strike){leg.strike=value;refreshQuote(leg);}
      else if(field==='type'&&['call','put'].includes(value)&&value!==leg.type){leg.type=value;refreshQuote(leg);}
      else if(field==='premium'){leg.premium=finite(value)&&value>=0?value:null;leg.premiumSource='custom';}
      else if(field==='quantity')leg.quantity=Number.isInteger(value)&&value>0?value:null;
      else if(field==='ivPct'){leg.ivPct=finite(value)&&value>=0?value:null;leg.ivSource='custom';}
      else if(field==='side'&&['long','short'].includes(value))leg.side=value;
      render();
    }
    function removeLeg(id){state.legs=state.legs.filter(leg=>leg.id!==id);if(state.selectedLegId===id)state.selectedLegId=state.legs[0]?.id||null;render();}
    function selectFromChain(strike,type){
      let leg=find(state.selectedLegId);
      if(!leg){leg=makeLeg(type,Number(strike));state.legs.push(leg);state.selectedLegId=leg.id;}
      else{leg.type=type;leg.strike=Number(strike);refreshQuote(leg);}
      render();
    }
    function butterflyStrikes(){
      const values=(state.chain?.options||[]).map(row=>Number(row.strike)).filter(v=>finite(v)&&v>0).sort((a,b)=>a-b);
      const spot=Number(state.chain?.spot_price);let found=null,best=Infinity;
      for(let i=0;i<values.length-2;i++)for(let j=i+1;j<values.length-1;j++){
        const width=values[j]-values[i],k=values.findIndex((v,index)=>index>j&&Math.abs((v-values[j])-width)<0.01);
        if(k>=0&&Math.abs(values[j]-spot)<best){best=Math.abs(values[j]-spot);found=[values[i],values[j],values[k]];}
      }return found;
    }
    function applyPreset(name){
      const center=nearestIndex();if(center<0)return false;
      const at=offset=>strikeAt(center+offset);let legs=null;
      if(name==='longCall')legs=[makeLeg('call',at(0))];
      if(name==='longPut')legs=[makeLeg('put',at(0))];
      if(name==='bullCall'&&at(1))legs=[makeLeg('call',at(0)),makeLeg('call',at(1),'short')];
      if(name==='bearPut'&&at(-1))legs=[makeLeg('put',at(0)),makeLeg('put',at(-1),'short')];
      if(name==='bullPut'&&at(-1))legs=[makeLeg('put',at(0),'short'),makeLeg('put',at(-1))];
      if(name==='bearCall'&&at(1))legs=[makeLeg('call',at(0),'short'),makeLeg('call',at(1))];
      if(name==='straddle')legs=[makeLeg('call',at(0)),makeLeg('put',at(0))];
      if(name==='strangle'&&at(-1)&&at(1))legs=[makeLeg('put',at(-1)),makeLeg('call',at(1))];
      if(name==='butterfly'){const triple=butterflyStrikes();if(triple)legs=[makeLeg('call',triple[0]),makeLeg('call',triple[1],'short',2),makeLeg('call',triple[2])];}
      if(name==='ironCondor'&&center>=2&&at(1))legs=[makeLeg('put',at(-2)),makeLeg('put',at(-1),'short'),makeLeg('call',at(0),'short'),makeLeg('call',at(1))];
      if(!legs){state.statusMessage='This preset needs more loaded strikes around the current price.';render();return false;}
      state.legs=legs;state.selectedLegId=legs[0].id;state.statusMessage='';render();return true;
    }
    function money(value,positive=false){
      if(value===Infinity||value===-Infinity)return 'Unlimited';if(!finite(value))return 'N/A';
      const result='$'+new Intl.NumberFormat('en-US',{maximumFractionDigits:2}).format(Math.abs(value));
      return value<0?'−'+result:positive&&value>0?'+'+result:result;
    }
    function risk(value,loss=false){return value===Infinity?'Unlimited':money(loss?-value:value);}
    function legErrors(leg,tYears){return model.validateLeg(leg,{requireIv:tYears>0});}
    function analysis(){return state.status!=='empty'&&state.chain?.options?.length&&state.legs.length&&!state.legs.some(leg=>legErrors(leg,0).length)?model.analyzeExpiry(state.legs):null;}
    function yearsAt(timestamp){return finite(state.expiryMs)&&finite(timestamp)?Math.max(0,(state.expiryMs-timestamp)/YEAR):null;}
    function rate(){const value=Number(getRate());return finite(value)?value:null;}
    function position(spot,timestamp){
      const t=yearsAt(timestamp),r=rate();return t==null||r==null?null:model.valuePosition(state.legs,{spot,tYears:t,rate:r,ivMultiplier:state.ivMultiplier});
    }
    function prices(){
      const grid=model.scenarioPrices(Number(state.chain?.spot_price),state.rangePct);
      const all=[...grid,...state.legs.map(leg=>leg.strike)].filter(finite).sort((a,b)=>b-a);
      return all.filter((value,index)=>index===0||Math.abs(value-all[index-1])>1e-8);
    }
    function times(){
      if(state.expiryMs==null)return [];
      const samples=model.scenarioTimes(state.receivedAt,state.expiryMs);
      return samples.map((timestamp,index)=>({
        timestamp,label:state.expiryMs<=state.receivedAt?'Expiration':index===0?'Now':index===samples.length-1?'Expiration':
          new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',month:'short',day:'numeric'}).format(new Date(timestamp))
      }));
    }
    function chosenTime(list){
      if(!list.length)return null;const target=state.selectedTimeMs??state.expiryMs;
      return list.reduce((best,item)=>Math.abs(item.timestamp-target)<Math.abs(best.timestamp-target)?item:best,list[0]);
    }
    function metricValue(value,a){return model.metricValue(value,state.metric,a);}
    function metricText(value){
      if(!finite(value))return 'N/A';
      if(state.metric==='pnlPercent'||state.metric==='maxRiskPercent')return (value>0?'+':'')+value.toFixed(1)+'%';
      return money(value,value>0);
    }
    function renderLegs(container,tYears){
      const table=el('table','payoff-legs-table'),thead=el('thead'),head=el('tr');table.id='payoff-legs';
      ['Side','Type','Strike','Premium / share','Contracts','IV %',''].forEach(text=>head.append(el('th','',text)));
      add(thead,head);table.append(thead);const body=el('tbody');
      state.legs.forEach(leg=>{
        const row=el('tr',leg.id===state.selectedLegId?'payoff-leg-selected':'');
        const side=select(leg.side,[['Buy','long'],['Sell','short']],event=>updateLeg(leg.id,'side',event.target.value),'payoff-input');
        const type=select(leg.type,[['Call','call'],['Put','put']],event=>updateLeg(leg.id,'type',event.target.value),'payoff-input');
        side.setAttribute('aria-label','Position side');type.setAttribute('aria-label','Option type');
        const inputs=[];
        [['strike','Strike price','0.01'],['premium','Premium per share','0.01'],['quantity','Contracts','1'],['ivPct','Implied volatility','0.1']].forEach(([field,label,step])=>{
          const input=el('input','payoff-input');input.type='number';input.step=step;input.min=field==='quantity'?'1':'0';input.value=leg[field]==null?'':String(leg[field]);
          input.setAttribute('aria-label',label);input.setAttribute('data-focus-key',leg.id+'-'+field);
          input.onchange=()=>updateLeg(leg.id,field,input.value===''?null:Number(input.value));inputs.push(input);
        });
        [side,type,...inputs].forEach((input,index)=>{
          const cell=el('td');cell.append(input);
          if(index===3)cell.append(el('small','payoff-source',leg.premiumSource==='missing'?'Missing quote':leg.premiumSource==='last'?'Last':leg.premiumSource==='custom'?'Manual':'Mid'));
          if(index===5&&leg.ivPct==null)cell.append(el('small','payoff-source','IV unavailable'));
          row.append(cell);
        });
        const actions=el('td');actions.append(button(leg.id===state.selectedLegId?'Selected':'Select',()=>selectLeg(leg.id),'payoff-button payoff-select-leg'));
        actions.append(button('Remove',()=>removeLeg(leg.id),'payoff-button payoff-remove-leg'));row.append(actions);body.append(row);
      });
      table.append(body);container.append(table);
      if(!state.legs.length)container.append(el('p','payoff-empty','Add a call or put to begin a strategy.'));
      if(state.legs.some(leg=>legErrors(leg,tYears).length))container.append(el('p','payoff-error','Complete each leg’s strike, premium, contracts, and volatility before viewing modeled P&L.'));
    }
    function renderSummary(container,a){
      const row=el('div','payoff-summary');const breakeven=a?[
        ...a.breakevens.map(v=>'$'+v.toFixed(2)),
        ...a.breakevenRanges.map(([lo,hi])=>'$'+lo.toFixed(2)+'–'+(hi===Infinity?'∞':'$'+hi.toFixed(2)))
      ].join(' / ')||'None':'N/A';
      [['Net Debit / Credit',a?money(a.netDebit):'N/A'],['Max Profit',a?risk(a.maxProfit):'N/A'],
        ['Max Loss',a?risk(a.maxLoss,true):'N/A'],['Breakeven',breakeven]].forEach(([name,value])=>{
        const card=el('div','payoff-stat');card.append(el('span','payoff-stat-label',name),el('strong','payoff-stat-value',value));row.append(card);
      });container.append(row);
    }
    function color(value,max){
      if(!finite(value)||value===0)return null;
      const alpha=Math.min(.82,.12+.7*Math.abs(value)/(max||1));
      const rgb=value>0?[16,185,129]:[225,29,72],base=doc.documentElement.classList.contains('dark')?[24,24,27]:[255,255,255];
      const luminance=rgb.map((channel,index)=>{
        const mixed=(channel*alpha+base[index]*(1-alpha))/255;
        const linear=mixed<=.04045?mixed/12.92:((mixed+.055)/1.055)**2.4;
        return linear*[.2126,.7152,.0722][index];
      }).reduce((sum,value)=>sum+value,0);
      return {background:'rgba('+rgb.join(',')+','+alpha+')',foreground:luminance>.179?'#18181b':'#fff'};
    }
    function renderGrid(container,a){
      const dateList=times(),priceList=prices();
      if(!dateList.length||!priceList.length){container.append(el('p','payoff-empty','Scenario dates are unavailable for this expiration.'));return;}
      const matrix=priceList.map(price=>dateList.map(date=>position(price,date.timestamp)));
      const max=Math.max(1,...matrix.flat().map(value=>value?.pnl).filter(finite).map(Math.abs));
      const wrap=el('div','payoff-grid-wrap'),table=el('table','payoff-grid-table');table.setAttribute('role','grid');table.setAttribute('aria-label','Strategy P&L by underlying price and date');
      const head=el('thead'),hrow=el('tr');hrow.append(el('th','payoff-price-sticky','Underlying'));
      dateList.forEach(date=>hrow.append(el('th','',date.label)));head.append(hrow);table.append(head);const body=el('tbody');
      priceList.forEach((price,rowIndex)=>{
        const row=el('tr'),move=(price/Number(state.chain.spot_price)-1)*100;
        row.append(el('th','payoff-price-sticky','$'+price.toFixed(2)+' ('+(move>0?'+':'')+move.toFixed(1)+'%)'));
        dateList.forEach((date,colIndex)=>{
          const cell=el('td','payoff-cell'),pos=matrix[rowIndex][colIndex],shown=pos?metricValue(pos,a):null;
          cell.textContent=metricText(shown);
          if(pos){
            const shade=color(pos.pnl,max);if(shade){cell.style.backgroundColor=shade.background;cell.style.color=shade.foreground;}
            cell.className+=' '+(pos.pnl>0?'payoff-positive':pos.pnl<0?'payoff-negative':'payoff-zero');
            const details=state.legs.map((leg,index)=>{
              const contribution=(pos.legValues[index]-sign(leg.side)*leg.premium*leg.quantity*100);
              return leg.side+' '+leg.type+' '+leg.strike+' · '+money(contribution,true);
            }).join('\n');
            cell.title=date.label+' · underlying $'+price.toFixed(2)+' ('+move.toFixed(1)+'%) · '+money(pos.pnl,true)+' P&L\n'+details;
            cell.tabIndex=0;cell.setAttribute('aria-label',date.label+', underlying $'+price.toFixed(2)+', '+money(pos.pnl,true)+' profit or loss. '+details.replaceAll('\n',', '));
          }else{cell.title='Unavailable: complete all premiums and required volatility values.';cell.setAttribute('aria-label',cell.title);}
          row.append(cell);
        });body.append(row);
      });
      table.id='payoff-grid';table.append(body);wrap.append(table);container.append(wrap,el('p','payoff-legend','Green: profit · Red: loss · Neutral: $0 · Values show '+(labels[state.metric]||labels.pnl)));
    }
    function renderChart(container,a){
      const dateList=times(),choice=chosenTime(dateList);
      const dateSelect=select(String(Math.max(0,dateList.indexOf(choice))),dateList.map((item,index)=>[item.label+' · '+new Date(item.timestamp).toLocaleDateString('en-US',{timeZone:'America/New_York'}),String(index)]),event=>{
        state.selectedTimeMs=dateList[Number(event.target.value)]?.timestamp;render();
      },'payoff-input');
      container.append(labeled('Modeled date',dateSelect));
      const chart=el('div','options-visualizer-chart payoff-chart');chart.id='payoff-chart';container.append(chart);
      const ps=prices();
      if(!a||!ps.length||!choice){container.append(el('p','payoff-empty','Complete all leg inputs to plot the strategy.'));return;}
      const x=Array.from({length:121},(_,i)=>ps.at(-1)+(ps[0]-ps.at(-1))*i/120);
      a.knots.concat(a.breakevens).forEach(value=>{if(value>=ps.at(-1)&&value<=ps[0])x.push(value);});
      x.sort((u,v)=>u-v);const xs=[...new Set(x)];
      const expiry=xs.map(value=>model.expiryPnl(state.legs,value));
      const modeled=xs.map(value=>position(value,choice.timestamp)?.pnl??null);
      const series=[charts.line('P&L at expiration',xs,expiry,'#6366f1'),charts.line('P&L on '+choice.label,xs,modeled,'#10b981')];
      const refs=[charts.ref('y',0,'$0 P&L','#71717a'),charts.ref('x',Number(state.chain.spot_price),'Current price')];
      state.legs.forEach(leg=>refs.push(charts.ref('x',leg.strike,(leg.type==='call'?'Call':'Put')+' strike','#f59e0b')));
      a.breakevens.forEach(price=>refs.push(charts.ref('x',price,'Breakeven price','#10b981')));
      renderReferences('payoff-references',refs);series.forEach(item=>item.markLine={symbol:'none',data:refs.filter(ref=>ref.xAxis==null||ref.xAxis>=0)});
      const option=charts.cartesian(series,'Underlying price ($)','Total P&L ($)');
      const finiteRoots=a.breakevens.filter(finite);
      option.xAxis={...option.xAxis,min:Math.max(0,Math.min(ps.at(-1),...finiteRoots)),max:Math.max(ps[0],...finiteRoots),scale:true};
      option.yAxis.axisLabel={fontSize:12,hideOverlap:true};option.yAxis.nameGap=55;charts.mount('payoff-chart',option);
    }
    function render(){
      const active=doc.activeElement,key=active?.getAttribute?.('data-focus-key'),start=active?.selectionStart,end=active?.selectionEnd;
      root.replaceChildren();const shell=el('div','payoff-shell'),heading=el('div','payoff-heading');
      heading.append(el('h3','', 'Options Payoff Simulator'),el('p','payoff-assumptions','Pre-expiration P&L uses Black–Scholes with loaded implied volatility. Expiration uses intrinsic value. Estimates exclude dividends, early assignment, commissions, and slippage.'));
      const context=[state.chain?.selected_expiration?'Expiration '+state.chain.selected_expiration:'Waiting for option chain',state.chain?.source,state.chain?.source_as_of].filter(Boolean).join(' · ');
      heading.append(el('p','payoff-source',context));shell.append(heading);
      if(state.status==='loading')shell.append(el('p','payoff-status','Loading '+(state.requestedExpiration||state.loadedExpiration||'selected expiration')+'…'));
      if(state.status==='error')shell.append(el('p','payoff-status payoff-error','Chain refresh failed. Showing last loaded positions and prices.'));
      if(state.status==='empty')shell.append(el('p','payoff-status','No option strikes are loaded for this expiration.'));
      if(state.statusMessage)shell.append(el('p','payoff-error',state.statusMessage));
      const actions=el('div','payoff-actions'),preset=select('',[['Load strategy preset',''],...presets.map(([key,name])=>[name,key])],event=>{if(event.target.value)applyPreset(event.target.value);},'payoff-input');
      actions.append(labeled('Strategy',preset),button('Add Call',()=>addLeg('call')),button('Add Put',()=>addLeg('put')));shell.append(actions);
      const editor=el('div','payoff-leg-editor');renderLegs(editor,Number(state.chain?.time_to_expiration_years)||0);shell.append(editor);
      const a=analysis();renderSummary(shell,a);
      const controls=el('div','payoff-view-controls');
      [['table','Table'],['graph','Graph']].forEach(([key,name])=>{const b=button(name,()=>{state.view=key;render();});b.setAttribute('aria-pressed',String(state.view===key));controls.append(b);});
      const metric=select(state.metric,Object.entries(labels).map(([key,name])=>[name,key]),event=>{state.metric=event.target.value;render();},'payoff-input');
      controls.append(labeled('Display',metric));shell.append(controls);
      const settings=el('div','payoff-settings');
      [['rangePct','Underlying price range (%)',1,100,1],['ivMultiplier','IV multiplier',0,3,.1]].forEach(([field,name,min,max,step])=>{
        const input=el('input','payoff-input');input.type='range';input.min=String(min);input.max=String(max);input.step=String(step);input.value=String(state[field]);input.id='payoff-'+field;
        input.setAttribute('data-focus-key','slider-'+field);
        input.oninput=()=>{state[field]=Number(input.value);const value=shell.querySelector('#payoff-'+field+'-value');if(value)value.textContent=field==='ivMultiplier'?state.ivMultiplier.toFixed(1)+'×':String(state.rangePct)+'%';};
        input.onchange=render;
        const wrap=el('div','payoff-control');const label=el('label','payoff-label',name);label.htmlFor=input.id;
        wrap.append(label,input,el('output','payoff-slider-value',field==='ivMultiplier'?state.ivMultiplier.toFixed(1)+'×':String(state.rangePct)+'%'));
        const output=wrap.querySelector('output');output.id='payoff-'+field+'-value';settings.append(wrap);
      });
      settings.append(button('Reset IV to 1×',()=>{state.ivMultiplier=1;render();}));shell.append(settings);
      const refs=el('div','payoff-references');refs.id='payoff-references';const output=el('div','payoff-output');output.append(refs);
      shell.append(output);root.append(shell);
      if(state.view==='table')renderGrid(output,a);else renderChart(output,a);
      if(key){const target=root.querySelector('[data-focus-key="'+key+'"]');target?.focus?.();if(Number.isInteger(start)&&Number.isInteger(end))target?.setSelectionRange?.(start,end);}
    }
    function resize(){if(state.view==='graph')charts.resize?.('payoff-chart');}
    return {state,setStatus,setChain,addLeg,selectLeg,selectFromChain,updateLeg,removeLeg,applyPreset,render,resize};
  }
  return {create};
});
