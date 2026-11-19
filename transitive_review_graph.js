// Embedded in each offline report; no CDN or browser extension is needed.
const GRAPH_LIMITS = {records:150, groups:100, links:1500};
function buildMatchGraph(records,groups){
 const recordIds=new Set(records.slice(0,GRAPH_LIMITS.records).map(r=>r.record_id));
 const nodes=records.slice(0,GRAPH_LIMITS.records).map(r=>({id:'r:'+r.record_id,kind:'record',record:r,
   label:r.name||'Name unavailable',title:recordLabel(r.record_id)+'\nProposed '+r.proposed_cluster_id}));
 const links=[];let visibleGroups=0;
 for(const g of groups){
  const members=g.members.filter(id=>recordIds.has(id));
  if(!members.length||visibleGroups>=GRAPH_LIMITS.groups||links.length+members.length>GRAPH_LIMITS.links)continue;
  const values=Object.entries(g.shared_values).map(([k,v])=>readable(k)+': '+displayValue(v));
  const label=g.conditions.map(c=>c.split(' [block_on:')[0]).map(field=>readable(field)+': '+displayValue(g.shared_values[field])).join(' + ');
  nodes.push({id:'g:'+g.group_id,kind:'group',group:g,label,
    title:'Match group '+g.group_id+'\n'+g.conditions.map(conditionLabel).join(' + ')+'\n'+values.join('\n')+'\n'+members.length+' of '+g.members.length+' members shown'});
  for(const id of members)links.push({source:'r:'+id,target:'g:'+g.group_id,id:'r:'+id+'|g:'+g.group_id});
  visibleGroups++;
 }
 return {nodes,links,recordCount:recordIds.size,groupCount:visibleGroups,
  truncated:recordIds.size<records.length||visibleGroups<groups.length};
}
function selectGraphNode(node){
 state.path=null;
 if(node.kind==='record'){state.record=state.record===node.record.record_id?null:node.record.record_id;state.group=null;}
 else {state.group=state.group===node.group.group_id?null:node.group.group_id;state.record=null;}
 for(const key of ['records-table','evidence-table','pairs-table'])if(state.tables[key])state.tables[key].page=0;
 renderDetails();
}
let matchGraph=null;
function renderMatchGraph(records,groups){
 const host=$('match-graph');
 if(!data.has_exact_evidence||state.selected===null){if(matchGraph)matchGraph.stop();matchGraph=null;host.replaceChildren();$('graph-status').textContent=data.has_exact_evidence?'Select a cluster to display its connections.':'Exact match evidence is needed to draw this graph.';$('graph-legend').replaceChildren();return;}
 if(!matchGraph||matchGraph.cluster!==state.selected){
  if(matchGraph)matchGraph.stop();
  const model=buildMatchGraph(records,groups);
  matchGraph=createMatchGraph(host,model);matchGraph.cluster=state.selected;matchGraph.model=model;
  $('graph-pause').textContent='Pause layout';
  const legend=$('graph-legend');legend.replaceChildren();
  const nameGroups=[...new Set(records.map(r=>r.proposed_cluster_id))].sort();
  $('graph-labels').textContent=matchGraph.labelsAll?'Hide record labels':'Show record labels';
  matchGraph.colors=new Map(nameGroups.map((id,i)=>[id,`hsl(${(i*137.508+205)%360} 54% 43%)`]));
  nameGroups.slice(0,8).forEach(id=>{const chip=make('span',undefined,'graph-chip'),swatch=make('span',undefined,'graph-swatch');swatch.style.background=matchGraph.colors.get(id);chip.append(swatch,make('span','Name group '+(Number(id.split(':').pop())+1)));legend.append(chip);});
  if(nameGroups.length>8)legend.append(make('span','+'+(nameGroups.length-8)+' more name groups (hover records for details)','muted'));
  legend.append(make('span','◇ Shared-value match group','graph-chip'));
 }
 const model=matchGraph.model;
 let note=`${model.recordCount} of ${records.length} records · ${model.groupCount} of ${groups.length} match groups · ${model.links.length} membership links.`;
 if(model.truncated)note+=' Limited preview: omitted nodes/links do not imply disconnection. Tables and path lookup retain full evidence.';
 if(state.record&&!model.nodes.some(n=>n.id==='r:'+state.record))note+=' Selected record is outside this graph preview.';
 if(state.path&&[...state.path.nodes].some(id=>!model.nodes.some(n=>n.id===id)))note+=' Part of the highlighted path is outside this preview; see the complete path below.';
 $('graph-status').textContent=note;
 matchGraph.highlight();
}
function createMatchGraph(host,model){
 host.replaceChildren();const ns='http://www.w3.org/2000/svg';
 function svgElement(tag,attrs={}){const el=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);return el;}
 const svg=svgElement('svg',{viewBox:'0 0 1100 540',role:'group','aria-label':'Records and exact shared-value match groups'}),viewport=svgElement('g');svg.append(viewport);host.append(svg);
 const nodes=model.nodes.map((n,i)=>({...n,x:550+250*Math.cos(i*2.39996)*Math.sqrt((i+.5)/model.nodes.length),y:270+190*Math.sin(i*2.39996)*Math.sqrt((i+.5)/model.nodes.length),vx:0,vy:0}));
 const byNode=new Map(nodes.map(n=>[n.id,n])),links=model.links.map(l=>({...l,a:byNode.get(l.source),b:byNode.get(l.target)}));
 const lines=links.map(l=>{const el=svgElement('line',{'class':'match-edge'});viewport.append(el);return {l,el};});
 let camera={x:0,y:0,k:1},frame=null,drag=null,paused=false,iteration=0;
 const elements=nodes.map(n=>{
  const el=svgElement('g',{'class':'match-node '+n.kind+'-node',tabindex:0,role:'button','aria-label':n.title});
  const title=svgElement('title');title.textContent=n.title;el.append(title);
  const shape=n.kind==='record'?svgElement('circle',{r:10}):svgElement('path',{d:'M 0 -13 L 13 0 L 0 13 L -13 0 Z'});
  const label=svgElement('text',{x:17,y:4});label.textContent=n.label.length>35?n.label.slice(0,32)+'…':n.label;el.append(shape,label);viewport.append(el);
  el.addEventListener('click',()=>{if(!el.suppressClick)selectGraphNode(n);el.suppressClick=false;});
  el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectGraphNode(n);}});
  el.addEventListener('pointerdown',e=>{e.stopPropagation();drag={node:n,startX:e.clientX,startY:e.clientY,element:el};el.setPointerCapture(e.pointerId);});
  return {n,el,shape};
 });
 function transform(){viewport.setAttribute('transform',`translate(${camera.x} ${camera.y}) scale(${camera.k})`);}
 function position(){for(const {l,el}of lines){el.setAttribute('x1',l.a.x);el.setAttribute('y1',l.a.y);el.setAttribute('x2',l.b.x);el.setAttribute('y2',l.b.y);}for(const {n,el}of elements)el.setAttribute('transform',`translate(${n.x} ${n.y})`);}
 function tick(){
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=nodes[i],b=nodes[j];let dx=a.x-b.x,dy=a.y-b.y;const d2=Math.max(100,dx*dx+dy*dy),force=25/d2;dx*=force;dy*=force;a.vx+=dx;a.vy+=dy;b.vx-=dx;b.vy-=dy;}
  for(const l of links){const dx=l.b.x-l.a.x,dy=l.b.y-l.a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-110)*.012;l.a.vx+=dx/d*f;l.a.vy+=dy/d*f;l.b.vx-=dx/d*f;l.b.vy-=dy/d*f;}
  for(const n of nodes){if(drag&&drag.node===n){n.vx=n.vy=0;continue;}n.vx=(n.vx+(550-n.x)*.012)*.78;n.vy=(n.vy+(270-n.y)*.035)*.78;n.x=Math.max(35,Math.min(1040,n.x+n.vx));n.y=Math.max(30,Math.min(510,n.y+n.vy));}
  position();iteration++;if(!paused&&iteration<260)frame=requestAnimationFrame(tick);
 }
 function start(){if(frame!==null)cancelAnimationFrame(frame);iteration=0;paused=false;frame=requestAnimationFrame(tick);}
 function point(e){if(svg.createSVGPoint&&svg.getScreenCTM()){const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());}const rect=svg.getBoundingClientRect();return {x:(e.clientX-rect.left)*1100/rect.width,y:(e.clientY-rect.top)*540/rect.height};}
 svg.addEventListener('pointerdown',e=>{drag={start:point(e),cameraX:camera.x,cameraY:camera.y};svg.setPointerCapture(e.pointerId);});
 svg.addEventListener('pointermove',e=>{if(!drag)return;const p=point(e);if(drag.node){drag.node.x=(p.x-camera.x)/camera.k;drag.node.y=(p.y-camera.y)/camera.k;drag.node.vx=drag.node.vy=0;if(Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>4)drag.element.suppressClick=true;position();}else{camera.x=drag.cameraX+p.x-drag.start.x;camera.y=drag.cameraY+p.y-drag.start.y;transform();}});
 function endDrag(){drag=null;}svg.addEventListener('pointerup',endDrag);svg.addEventListener('pointercancel',endDrag);
 function zoom(factor,center={x:550,y:270}){const k=Math.max(.3,Math.min(4,camera.k*factor));camera.x=center.x-(center.x-camera.x)*k/camera.k;camera.y=center.y-(center.y-camera.y)*k/camera.k;camera.k=k;transform();}
 svg.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY<0?1.12:1/1.12,point(e));},{passive:false});
 position();start();
 return {labelsAll:nodes.length<=40,stop(){paused=true;if(frame!==null)cancelAnimationFrame(frame);},zoom,reset(){camera={x:0,y:0,k:1};transform();},toggle(){paused=!paused;if(!paused)start();return paused;},
  highlight(){
   svg.setAttribute('class',matchGraph.labelsAll?'':'compact-labels');
   const focusNodes=new Set(),focusLinks=new Set();
   if(state.path){for(const id of state.path.nodes)focusNodes.add(id);for(const id of state.path.links)focusLinks.add(id);}
   else if(state.group){const g=groupsForCurrentCluster().find(g=>g.group_id===state.group);if(g){focusNodes.add('g:'+g.group_id);for(const id of g.members){focusNodes.add('r:'+id);focusLinks.add('r:'+id+'|g:'+g.group_id);}}}
   else if(state.record){focusNodes.add('r:'+state.record);for(const g of groupsForCurrentCluster().filter(g=>g.members.includes(state.record))){focusNodes.add('g:'+g.group_id);for(const id of g.members){focusNodes.add('r:'+id);focusLinks.add('r:'+id+'|g:'+g.group_id);}}}
   for(const {n,el,shape}of elements){el.setAttribute('class','match-node '+n.kind+'-node'+(n.id==='r:'+state.record||(state.path&&state.path.nodes.has(n.id))?' picked':''));shape.setAttribute('fill',n.kind==='record'?matchGraph.colors.get(n.record.proposed_cluster_id):'#f4f0e6');shape.setAttribute('stroke',n.kind==='record'?'#fff':'#9e8354');el.setAttribute('opacity',focusNodes.size&&!focusNodes.has(n.id)?'.16':'1');shape.setAttribute('stroke-width',focusNodes.has(n.id)?'3':'1.5');}
   for(const {l,el}of lines){el.setAttribute('stroke',focusLinks.has(l.id)?'#d17b24':'#bac9d8');el.setAttribute('stroke-width',focusLinks.has(l.id)?'2.5':'1');el.setAttribute('opacity',focusNodes.size&&!focusLinks.has(l.id)?'.12':'.85');}
  }};
}
function groupsForCurrentCluster(){return grouped.groups.get(state.selected)||[];}
