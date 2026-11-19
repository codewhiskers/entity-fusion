// Minimal DOM harness for the offline dashboard; no npm/browser dependency.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[2], 'utf8');
class Element {
  constructor(tag='div') {this.tagName=tag;this.children=[];this.listeners={};this.value='';this._text='';this.attrs={};this.style={};}
  set textContent(v){this._text=String(v);this.children=[];}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join(' ');}
  append(...elements){this.children.push(...elements);}
  replaceChildren(...elements){this.children=elements;this._text='';}
  addEventListener(event,callback){this.listeners[event]=callback;}
  setAttribute(key,value){this.attrs[key]=String(value);}
  setPointerCapture(){}
  getBoundingClientRect(){return {left:0,top:0,width:1100,height:540};}
}
const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
elements.get('review-data').textContent=html.match(/id="review-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
elements.get('review-filter').value='all';
const document={getElementById:id=>elements.get(id),createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');
const frames=new Map();let nextFrame=0;
const requestAnimationFrame=fn=>{frames.set(++nextFrame,fn);return nextFrame;};
const cancelAnimationFrame=id=>frames.delete(id);
const assertions=`
assert.equal(state.selected,'0');
assert.equal(matchGraph.model.nodes.filter(n=>n.kind==='record').length,5);
assert.equal(matchGraph.model.nodes.filter(n=>n.kind==='group').length,1);
assert.equal(matchGraph.model.links.length,5);
assert.equal(matchGraph.model.truncated,false);
const originalLabels=matchGraph.labelsAll;
$('graph-labels').listeners.click();assert.equal(matchGraph.labelsAll,!originalLabels);
$('graph-labels').listeners.click();assert.equal(matchGraph.labelsAll,originalLabels);
selectGraphNode(matchGraph.model.nodes.find(n=>n.kind==='group'));
assert.equal(state.group,data.groups[0].group_id);
assert.match($('evidence-focus').textContent,/Showing match group/);
selectGraphNode(matchGraph.model.nodes.find(n=>n.kind==='record'));
assert.equal(state.record,'A');
assert.equal(state.group,null);
$('clear-record').listeners.click();
for(let i=0;i<8&&frames.size;i++){const [id,fn]=frames.entries().next().value;frames.delete(id);fn();}
assert.ok($('match-graph').children.length>0);
const large=buildMatchGraph(Array.from({length:200},(_,i)=>({record_id:String(i),name:'Name '+i,proposed_cluster_id:'name:0:0'})),[{group_id:'large',members:Array.from({length:200},(_,i)=>String(i)),shared_values:{key:'same'},conditions:['key']}]);
assert.equal(large.recordCount,150);assert.equal(large.links.length,150);assert.equal(large.truncated,true);
const manyGroups=buildMatchGraph(data.records,Array.from({length:120},(_,i)=>({group_id:String(i),members:['A','B'],shared_values:{key:'same'},conditions:['key']})));
assert.equal(manyGroups.groupCount,100);assert.equal(manyGroups.links.length,200);assert.equal(manyGroups.truncated,true);
assert.match($('detail-title').textContent,/Cluster 0/);
assert.match($('records-table').textContent,/NORTH HARBOR/);
assert.match($('evidence-table').textContent,/Shared: x/);
assert.match($('records-table').textContent,/Source city/);
$('path-from').value='A';$('path-to').value='D';explainPath();
assert.match($('path-output').textContent,/Shared values: Shared = x/);
assert.ok(state.path.nodes.has('r:A'));assert.ok(state.path.nodes.has('r:D'));
assert.equal(state.path.links.size,2);
state.record='A';renderDetails();
assert.match($('evidence-focus').textContent,/A/);
$('cluster-filter').value='5';renderClusters();
assert.equal(state.selected,'5');
assert.match($('records-table').textContent,/F/);
assert.doesNotMatch($('records-table').textContent,/DESERT VALLEY/);
assert.match($('evidence-table').textContent,/No matching rows/);
$('cluster-filter').value='does-not-exist';renderClusters();
assert.equal(state.selected,null);
assert.match($('detail-title').textContent,/No cluster selected/);
$('reset-filters').listeners.click();
assert.equal(state.selected,'0');
$('record-search').value='DESERT';renderDetails();
assert.match($('records-table').textContent,/DESERT/);
assert.doesNotMatch($('records-table').textContent,/NORTH HARBOR/);
$('record-search').value='';$('cluster-search').value='DESERT';renderClusters();
assert.equal(state.selected,'0');
console.log('Dashboard selection, filtering, source columns, exact evidence, and connection paths passed.');
`;
vm.runInNewContext(scripts+'\n'+assertions,{document,assert,console,requestAnimationFrame,cancelAnimationFrame,frames});
