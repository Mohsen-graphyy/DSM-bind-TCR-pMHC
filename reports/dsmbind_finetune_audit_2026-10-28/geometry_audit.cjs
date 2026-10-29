// Read-only: reconstruct the model's 14-Angstrom atom-pair mask from prepared test coordinates.
// Run from the workspace root with Node 22+: node reports/dsmbind_finetune_audit_2026-10-28/geometry_audit.cjs
const fs=require('node:fs'), zlib=require('node:zlib');
const {DatabaseSync}=require('node:sqlite');
function csv(name){const lines=fs.readFileSync('C:/Users/hajmo/Downloads/'+name,'utf8').trim().split(/\r?\n/);const h=lines[0].split(',');return lines.slice(1).map(l=>Object.fromEntries(l.split(',').map((v,i)=>[h[i],v])));}
function stats(a){const sorted=[...a].sort((a,b)=>a-b),n=a.length,m=a.reduce((s,x)=>s+x,0)/n;return {n,mean:m,sd:Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(n-1)),min:sorted[0],p05:sorted[Math.floor(n*.05)],median:sorted[Math.floor(n*.5)],p95:sorted[Math.floor(n*.95)],max:sorted[n-1]};}
function corr(a,b){const n=a.length,ma=a.reduce((s,x)=>s+x,0)/n,mb=b.reduce((s,x)=>s+x,0)/n;let ab=0,aa=0,bb=0;for(let i=0;i<n;i++){const x=a[i]-ma,y=b[i]-mb;ab+=x*y;aa+=x*x;bb+=y*y;}return ab/Math.sqrt(aa*bb);}
const bp=new Map(csv('pair_scores.csv').map(r=>[[r.structure_id,r.binder_chain,r.target_chain].join('|'),r]));
const fp=new Map(csv('pair_scores (1).csv').map(r=>[[r.structure_id,r.binder_chain,r.target_chain].join('|'),r]));
const db=new DatabaseSync('data/dsmbind_tcr_pmhc/test.sqlite',{readOnly:true});
const chainStmt=db.prepare('SELECT coords,length,sequence FROM chains WHERE structure_id=? AND chain_id=?');const chainCache=new Map();
function chain(id,c){const key=id+'|'+c;if(chainCache.has(key))return chainCache.get(key);const row=chainStmt.get(id,c);const buf=zlib.inflateSync(row.coords);const vals=[];for(let i=0;i<buf.length;i+=4)vals.push(buf.readFloatLE(i));const o={vals,seq:row.sequence};chainCache.set(key,o);return o;}
function idx(blob){const b=zlib.inflateSync(blob),a=[];for(let i=0;i<b.length;i+=4)a.push(b.readInt32LE(i));return a;}
function patch(atoms,indices){const a=[];for(const i of indices)for(let j=0;j<14;j++){const o=i*42+j*3,x=atoms[o],y=atoms[o+1],z=atoms[o+2];if(x*x+y*y+z*z>1e-8)a.push([x,y,z]);}return a;}
const rows=db.prepare('SELECT * FROM pairs ORDER BY id').all();
const out=[];for(let k=0;k<rows.length;k++){
 const r=rows[k],b=chain(r.structure_id,r.binder_chain),t=chain(r.structure_id,r.target_chain),bi=idx(r.binder_indices),ti=idx(r.target_indices),ba=patch(b.vals,bi),ta=patch(t.vals,ti);
 let n14=0;for(const a of ba)for(const b of ta){const x=a[0]-b[0],y=a[1]-b[1],z=a[2]-b[2];if(x*x+y*y+z*z<196)n14++;}
 const key=[r.structure_id,r.binder_chain,r.target_chain].join('|'),baseline=bp.get(key),fine=fp.get(key);
 if(!baseline || !fine)throw Error('Missing row '+key);
 out.push({key,n14,combo:r.binder_chain+'-'+r.target_chain,base:+baseline.symmetric_score,fine:+fine.symmetric_score,atoms_b:ba.length,atoms_t:ta.length});
}
for(const field of ['base','fine']){const x=out.map(r=>r.n14),y=out.map(r=>r[field]);const xy=x.reduce((s,v,i)=>s+v*y[i],0),xx=x.reduce((s,v)=>s+v*v,0),slope=xy/xx,m=y.reduce((s,v)=>s+v,0)/y.length;const res=y.map((v,i)=>v-slope*x[i]);console.log(JSON.stringify({field,atom14_count_correlation:corr(x,y),r2_through_origin:1-res.reduce((s,v)=>s+v*v,0)/y.reduce((s,v)=>s+(v-m)**2,0),slope,score_per_masked_atom_pair:stats(out.filter(r=>r.n14>0).map(r=>r[field]/r.n14)),zero_n14:out.filter(r=>!r.n14).length}));}
for(const combo of [...new Set(out.map(r=>r.combo))].sort()){
 const a=out.filter(r=>r.combo===combo);console.log(JSON.stringify({combo,n:a.length,baseline_correlation:corr(a.map(r=>r.n14),a.map(r=>r.base)),finetuned_correlation:corr(a.map(r=>r.n14),a.map(r=>r.fine)),finetuned_score_per_atom_pair:stats(a.map(r=>r.fine/r.n14))}));
}
console.log(JSON.stringify({examples:out.slice(0,6)}));
db.close();
