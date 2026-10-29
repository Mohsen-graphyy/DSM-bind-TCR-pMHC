// Read-only inspection of the supplied PyTorch ZIP checkpoints. No pickle globals are executed.
// Run from the workspace root with Node 22+: node reports/dsmbind_finetune_audit_2026-10-28/checkpoint_audit.cjs --verify
const fs = require('node:fs');
const zlib = require('node:zlib');
function archive(path) {
  const b=fs.readFileSync(path), entries=new Map();
  let e=b.length-22; while(e>=0 && b.readUInt32LE(e)!==0x06054b50)e--;
  if(e<0)throw Error('No ZIP footer');
  let p=b.readUInt32LE(e+16), n=b.readUInt16LE(e+10);
  for(let i=0;i<n;i++){
    if(b.readUInt32LE(p)!==0x02014b50)throw Error('Invalid ZIP index');
    const method=b.readUInt16LE(p+10),size=b.readUInt32LE(p+20), nl=b.readUInt16LE(p+28),el=b.readUInt16LE(p+30),cl=b.readUInt16LE(p+32), off=b.readUInt32LE(p+42),name=b.subarray(p+46,p+46+nl).toString();
    const start=off+30+b.readUInt16LE(off+26)+b.readUInt16LE(off+28),raw=b.subarray(start,start+size);
    entries.set(name, method===0?raw:zlib.inflateRawSync(raw));p+=46+nl+el+cl;
  }return entries;
}
const MARK=Symbol('mark');
function unpickle(b){
  let p=0; const stack=[],memo=new Map();
  const line=()=>{const q=b.indexOf(10,p),s=b.subarray(p,q).toString();p=q+1;return s;};
  const popmark=()=>{const j=stack.lastIndexOf(MARK);if(j<0)throw Error('No mark');return stack.splice(j).slice(1);};
  const setitems=(o,a)=>{for(let i=0;i<a.length;i+=2)o[a[i]]=a[i+1];};
  const reduce=(fn,args)=>{
    if(fn?.global==='collections.OrderedDict')return Object.fromEntries(args[0]||[]);
    if(fn?.global==='torch._utils._rebuild_tensor_v2'||fn?.global==='torch._utils._rebuild_tensor')return {tensor:true,storage:args[0],offset:args[1],shape:args[2],stride:args[3],requires_grad:args[4]};
    return {reduced:fn,args};
  };
  while(p<b.length){const op=b[p++];switch(op){
    case 0x80:p++;break;case 0x95:p+=8;break;
    case 0x2e:return stack.pop();case 0x28:stack.push(MARK);break;
    case 0x4e:stack.push(null);break;case 0x88:stack.push(true);break;case 0x89:stack.push(false);break;
    case 0x4b:stack.push(b[p++]);break;case 0x4d:stack.push(b.readUInt16LE(p));p+=2;break;case 0x4a:stack.push(b.readInt32LE(p));p+=4;break;
    case 0x47:stack.push(b.readDoubleBE(p));p+=8;break;
    case 0x8a:{const n=b[p++];let v=0n;for(let i=0;i<n;i++)v|=BigInt(b[p+i])<<BigInt(8*i);p+=n;stack.push(Number(v));break;}
    case 0x58:{const n=b.readUInt32LE(p);p+=4;stack.push(b.subarray(p,p+n).toString());p+=n;break;}
    case 0x8c:case 0x55:{const n=b[p++];stack.push(b.subarray(p,p+n).toString());p+=n;break;}
    case 0x54:case 0x42:{const n=b.readUInt32LE(p);p+=4;stack.push(b.subarray(p,p+n));p+=n;break;}
    case 0x43:{const n=b[p++];stack.push(b.subarray(p,p+n));p+=n;break;}
    case 0x63:stack.push({global:line()+'.'+line()});break;
    case 0x93:{const name=stack.pop(),mod=stack.pop();stack.push({global:mod+'.'+name});break;}
    case 0x52:{const args=stack.pop(),fn=stack.pop();stack.push(reduce(fn,args));break;}
    case 0x81:{const args=stack.pop(),fn=stack.pop();stack.push({instance:fn,args});break;}
    case 0x62:{const state=stack.pop();const o=stack[stack.length-1];if(state&&typeof state==='object')Object.assign(o,state);break;}
    case 0x51:stack.push({persistent:stack.pop()});break;
    case 0x71:memo.set(b[p++],stack.at(-1));break;case 0x72:memo.set(b.readUInt32LE(p),stack.at(-1));p+=4;break;
    case 0x94:memo.set(memo.size,stack.at(-1));break;case 0x68:stack.push(memo.get(b[p++]));break;case 0x6a:stack.push(memo.get(b.readUInt32LE(p)));p+=4;break;
    case 0x29:stack.push([]);break;case 0x74:stack.push(popmark());break;
    case 0x85:stack.push([stack.pop()]);break;case 0x86:{const z=stack.pop(),y=stack.pop();stack.push([y,z]);break;}
    case 0x87:{const z=stack.pop(),y=stack.pop(),x=stack.pop();stack.push([x,y,z]);break;}
    case 0x5d:stack.push([]);break;case 0x6c:stack.push(popmark());break;
    case 0x61:{const v=stack.pop();stack.at(-1).push(v);break;}case 0x65:{const a=popmark();stack.at(-1).push(...a);break;}
    case 0x7d:stack.push({});break;case 0x64:{const a=popmark(),o={};setitems(o,a);stack.push(o);break;}
    case 0x73:{const v=stack.pop(),k=stack.pop();stack.at(-1)[k]=v;break;}case 0x75:{const a=popmark();setitems(stack.at(-1),a);break;}
    default:throw Error('Unknown pickle opcode 0x'+op.toString(16)+' at '+(p-1));
  }}throw Error('No STOP');
}
function load(path){const zip=archive(path), pk=[...zip.keys()].find(k=>k.endsWith('/data.pkl')),prefix=pk.slice(0,-8);return {obj:unpickle(zip.get(pk)),zip,prefix};}
function values(c,t){
  if(!t?.tensor)throw Error('Not tensor');const s=t.storage.persistent,dtype=s[1].global,key=s[2];
  if(dtype!=='torch.FloatStorage')throw Error('Only float supported '+dtype);
  const buf=c.zip.get(c.prefix+'data/'+key), n=t.shape.reduce((a,b)=>a*b,1),a=new Float32Array(n);
  for(let i=0;i<n;i++)a[i]=buf.readFloatLE((t.offset+i)*4);return a;
}
function stats(a){let sum=0,ss=0,min=Infinity,max=-Infinity,nonfinite=0,zeros=0;for(const x of a){if(!Number.isFinite(x))nonfinite++;if(x===0)zeros++;sum+=x;ss+=x*x;min=Math.min(min,x);max=Math.max(max,x);}return {n:a.length,l2:Math.sqrt(ss),rms:Math.sqrt(ss/a.length),mean:sum/a.length,min,max,nonfinite,zeros};}
function compare(a,b){let ss=0,aa=0,bb=0,ab=0,mx=0,changed=0;for(let i=0;i<a.length;i++){let d=a[i]-b[i];ss+=d*d;aa+=a[i]*a[i];bb+=b[i]*b[i];ab+=a[i]*b[i];mx=Math.max(mx,Math.abs(d));if(d!==0)changed++;}return {rel_l2:Math.sqrt(ss/aa),max_abs:mx,cosine:ab/Math.sqrt(aa*bb),changed_fraction:changed/a.length};}
const dir='C:/Users/hajmo/Downloads/';
const files=[['pretrained','DSMBind/ckpts/model.skempi.allatom'],['best',dir+'best.pt'],['last',dir+'last.pt'],['pilot_best',dir+'best (1).pt'],['pilot_last',dir+'last (1).pt']];
const checkpoints={};for(const [name,path]of files){const c=load(path);checkpoints[name]=c;const o=c.obj;console.log(JSON.stringify({checkpoint:name,keys:Object.keys(o),architecture:Array.isArray(o)?o[2]:o.architecture,epoch:o.epoch,history:o.history,best_validation_loss:o.best_validation_loss,training_config:o.training_config}));}
for(const c of Object.values(checkpoints))c.state=Array.isArray(c.obj)?c.obj[0]:c.obj.model_state;
if(process.argv.includes('--verify')){
 const crypto=require('node:crypto');
 function raw(c,t){const n=t.shape.reduce((a,b)=>a*b,1);return c.zip.get(c.prefix+'data/'+t.storage.persistent[2]).subarray(t.offset*4,(t.offset+n)*4);}
 const report=[];
 for(const [name,c]of Object.entries(checkpoints)){
  const o=Array.isArray(c.obj)?c.obj[1]:c.obj.optimizer_state,keys=Object.keys(c.state).filter(k=>c.state[k]?.tensor && !k.endsWith('.scale_x'));
  if(keys.length!==o.param_groups[0].params.length)throw Error('Parameter count mismatch');
  for(let i=0;i<keys.length;i++){
   const s=o.state[o.param_groups[0].params[i]];
   if(s?.exp_avg && JSON.stringify(c.state[keys[i]].shape)!==JSON.stringify(s.exp_avg.shape))throw Error('Parameter mapping mismatch '+keys[i]);
  }
  const heads=keys.filter(k=>k.startsWith('W_o.')||k.startsWith('U_o.')).map(key=>({key,n:c.state[key].shape.reduce((a,b)=>a*b,1),sha256:crypto.createHash('sha256').update(raw(c,c.state[key])).digest('hex'),identical_to_pretrained:raw(c,c.state[key]).equals(raw(checkpoints.pretrained,checkpoints.pretrained.state[key])),optimizer_state_present:!!o.state[o.param_groups[0].params[keys.indexOf(key)]]}));
  const changed=keys.filter(k=>!raw(c,c.state[k]).equals(raw(checkpoints.pretrained,checkpoints.pretrained.state[k])));
  report.push({name,parameter_tensors:keys.length,optimizer_state_tensors:Object.keys(o.state).length,trained_steps:Array.from(values(c,o.state[0].step))[0],changed_vs_pretrained:changed,heads});
 }console.log(JSON.stringify({verified_checkpoint_audit:report},null,2));process.exit(0);
}
for(const name of ['pretrained','best','last','pilot_best']){
  const c=checkpoints[name],o=Array.isArray(c.obj)?c.obj[1]:c.obj.optimizer_state,keys=Object.keys(c.state).filter(k=>c.state[k]?.tensor && !k.endsWith('.scale_x'));
  console.log(JSON.stringify({optimizer:name,groups:o.param_groups,states:o.param_groups[0].params.map((id,i)=>({key:keys[i],id,state_keys:Object.keys(o.state[id]||{}),step:o.state[id]?.step?.tensor?Array.from(values(c,o.state[id].step)):o.state[id]?.step,exp_avg_shape:o.state[id]?.exp_avg?.shape,exp_avg_rms:o.state[id]?.exp_avg?stats(values(c,o.state[id].exp_avg)).rms:null}))}));
}
if(process.argv.includes('--compact')){
 for(const name of ['best','last','pilot_best']){
   const a=checkpoints.pretrained,b=checkpoints[name];
   console.log(JSON.stringify({comparison:'pretrained vs '+name,parameters:Object.entries(a.state).filter(([k,t])=>t?.tensor).map(([key,t])=>({key,...compare(values(a,t),values(b,b.state[key]))}))}));
 }
 process.exit(0);
}
for(const [name,c]of Object.entries(checkpoints)){
  console.log(JSON.stringify({name,parameters:Object.entries(c.state).filter(([k,t])=>t?.tensor).map(([key,t])=>({key,shape:t.shape,stats:stats(values(c,t))}))}));
}
for(const [left,right]of [['pretrained','best'],['best','last'],['pretrained','pilot_best']]){
  const a=checkpoints[left],b=checkpoints[right];console.log(JSON.stringify({comparison:left+' vs '+right,parameters:Object.entries(a.state).filter(([k,t])=>t?.tensor).map(([key,t])=>({key,...compare(values(a,t),values(b,b.state[key]))}))}));
}
