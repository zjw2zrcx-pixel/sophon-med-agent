#!/usr/bin/env python3
"""Hybrid VITS server: TPU encoder/flow-decoder; CPU duration controller only."""
import argparse,asyncio,base64,io,os,re,sys,time,wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
import uvicorn
state='initializing';error='';runtime=None;frontend=None;pool=ThreadPoolExecutor(1)
MAX_INPUT_TOKENS=50
# The lexicon frontend accepts Chinese sentence punctuation but has no entry
# for several typographic variants commonly emitted by LLMs.  Normalize those
# variants before tokenization instead of turning a valid sentence into a 500.
PUNCTUATION_NORMALIZATION=str.maketrans({
 '、':'，','；':'，','：':'，','（':'，','）':'，','(':'，',')':'，',
 '“':'','”':'','‘':'','’':'','"':'',"'":'',
 '—':'，','–':'，','-':'，','…':'。',
})
SPOKEN_DIGITS=str.maketrans('0123456789','零一二三四五六七八九')

def _normalise_speech_text(value):
 # The Chinese lexicon has no ASCII number/quote entries.  Digit-by-digit
 # wording is deliberately conservative: it correctly handles years, IDs and
 # clock fields without inventing numeric semantics.
 return str(value or '').translate(PUNCTUATION_NORMALIZATION).translate(SPOKEN_DIGITS)
def load(root,configured_max_tokens=None):
 global runtime,frontend,state,error,MAX_INPUT_TOKENS
 try:
  state='loading'; b=root;sys.path.insert(0,str(b));from hybrid_vits_runtime import HybridVitsRuntime,TOKENS
  from melo_zh_lexicon_frontend import MeloZhLexiconFrontend
  if configured_max_tokens is None: configured_max_tokens=TOKENS
  if not 1<=configured_max_tokens<=TOKENS:
   raise ValueError(f'max_input_tokens must be in [1, {TOKENS}], got {configured_max_tokens}')
  MAX_INPUT_TOKENS=configured_max_tokens
  frontend=MeloZhLexiconFrontend(root/'preprocess_assets');runtime=HybridVitsRuntime(b,0);state='ready'
 except Exception as e:error=str(e);state='error'
def _segments(text):
 # The exported encoder has a fixed 50-token input tensor.  Always measure
 # actual frontend output because Chinese characters, English words, and the
 # add_blank convention do not have a reliable character-count approximation.
 def fits(candidate):
  x,_=frontend.convert(candidate)
  return len(x)<=MAX_INPUT_TOKENS

 def hard_split(unit):
  """Use lexical units only when a punctuation-delimited unit is too long."""
  parts=re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9']",unit)
  current=[];result=[]
  for part in parts:
   candidate=''.join(current+[part])
   if part.isspace():
    current.append(part);continue
   if fits(candidate):
    current.append(part);continue
   if not current:
    raise ValueError(f"single lexical unit exceeds {MAX_INPUT_TOKENS} VITS tokens: {part!r}")
   result.append(''.join(current))
   if not fits(part):
    raise ValueError(f"single lexical unit exceeds {MAX_INPUT_TOKENS} VITS tokens: {part!r}")
   current=[part]
  if current:result.append(''.join(current))
  return result

 # Preserve punctuation-delimited clauses first.  Greedily combine adjacent
 # clauses whenever the complete combined text still fits, so short sentences
 # do not waste the fixed input window.
 punctuation=r"，。！？；：,.!?;:"
 units=re.findall(rf"[^{punctuation}]+[{punctuation}]*|[{punctuation}]+",text)
 current="";result=[]
 for unit in units:
  candidate=current+unit
  if fits(candidate):
   current=candidate;continue
  if current:
   result.append(current)
   current=""
  if fits(unit):
   current=unit;continue
  split=hard_split(unit)
  result.extend(split[:-1])
  current=split[-1]
 if current:result.append(current)
 return result
def synth(text,sid):
 chunks=[]
 for segment in _segments(text):
  x,t=frontend.convert(segment);chunks.append(runtime.synthesize_tokens(x,t,sid=sid))
 # A short silent boundary prevents clicks between independently decoded
 # static-50 chunks while preserving a natural continuous spoken response.
 gap=np.zeros(882,dtype=np.float32)
 a=np.concatenate([v for pair in zip(chunks,[gap]*len(chunks)) for v in pair][:-1])
 o=io.BytesIO()
 sf.write(o,a,44100,format='WAV',subtype='FLOAT')
 return o.getvalue()
@asynccontextmanager
async def lifespan(a):asyncio.get_running_loop().run_in_executor(pool,load,Path(a.state.root),a.state.max_input_tokens);yield
app=FastAPI(lifespan=lifespan)
@app.get('/health')
async def health():return {'status':state,'error':error or None,'backend':'TPU-hybrid'}
@app.post('/v1/audio/speech')
async def speech(q:Request):
 if state!='ready':return JSONResponse({'error':{'message':error or state}},503)
 b=await q.json();p=_normalise_speech_text(b.get('input') or b.get('text') or '').strip()
 if not p:return JSONResponse({'error':{'message':'input is required'}},422)
 try:
  st=time.monotonic();wav=await asyncio.get_running_loop().run_in_executor(pool,synth,p,int(b.get('sid',1)));return {'audio':base64.b64encode(wav).decode(),'format':'wav','sample_rate':44100,'latency_ms':round((time.monotonic()-st)*1000,2)}
 except Exception as e:return JSONResponse({'error':{'message':str(e)}},500)
def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8005);p.add_argument('--model-path',required=True);p.add_argument('--config-path');p.add_argument('--module-path');p.add_argument('--devid',type=int,default=0);p.add_argument('--max-input-tokens',type=int,default=None);a=p.parse_args();app.state.root=a.model_path;app.state.max_input_tokens=a.max_input_tokens;uvicorn.run(app,host=a.host,port=a.port,access_log=False)
if __name__=='__main__':main()
