"""Native multimodal adapters; no architecture-name-only substitution.

See docs/APERTURE_TABLE_PROTOCOL.md for pinned upstream API sources.
"""
import inspect
import re
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from .protocol import MODELS
from aperture_final.numeric import probabilities
from vigor_handoff.protocol import digest


def _decoder_modules(model):
 result={}
 for name,module in model.named_modules():
  if any(part in name.lower() for part in ('vision','visual','multi_modal','projector')):continue
  match=re.search(r'(?:^|\.)layers\.(\d+)\.self_attn\.([qkvo])_proj$',name)
  if match and hasattr(module,'out_features'):
   key=(int(match.group(1)),match.group(2).upper())
   if key in result:raise ValueError('ambiguous decoder projections')
   result[key]=(name,module)
 if not result:raise ValueError('no explicit decoder attention projections found')
 return result


def decoder_sites(model,layers):
 modules=_decoder_modules(model);sites=[]
 for l in layers:
  for kind in ('K','V'):
   if (l,kind) not in modules:raise ValueError(f'missing decoder site {l}/{kind}')
   sites.append((l,kind,modules[(l,kind)][1]))
 return sites


def lora_targets(model):
 modules=_decoder_modules(model)
 layers=sorted({l for l,_ in modules})
 if layers!=list(range(len(layers))) or set(modules)!={(l,k) for l in layers for k in 'QKVO'}:
  raise ValueError('incomplete decoder Q/K/V/O collection')
 return [modules[k][0] for k in sorted(modules)]


def answer_span(prefix,full):
 if prefix.ndim!=2 or full.ndim!=2 or prefix.shape[0]!=1 or full.shape[0]!=1:raise ValueError('single sample required')
 start=prefix.shape[1];end=full.shape[1]
 if start<1 or end<=start or not torch.equal(prefix,full[:,:start]):raise ValueError('answer tokenization changed prompt prefix')
 return start,end


def image_mask(batch,config,family,answer_start):
 ids=batch['input_ids']
 token=getattr(config,'image_token_index' if family=='gemma3' else 'image_token_id',None)
 if not isinstance(token,int):raise ValueError('model config has no explicit image placeholder ID')
 mask=ids==token
 if not mask.any() or mask[:,answer_start:].any():raise ValueError('missing or answer-overlapping image rows')
 if family=='gemma3':
  types=batch.get('token_type_ids')
  if types is None or not torch.equal(types==1,mask):raise ValueError('Gemma image token types missing or inconsistent')
  expected=getattr(config,'mm_tokens_per_image',None)
  if expected is not None and int(mask.sum())!=expected:raise ValueError('unexpected Gemma image crop/token count')
 else:
  grid=batch.get('image_grid_thw')
  merge=getattr(getattr(config,'vision_config',None),'spatial_merge_size',None)
  if grid is None or merge is None or grid.shape!=(1,3):raise ValueError('Qwen requires one actual image grid')
  if int(grid.prod().item())//(int(merge)**2)!=int(mask.sum()):raise ValueError('Qwen visual token/grid mismatch')
 return mask


def answer_logprobs(logits,ids,span):
 start,end=span;n=end-start
 if n<=0 or logits.ndim!=3 or logits.shape[0]!=1:raise ValueError('invalid candidate span/logits')
 if logits.shape[1]==ids.shape[1]:selected=logits[0,start-1:end-1]
 elif logits.shape[1]==n+1:selected=logits[0,:-1]
 else:raise ValueError('unexpected selective-logit length')
 logp=torch.log_softmax(selected.float(),-1)
 return logp.gather(1,ids[0,start:end].view(-1,1)).flatten()


class Backend:
 def __init__(self,job,model,processor):
  self.job=job;self.model=model;self.processor=processor
  self.base=model.get_base_model() if hasattr(model,'get_base_model') else model
  self.config=self.base.config
  self.device=next(model.parameters()).device
  self.contract=None

 def batch(self,row,candidate):
  if candidate not in row['candidate_labels']:raise ValueError('unknown candidate')
  with Image.open(row['image']) as im:im=im.convert('RGB')
  if self.job['family']!='gemma3':im=im.resize((self.job['image_size'],self.job['image_size']),Image.Resampling.BICUBIC)
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
  prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  images=[[im]] if self.job['family']=='gemma3' else [im]
  kw=dict(images=images,return_tensors='pt',add_special_tokens=False)
  if self.job['family']=='gemma3':kw.update(do_pan_and_scan=False,return_mm_token_type_ids=True)
  prefix=self.processor(text=[prompt],**kw)
  full=self.processor(text=[prompt+candidate],**kw)
  span=answer_span(prefix['input_ids'],full['input_ids'])
  # Token masks/pixel values/position metadata from the native processor survive.
  for key in ('pixel_values','image_grid_thw'):
   if key in prefix and (key not in full or not torch.equal(prefix[key],full[key])):
    raise ValueError('candidate changed image preprocessing')
  mask=image_mask(full,self.config,self.job['family'],span[0])
  contract={'prefix_ids':prefix['input_ids'][0].tolist(),'candidates':row['candidate_labels'],
    'image_tokens':int(mask.sum()),'family':self.job['family'],
    'pixel_shape':list(full['pixel_values'].shape)}
  batch={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in full.items()}
  if 'pixel_values' in batch:batch['pixel_values']=batch['pixel_values'].to(dtype=next(self.model.parameters()).dtype)
  return batch,span,mask.to(self.device),contract

 def candidate(self,row,candidate,controller=None):
  batch,span,mask,contract=self.batch(row,candidate)
  if controller is not None:controller.set_mask(mask)
  n=span[1]-span[0]
  if 'logits_to_keep' not in inspect.signature(self.base.forward).parameters:
   raise RuntimeError('requires upstream selective-logit API; install the documented Transformers version')
  out=self.model(**batch,use_cache=False,logits_to_keep=n+1,return_dict=True)
  return answer_logprobs(out.logits,batch['input_ids'],span),mask,contract

 def loss_and_mask(self,row,controller=None):
  lp,mask,_=self.candidate(row,row['label'],controller)
  return -lp.mean(),mask

 def score(self,row,controller=None):
  scores=[];contract=None
  try:
   with torch.no_grad():
    for c in row['candidate_labels']:
     lp,_,ctx=self.candidate(row,c,controller)
     if contract is not None and ctx!=contract:raise ValueError('candidate-specific prompt context')
     contract=ctx;scores.append(float(lp.sum()))
   p=probabilities([scores])[0]
   return {'sample_id':row['sample_id'],'label_id':row['label_id'],
       'mean_log_scores':scores,'probabilities':p.tolist(),'prediction':int(np.argmax(scores)),
       'prompt_contract_hash':digest(contract),'image_tokens':contract['image_tokens']}
  finally:
   if controller is not None:controller.clear_mask()


@contextmanager
def loaded(job,restore=None):
 import gc
 import transformers
 from packaging.version import Version
 from torch.nn.attention import sdpa_kernel,SDPBackend
 from vigor_handoff.core import ResidualController
 if not (Version('4.57.1')<=Version(transformers.__version__)<Version('5')):
  raise RuntimeError('This adapter targets Transformers >=4.57.1,<5; use an isolated environment, not a silent library upgrade')
 if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<22*1024**3:
  raise RuntimeError('fixed BF16 protocol requires a 24GB-or-larger CUDA GPU')
 cls=getattr(transformers,MODELS[job['model_key']][2])
 processor=transformers.AutoProcessor.from_pretrained(job['model_path'],local_files_only=True)
 model=controller=None
 with sdpa_kernel(SDPBackend.MATH):
  try:
   model=cls.from_pretrained(job['model_path'],local_files_only=True,
      dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
   if model.config.model_type!=job['family']:raise ValueError('loaded model family mismatch')
   if getattr(model,'is_quantized',False):raise ValueError('quantization is not part of this protocol')
   for p in model.parameters():p.requires_grad_(False)
   if job['arm'].startswith('lora'):
    from peft import LoraConfig,get_peft_model,PeftModel
    if restore:
     model=PeftModel.from_pretrained(model,str(Path(restore)/'adapter'),is_trainable=False,local_files_only=True)
    else:
     targets=lora_targets(model)
     model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
               target_modules=targets,bias='none',task_type='CAUSAL_LM'))
     names=[n for n,p in model.named_parameters() if p.requires_grad]
     if not names or any('lora_' not in n or any(x in n for x in ['vision','visual']) for n in names):
      raise ValueError('trainable parameters escaped decoder-only LoRA')
     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
     model.enable_input_require_grads()
   elif restore and job['arm']=='aperture':
    payload=torch.load(Path(restore)/'controller.pt',weights_only=True,map_location='cpu')
    bases={(int(k.split(':')[0]),k.split(':')[1]):v for k,v in payload['bases'].items()}
    controller=ResidualController(decoder_sites(model,job['layers']),bases,payload['alpha'],'full')
    controller.restore(payload)
   if restore:
    for p in model.parameters():p.requires_grad_(False)
   model.eval();model.config.use_cache=False
   yield model,Backend(job,model,processor),controller
  finally:
   if controller is not None:controller.close()
   del model;gc.collect();torch.cuda.empty_cache()
