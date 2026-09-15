"""Minimal Phi-3.5-Vision single-image candidate scorer.

Phi uses numbered image tags and a negative image-token representation rather
than Qwen's chat-template image objects; keeping this adapter separate makes
the task protocol comparable without changing the established Qwen path.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
from PIL import Image
from .aggregation import product_of_experts
from .schemas import TaskSample

def load_phi(model_id, source_adapter=None, use_lora=False, efficient_attention=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # The pinned Phi config requests FlashAttention2, which is not part of the
    # reproducible environment; eager attention is numerically equivalent for
    # this evaluation path.
    config._attn_implementation = "eager"
    config._attn_implementation_internal = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
        config=config, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    if efficient_attention:
        # Transformers refuses ``attn_implementation='sdpa'`` for this older
        # remote Phi config even though its bundled modeling file implements a
        # correct SDPA class. Replace the eager attention modules after loading
        # and copy weights exactly; this is required for support backward to fit
        # on the 24-GiB card and is used consistently for Phi runs.
        import importlib
        module = importlib.import_module(type(model.model.layers[0].self_attn).__module__)
        config._attn_implementation = "sdpa"
        config._attn_implementation_internal = "sdpa"
        for layer in model.model.layers:
            old = layer.self_attn
            new = module.Phi3SdpaAttention(config, layer_idx=old.layer_idx)
            new.to(device=old.qkv_proj.weight.device, dtype=old.qkv_proj.weight.dtype)
            new.load_state_dict(old.state_dict())
            layer.self_attn = new
    if source_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, source_adapter, is_trainable=False)
    elif use_lora:
        from peft import LoraConfig, get_peft_model
        config_lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            bias='none', task_type='CAUSAL_LM', target_modules=['qkv_proj','o_proj'])
        model = get_peft_model(model, config_lora)
    # The remote Phi-3.5-Vision modeling file still expects the legacy cache
    # API. Disabling cache is correct for scoring/training and keeps it
    # compatible with current Transformers releases.
    model.config.use_cache = False
    return model, processor

def load_task_image(sample: TaskSample, max_size=448):
    image=Image.open(sample.image).convert('RGB')
    if max(image.size)>max_size:
        scale=max_size/max(image.size)
        image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
    return image

def _batch(processor, sample, image, label):
    image=image if image is not None else load_task_image(sample)
    text=("<|user|>\n<|image_1|>\n"+sample.question+
          "<|end|>\n<|assistant|>\n"+label+"<|end|>\n")
    batch=processor(text=text, images=[image], padding=True, return_tensors='pt')
    ids=processor.tokenizer(label, add_special_tokens=False)['input_ids']
    row=batch['input_ids'][0].tolist()
    start=-1
    for i in range(len(row)-len(ids),-1,-1):
        if row[i:i+len(ids)]==ids: start=i; break
    if start<0: raise ValueError('Phi answer tokens not found')
    return batch,(start,start+len(ids))

def labeled_batch(processor, sample, image=None):
    batch, (start, end) = _batch(processor, sample, image, sample.label)
    labels = torch.full_like(batch['input_ids'], -100)
    labels[0, start:end] = batch['input_ids'][0, start:end]
    batch['labels'] = labels
    return batch, (start, end)

def candidate_scores(model, processor, sample, device=None, controller=None, visual_mask=None):
    device=device or next(model.parameters()).device; scores=[]; model.eval()
    with torch.inference_mode():
        image=load_task_image(sample)
        for label in sample.candidate_labels:
            batch,(start,end)=_batch(processor,sample,image,label)
            batch={k:v.to(device) for k,v in batch.items()}
            if controller is not None:
                controller.set_mask(visual_mask(batch['input_ids']))
            logits=model(**batch).logits.float()[0]
            if controller is not None:
                controller.clear_mask()
            targets=batch['input_ids'][0,start:end]
            lp=torch.log_softmax(logits[start-1:end-1],dim=-1)
            scores.append(float(lp.gather(-1,targets[:,None]).sum()))
    return np.asarray(scores,dtype=np.float64)

def score_task_sample(model, processor, sample, device=None, controller=None, visual_mask=None):
    scores=candidate_scores(model,processor,sample,device,controller,visual_mask)
    probs=product_of_experts([scores.tolist()]); pred=int(np.argmax(probs))
    return {'sample_id':sample.sample_id,'domain_id':sample.domain_id,'group_id':sample.group_id,
            'label':sample.label,'label_id':sample.label_id,'prediction':sample.candidate_labels[pred],
            'probabilities':probs.tolist(),'mean_log_scores':scores.tolist()}

def fit_phi_lora(model, processor, samples: Sequence[TaskSample], passes=4,
                 learning_rate=2e-4, seed=0, max_grad_norm=1.0):
    if not samples or passes == 0: return []
    gen=torch.Generator().manual_seed(seed); device=next(model.parameters()).device
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    model.train(); losses=[]
    for _ in range(passes):
        total=0.; order=torch.randperm(len(samples),generator=gen).tolist()
        for i in order:
            batch,_=labeled_batch(processor,samples[i]); batch={k:v.to(device) for k,v in batch.items()}; loss=model(**batch).loss; loss.backward(); total+=float(loss.detach()); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],max_grad_norm); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(total/len(samples))
    return losses
