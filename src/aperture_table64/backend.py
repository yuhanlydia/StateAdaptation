"""Native Qwen2.5/Qwen3-VL/Gemma3 interfaces with one explicit scoring contract.

Transformers and PEFT are imported only for real-model execution. No generation,
query labels, remote code, fallback quantization, or candidate-specific crop is used.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from vigor_handoff.protocol import digest
from aperture_final.numeric import probabilities


def answer_span(prefix: torch.Tensor, full: torch.Tensor) -> tuple[int, int]:
    if prefix.ndim != 2 or full.ndim != 2 or prefix.shape[0] != 1 or full.shape[0] != 1:
        raise ValueError('candidate scoring requires an unpadded single example')
    n = prefix.shape[1]
    if n < 1 or full.shape[1] <= n or not torch.equal(prefix, full[:, :n]):
        raise ValueError('assistant prefix is not an exact token prefix; refuse an ambiguous answer span')
    return n, full.shape[1]


def image_mask(batch, token_id: int, span, family: str):
    ids = batch['input_ids']
    mask = ids == token_id
    if ids.ndim != 2 or ids.shape[0] != 1 or not mask.any():
        raise ValueError('expected one single-image token group')
    positions = mask[0].nonzero().flatten()
    if (positions[1:] != positions[:-1] + 1).any() or int(positions[-1]) >= span[0]:
        raise ValueError('unexpected image grouping or answer-span overlap')
    if family == 'gemma3':
        types = batch.get('token_type_ids')
        if types is None or types.shape != ids.shape or not torch.equal(types == 1, mask):
            raise ValueError('Gemma image token types are missing or inconsistent; never discard token_type_ids')
    if 'attention_mask' in batch and not batch['attention_mask'][mask].bool().all():
        raise ValueError('image tokens are padded')
    return mask


def answer_logscore(logits, input_ids, span):
    start, end = span
    if not (1 <= start < end <= input_ids.shape[1]) or logits.shape[:2] != input_ids.shape:
        raise ValueError('invalid next-token alignment')
    logp = torch.log_softmax(logits[:, start-1:end-1, :].float(), -1)
    targets = input_ids[:, start:end].unsqueeze(-1)
    score = logp.gather(-1, targets).sum()
    if not torch.isfinite(score):
        raise ValueError('nonfinite candidate log score')
    return score


def decoder_targets(model):
    targets = []
    layers = {}
    for name, _ in model.named_modules():
        hit = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn\.([qkvo])_proj$', name)
        if hit and not any(x in name.lower() for x in ('vision', 'visual')):
            targets.append(name)
            layers.setdefault(int(hit[1]), []).append(hit[2])
    if not layers or any(sorted(v) != list('koqv') for v in layers.values()):
        raise ValueError('missing or duplicate decoder Q/K/V/O projection modules')
    return targets


def projection_sites(model, layers=(14, 27)):
    names = set(decoder_targets(model)); result = []
    for name, module in model.named_modules():
        if name not in names: continue
        hit = re.search(r'layers\.(\d+)\.self_attn\.([kv])_proj$', name)
        if hit and int(hit[1]) in layers:
            if not hasattr(module, 'out_features'):
                raise ValueError('projection has no actual output width')
            result.append((int(hit[1]), hit[2].upper(), module))
    if {(l,k) for l,k,_ in result} != {(l,k) for l in layers for k in ('K','V')} or len(result) != 4:
        raise ValueError('the fixed layers 14/27 do not expose four distinct decoder K/V sites')
    return sorted(result, key=lambda x: (x[0], x[1]))


def enable_checkpointing(model, enabled):
    disable = getattr(model, 'gradient_checkpointing_disable', None)
    if callable(disable): disable()
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.config.use_cache = False


def load_native(job, use_lora=False, restore_adapter=None):
    from packaging.version import Version
    import transformers
    if not (Version('4.57.1') <= Version(transformers.__version__) < Version('5')):
        raise RuntimeError('This isolated runner targets transformers>=4.57.1,<5. Use the supplied separate environment; do not upgrade a running historical study.')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('a BF16-capable CUDA GPU is required')
    if torch.cuda.get_device_properties(0).total_memory < 22 * 1024**3:
        raise RuntimeError('fixed BF16 protocol requires a 24GB-or-larger GPU')
    classes = {'qwen2': 'Qwen2_5_VLForConditionalGeneration',
               'qwen3_vl': 'Qwen3VLForConditionalGeneration', 'gemma3': 'Gemma3ForConditionalGeneration'}
    klass = getattr(transformers, classes[job['family']])
    options = dict(local_files_only=True)
    if job['family'] == 'gemma3':
        options['do_pan_and_scan'] = False
    else:
        options.update(min_pixels=4*28*28, max_pixels=448*448)
    processor = transformers.AutoProcessor.from_pretrained(job['model_path'], **options)
    model = klass.from_pretrained(job['model_path'], local_files_only=True, torch_dtype=torch.bfloat16,
                device_map={'': 0}, attn_implementation='eager', low_cpu_mem_usage=True)
    if {p.device.type for p in model.parameters()} != {'cuda'}:
        raise RuntimeError('CPU/meta/disk offload is not part of this experiment')
    for p in model.parameters(): p.requires_grad_(False)
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        if restore_adapter is not None:
            model = PeftModel.from_pretrained(model, str(restore_adapter), is_trainable=False)
        else:
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                  lora_dropout=.05, bias='none', target_modules=decoder_targets(model))
            model = get_peft_model(model, cfg)
            if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
                raise RuntimeError('unexpected trainable backbone parameter')
            enable_checkpointing(model, True)
    model.config.use_cache=False
    model.eval()
    projection_sites(model, job['layers'])
    return model, processor


class NativeBackend:
    """Same prefix, image processing and candidate loss for adaptation and evaluation."""
    def __init__(self, job, model, processor):
        self.job, self.model, self.processor = job, model, processor
        cfg = model.config
        self.image_token_id = getattr(cfg, 'image_token_id', None)
        if self.image_token_id is None: self.image_token_id = getattr(cfg, 'image_token_index', None)
        if self.image_token_id is None: raise ValueError('model configuration has no image token ID')
        if getattr(processor,'image_token_id',self.image_token_id) != self.image_token_id:
            raise ValueError('processor/model image token IDs disagree')
        self.device = next(model.parameters()).device
        self._cache = {}  # CPU batches only, explicitly bounded; never stores model states.

    def _prefix(self, sample):
        instruction = 'Answer the visual question using exactly one candidate label and no explanation. '
        # A single user message is supported natively by all three chat templates.
        messages = [{'role':'user','content':[{'type':'image'},
                    {'type':'text','text':instruction + sample.question}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _process(self, text, image):
        kw = dict(text=[text], images=[image], return_tensors='pt', padding=False, add_special_tokens=False)
        if self.job['family'] == 'gemma3': kw.update(do_pan_and_scan=False, return_mm_token_type_ids=True)
        batch = dict(self.processor(**kw))
        if 'labels' in batch: raise ValueError('processor injected training labels')
        if not torch.is_tensor(batch.get('pixel_values')): raise ValueError('processor dropped the image')
        return batch

    def batch(self, sample, candidate):
        if candidate not in sample.candidate_labels: raise ValueError('unknown candidate')
        key = (sample.sample_id, str(Path(sample.image).resolve()), sample.question, candidate)
        if key not in self._cache:
            with Image.open(sample.image) as handle: image = handle.convert('RGB')
            if self.job['family'] != 'gemma3':
                image = image.resize((448,448), Image.Resampling.BICUBIC)
            text = self._prefix(sample)
            prefix = self._process(text, image)
            full = self._process(text + candidate, image)
            span = answer_span(prefix['input_ids'], full['input_ids'])
            mask = image_mask(full, self.image_token_id, span, self.job['family'])
            for k in ('pixel_values','image_grid_thw'):
                if k in prefix and not torch.equal(prefix[k],full[k]):
                    raise ValueError('image processing changed between prefix and candidate')
            if len(self._cache) >= 24: self._cache.clear()
            self._cache[key] = full, span
        full,span = self._cache[key]
        # Token types and image-grid tensors are retained, with integer dtypes intact.
        moved = {k:v.to(device=self.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                 if torch.is_tensor(v) else v for k,v in full.items()}
        return moved, span

    def loss_and_mask(self, sample, controller=None):
        batch,span = self.batch(sample,sample.label)
        mask = image_mask(batch,self.image_token_id,span,self.job['family'])
        if controller is not None: controller.set_mask(mask)
        # Omit labels deliberately: both fitting and scoring use the same native
        # attention-mask path, including Gemma's bidirectional image-token blocks.
        out = self.model(**batch,use_cache=False,return_dict=True)
        loss = -answer_logscore(out.logits,batch['input_ids'],span)/(span[1]-span[0])
        return loss,mask

    def score(self,sample,controller=None):
        was_training=self.model.training; self.model.eval()
        scores=[];spans=[];contract=None
        try:
            with torch.no_grad():
                for candidate in sample.candidate_labels:
                    batch,span=self.batch(sample,candidate)
                    mask=image_mask(batch,self.image_token_id,span,self.job['family'])
                    this={'prefix':batch['input_ids'][0,:span[0]].tolist(),
                          'image_grid':batch['image_grid_thw'].tolist() if 'image_grid_thw' in batch else None,
                          'pixel_shape':list(batch['pixel_values'].shape), 'visual_tokens':int(mask.sum()),
                          'candidates':list(sample.candidate_labels)}
                    if contract is not None and this!=contract: raise ValueError('candidate prompt contracts differ')
                    contract=this
                    if controller is not None:controller.set_mask(mask)
                    out=self.model(**batch,use_cache=False,return_dict=True)
                    scores.append(float(answer_logscore(out.logits,batch['input_ids'],span)))
                    spans.append(batch['input_ids'][0,span[0]:span[1]].tolist())
                    del out
            p=probabilities([scores])[0];pred=int(np.argmax(scores))
            return dict(sample_id=sample.sample_id,group_id=sample.group_id,label=sample.label,
                label_id=sample.label_id,candidate_labels=list(sample.candidate_labels),
                mean_log_scores=scores,probabilities=p.tolist(),predicted_id=pred,
                predicted_label=sample.candidate_labels[pred],prompt_contract_hash=digest(contract),
                visual_tokens=contract['visual_tokens'],answer_token_ids=spans)
        finally:
            if controller is not None:controller.clear_mask()
            self.model.train(was_training)
