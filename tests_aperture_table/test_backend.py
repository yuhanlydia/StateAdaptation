from types import SimpleNamespace
import torch
from torch import nn
import pytest
from aperture_table.backend import decoder_sites,lora_targets,answer_span,image_mask,answer_logprobs

def toy_sites():
 class Attn(nn.Module):
  def __init__(self):
   super().__init__();self.q_proj=nn.Linear(8,8);self.k_proj=nn.Linear(8,4);self.v_proj=nn.Linear(8,4);self.o_proj=nn.Linear(8,8)
 class Model(nn.Module):
  def __init__(self):
   super().__init__();self.model=nn.Module();self.model.language_model=nn.Module()
   self.model.language_model.layers=nn.ModuleList([nn.Module() for _ in range(28)])
   for l in self.model.language_model.layers:l.self_attn=Attn()
   self.model.vision_tower=nn.Module();self.model.vision_tower.layers=nn.ModuleList([nn.Module()])
   self.model.vision_tower.layers[0].self_attn=Attn()
 return Model()

def test_sites_are_actual_decoder_modules_not_vision():
 m=toy_sites();sites=decoder_sites(m,[14,27])
 assert [(l,k) for l,k,_ in sites]==[(14,'K'),(14,'V'),(27,'K'),(27,'V')]
 assert all(s[2].out_features==4 for s in sites)
 targets=lora_targets(m)
 assert len(targets)==28*4 and not any('vision' in s for s in targets)
 with pytest.raises(ValueError):decoder_sites(m,[14,99])

def test_prefix_and_answer_boundaries():
 a=torch.tensor([[1,2,3]]);b=torch.tensor([[1,2,3,4,5]])
 assert answer_span(a,b)==(3,5)
 with pytest.raises(ValueError):answer_span(a,torch.tensor([[1,2,9,4]]))
 with pytest.raises(ValueError):answer_span(a,a)

def test_gemma_uses_image_token_index_and_token_types():
 ids=torch.tensor([[1,99,99,3,4]])
 cfg=SimpleNamespace(image_token_index=99)
 batch={'input_ids':ids,'token_type_ids':torch.tensor([[0,1,1,0,0]])}
 assert image_mask(batch,cfg,'gemma3',4).tolist()==[[False,True,True,False,False]]
 with pytest.raises(ValueError):image_mask({'input_ids':ids},cfg,'gemma3',4)
 with pytest.raises(ValueError):image_mask(batch,cfg,'gemma3',2)

def test_qwen_uses_expanded_image_ids_not_text():
 ids=torch.tensor([[1,99,99,3,4]])
 b={'input_ids':ids,'image_grid_thw':torch.tensor([[1,2,4]])}
 cfg=SimpleNamespace(image_token_id=99,vision_config=SimpleNamespace(spatial_merge_size=2))
 assert int(image_mask(b,cfg,'qwen3_vl',4).sum())==2
 b['image_grid_thw']=torch.tensor([[1,4,4]])
 with pytest.raises(ValueError):image_mask(b,cfg,'qwen3_vl',4)

def test_selective_logits_shift_and_gradient():
 logits=torch.randn(1,4,9,requires_grad=True);ids=torch.tensor([[1,2,3,4,5,6]])
 lp=answer_logprobs(logits,ids,(3,6))
 ref=torch.log_softmax(logits[0,:-1].float(),-1).gather(1,ids[0,3:].view(-1,1)).flatten()
 assert torch.allclose(lp,ref)
 (-lp.mean()).backward();assert logits.grad.abs().sum()>0
 with pytest.raises(ValueError):answer_logprobs(logits[:,:2],ids,(3,6))
