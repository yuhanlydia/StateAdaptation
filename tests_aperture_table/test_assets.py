import json
from pathlib import Path
import pytest
from aperture_table.worker import check_assets
from aperture_table.protocol import make_jobs,default_config


def checkpoint(tmp_path):
 p=tmp_path/'model';p.mkdir()
 (p/'config.json').write_text(json.dumps({'model_type':'qwen2_5_vl','text_config':{'num_hidden_layers':28}}))
 for n in ('tokenizer_config.json','preprocessor_config.json'):(p/n).write_text('{}')
 (p/'model.safetensors').write_bytes(b'synthetic fixture, not real model')
 j=make_jobs(default_config())[0];j['model_path']=str(p)
 return p,j


def test_asset_fingerprints_reject_changed_weights(tmp_path):
 p,j=checkpoint(tmp_path);a=check_assets(j,tmp_path/'assets');assert a==check_assets(j,tmp_path/'assets')
 (p/'model.safetensors').write_bytes(b'changed')
 with pytest.raises(ValueError):check_assets(j,tmp_path/'assets')


def test_quantization_or_missing_shards_fail(tmp_path):
 p,j=checkpoint(tmp_path)
 (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'missing.safetensors'}}))
 with pytest.raises(FileNotFoundError):check_assets(j,tmp_path/'assets')
 (p/'config.json').write_text(json.dumps({'model_type':'qwen2_5_vl','quantization_config':{'bits':4}}))
 with pytest.raises(ValueError):check_assets(j,tmp_path/'assets')
