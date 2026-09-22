import json
from pathlib import Path
import pytest
from aperture_table.protocol import make_jobs,default_config,expected_cells,source_signature
from aperture_table.reporting import summarize_records,fill_main,export


def records():
 out=[]
 for j in make_jobs(default_config()):
  metric={'macro_f1':.5+j['support_seed']*.1,'nll':.7,'balanced_accuracy':.6,'ece':.1,'brier':.3,'count':j['query_count']}
  r={k:j[k] for k in ('job_id','unit_id','model_key','model_id','domain_key','support_seed','arm')}
  r.update(source=source_signature(),query_ids_sha256=j['domain_key'],query_label_sha256=j['domain_key'],raw=metric,temperature={**metric,'nll':.65})
  if j['arm']=='frozen':r['bias']={**metric,'nll':.68}
  out.append(r)
 return out


def test_112_cells_two_seeds_and_percent_sd():
 summary=summarize_records(records())
 assert set(summary['cells'])==expected_cells()
 c=summary['cells']['q25-h2-aperture-f1']
 assert c['mean']==pytest.approx(55.) and c['sd']==pytest.approx(7.071067811865)
 assert len(summary['rows'])==40


def test_no_partial_or_mixed_export():
 r=records()
 with pytest.raises(ValueError):summarize_records(r[:-1])
 r[-1]['query_ids_sha256']='different'
 with pytest.raises(ValueError):summarize_records(r)
 r=records();r[-1]['model_id']='wrong model'
 with pytest.raises(ValueError):summarize_records(r)


def test_fill_requires_all_exact_keys_and_leaves_source_intact(tmp_path):
 source=tmp_path/'main.tex';dest=tmp_path/'filled.tex';ex=tmp_path/'export';ex.mkdir()
 s='% \\pending{comment}\n'+ '\n'.join('\\pending{'+k+'}' for k in sorted(expected_cells()))
 source.write_text(s)
 summ=summarize_records(records())
 (ex/'main_table_cells.json').write_text(json.dumps(summ))
 from vigor_handoff.protocol import file_hash
 (ex/'coverage.json').write_text(json.dumps({'paper_ready':True,'completed_states':64,'cells_sha256':file_hash(ex/'main_table_cells.json')}))
 fill_main(source,dest,ex)
 assert source.read_text()==s and '\\pending{q25-h2-frozen-f1}' not in dest.read_text()
 source.write_text(s+'\n\\pending{q25-h2-frozen-f1}')
 with pytest.raises(ValueError):fill_main(source,tmp_path/'bad.tex',ex)
 with pytest.raises(ValueError):fill_main(source,source,ex)


def test_failed_export_revokes_stale_table(tmp_path):
 c=default_config();c['run_root']=str(tmp_path/'aperture_table_v1')
 d=Path(c['run_root'])/'paper_exports';d.mkdir(parents=True)
 (d/'main_table.tex').write_text('old numbers')
 with pytest.raises((ValueError,FileNotFoundError)):export(c)
 assert not (d/'main_table.tex').exists()
 assert json.loads((d/'coverage.json').read_text())['paper_ready'] is False
