import csv
import json
from pathlib import Path
import pytest
from aperture_medical_multimodel.data import prepare
from aperture_medical_multimodel.protocol import make_jobs, check_job
from tests_aperture_medical_multimodel.test_protocol import cfg


def test_official_metadata_to_checked_multicenter(tmp_path):
    root=tmp_path/'official';root.mkdir();c=cfg()
    c.update(data_root=str(root),prepared_root=str(tmp_path/'aperture_medical_multimodel_data'),run_root=str(tmp_path/'aperture_medical_multimodel_run'),query_count=100)
    with (root/'metadata.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['patient','node','x_coord','y_coord','center','slide','tumor'])
        for h in range(5):
            for p in range(10):
                for label in (0,1):
                    for i in range(40):w.writerow([f'{h*20+p:03d}',0,label*1000+i,0,h,h*10+p,label])
    prepare(c);prepare(c)
    jobs=make_jobs(c)
    for j in [j for j in jobs if j['arm']=='aperture']:
        check_job(j,assets=False)
    j=jobs[0];p=Path(j['support']);r=json.loads(p.read_text().splitlines()[0]);r['question']='changed'
    lines=p.read_text().splitlines();lines[0]=json.dumps(r);p.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='content changed'):check_job(j,False)
