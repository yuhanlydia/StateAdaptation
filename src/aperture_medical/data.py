"""Official-data adapters. Nothing downloads or runs a model at import time."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from vigor_handoff.protocol import file_hash

PATH_URL='https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1'
PATH_MD5='2c51a510bcdc9cf8ddb2af93af1eadec'
TISSUES=['adipose','background','debris','lymphocytes','mucus','smooth muscle',
         'normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium']
CAM_QUESTION='Does the central region contain tumor tissue? Answer exactly normal or tumor.'


def pixel_hash(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert('RGB'))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def camelyon_population(spec):
    import pandas as pd
    root=Path(spec['source_root']).resolve()
    if not (root/'metadata.csv').is_file():
        raise FileNotFoundError(f'{root}/metadata.csv: need official CAMELYON17-WILDS root, not H2-only manifests')
    df=pd.read_csv(root/'metadata.csv',dtype={'patient':str})
    need={'patient','node','x_coord','y_coord','center','tumor','slide'}
    if not need<=set(df.columns):raise ValueError('official CAMELYON metadata columns missing')
    center=int(spec['name'].split('_')[-1]);df=df[df.center==center]
    if df.empty:raise ValueError(f'no records for hospital {center}')
    rows=[]
    for patient,node,x,y,tumor,slide in df[['patient','node','x_coord','y_coord','tumor','slide']].itertuples(index=False,name=None):
        patient=str(patient);node=int(node);x=int(x);y=int(y);c=int(tumor)
        if c not in (0,1):raise ValueError('CAMELYON label must be 0/1')
        filename=f'patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
        rows.append(dict(sample_id='cam17-'+filename[:-4],domain_id=spec['name'],group_id='patient_'+patient,
            image=str(root/'patches'/f'patient_{patient}_node_{node}'/filename),
            label=['normal','tumor'][c],label_id=c,candidate_labels=['normal','tumor'],
            question=CAM_QUESTION,dataset='camelyon17-wilds',
            metadata={'patient':patient,'slide':int(slide),'center':center,'node':node,
                      'label_definition':'tumor in central 32x32 of original 96x96', 'isolation':'patient'}))
    return rows,{'source':str(root/'metadata.csv'),'sha256':file_hash(root/'metadata.csv'),
                 'population_count':len(rows),'center':center}


def path_population(spec, cache_dir):
    """Only CRC test cohort is used; patient IDs are NOT supplied by the NPZ."""
    path=Path(spec['source_root']).resolve();cache=Path(cache_dir)/'path_images'
    if not path.is_file():raise FileNotFoundError(path)
    h=hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    if h.hexdigest()!=PATH_MD5:raise ValueError('PathMNIST-224 MD5 mismatch; no 28px substitution')
    cache.mkdir(parents=True,exist_ok=True)
    with np.load(path,allow_pickle=False) as data:
        images=data['test_images'];labels=data['test_labels'].reshape(-1)
    if images.shape!=(7180,224,224,3) or labels.shape!=(7180,):raise ValueError('unexpected official PathMNIST test shape')
    question='Classify the tissue in this histopathology image. '+ '; '.join(
        f'{chr(65+i)}: {s}' for i,s in enumerate(TISSUES))+'. Answer with exactly one letter A through I.'
    rows=[];seen={}
    for i,(a,c) in enumerate(zip(images,labels)):
        c=int(c)
        if not 0<=c<9:raise ValueError('unknown tissue label')
        ph=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
        if ph in seen:
            if seen[ph]!=c:raise ValueError('identical PathMNIST image has conflicting labels')
            continue
        seen[ph]=c
        p=cache/f'crc_{i:05d}.png'
        if p.exists():
            if pixel_hash(p)!=ph:raise ValueError(f'cached image changed: {p}')
        else:Image.fromarray(a).save(p)
        rows.append(dict(sample_id=f'pathmnist-test-{i}',domain_id=spec['name'],group_id='image_'+ph,
            image=str(p.resolve()),label=chr(65+c),label_id=c,candidate_labels=list('ABCDEFGHI'),
            question=question,dataset='pathmnist',metadata={'pixel_sha256':ph,'source_index':i,
            'isolation':'image_content_only; patient/slide IDs unavailable','source_partition':'test',
            'cohort':'CRC-VAL-HE-7K','class_name':TISSUES[c]}))
    return rows,{'source':str(path),'md5':h.hexdigest(),'population_count':len(rows),
                 'omitted_exact_duplicates':7180-len(rows),'patient_disjoint':False}


def population(spec, cache_dir):
    if spec['dataset']=='camelyon17-wilds':return camelyon_population(spec)
    if spec['dataset']=='pathmnist':return path_population(spec,cache_dir)
    raise ValueError('unregistered medical dataset')
