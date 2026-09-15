"""Arithmetic and local-operator checks; not an empirical VLM experiment."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
data = json.loads((root/'plotting/figure_data.json').read_text())
primary=data['primary']['methods']
assert primary[1]['scalars']==1024 and primary[2]['scalars']==10092544
assert primary[2]['scalars']/primary[1]['scalars']==9856
assert np.isclose(primary[1]['macro_f1']-primary[2]['macro_f1'],.0222)
assert np.isclose(primary[1]['macro_f1']-data['primary']['frozen_macro_f1'],.0974)
a=np.array(data['prior_reweighting']['original_kappa'])
b=np.array(data['prior_reweighting']['reweighted_kappa'])
assert np.isclose(a.mean(),.2092,atol=5e-5)
assert np.isclose(a.std(ddof=1),.3613,atol=5e-5)
assert np.isclose(b.mean(),.9340,atol=5e-5)
assert np.isclose(b.std(ddof=1),.0254,atol=5e-5)
assert data['basis_control']['methods'][1]['rank']==1
assert all(-1<=v<=1 for row in data['module_alignment']['kappa'] for v in row)

# Local mathematical operator / chain-rule sanity check in double precision.
rng=np.random.default_rng(91)
Z=rng.normal(size=(7,8));Y=rng.normal(size=(7,8))
B=np.linalg.qr(rng.normal(size=(8,3)))[0]
A=rng.normal(size=(3,3))*.15
M=np.array([1,0,1,0,1,0,0],dtype=float)[:,None]
U=Z@B

def loss(mat):
    pred=Z+M*(U@mat@B.T)
    return .5*np.square(pred-Y).sum()
G=Z+M*(U@A@B.T)-Y
analytic=U.T@(M*G)@B
numeric=np.empty_like(A);eps=1e-6
for i in range(3):
 for j in range(3):
  plus=A.copy();minus=A.copy();plus[i,j]+=eps;minus[i,j]-=eps
  numeric[i,j]=(loss(plus)-loss(minus))/(2*eps)
assert np.allclose(analytic,numeric,rtol=2e-7,atol=2e-8)
R=rng.normal(size=(3,3));alpha=3
boundA=alpha*R/(1+np.linalg.norm(R))
delta=M*((Z@B)@boundA@B.T)
assert np.linalg.norm(boundA,2)<alpha
assert np.linalg.norm(delta)<=alpha*np.linalg.norm(M*(Z@B))+1e-12
assert np.allclose(delta@(np.eye(8)-B@B.T),0,atol=1e-12)
assert np.array_equal((Z+delta)[M[:,0]==0],Z[M[:,0]==0])
print('PASS: reported arithmetic, data scopes, and finite-difference local controller gradient.')
print('This is not a backbone forward/backward test and does not generate new performance results.')
