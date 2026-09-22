import torch
import pytest
from aperture_table.audit import signature_checks


def signatures():
 g=torch.arange(1.,6.)
 base={'loss':1.,'controller_gradient':g,'forward_counts':{'14:K':1},'activation_gradients':{'14:K':g}}
 other={**base,'forward_counts':{'14:K':2}}
 return base,other


def test_flags_are_not_recomputation_evidence():
 a,b=signatures()
 assert all(signature_checks(a,b).values())
 b['forward_counts']={'14:K':1}
 assert not signature_checks(a,b)['checkpoint_replayed']


def test_invalid_gradients_fail():
 a,b=signatures();b['controller_gradient']=torch.zeros(5)
 assert not signature_checks(a,b)['controller_gradient_agrees']
 a,b=signatures();a['activation_gradients']={'14:K':torch.zeros(5)}
 assert not signature_checks(a,b)['nonzero_visual_gradients']
