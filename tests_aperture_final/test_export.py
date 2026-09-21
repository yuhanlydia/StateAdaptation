import copy
import numpy as np
import pytest
from aperture_final.reporting import aggregate_rows, table_tex


def test_aggregate_uses_sample_sd_and_does_not_pool_query_replicates():
    rows=[dict(domain='hospital_2',method='Aperture',support_seed=i,count=300,
               macro_f1=x,nll=x,balanced_accuracy=x,brier=x,ece=x) for i,x in enumerate([.4,.5,.6])]
    a=aggregate_rows(rows)[0]
    assert a['n_support_seeds']==3 and a['query_count_per_state']==300
    assert a['macro_f1_mean']==pytest.approx(.5) and a['macro_f1_sd']==pytest.approx(.1)
    assert '900' not in table_tex([a],'macro_f1')


def test_single_seed_has_no_invented_sd_and_duplicate_seed_fails():
    r=dict(domain='d',method='A',support_seed=0,count=8,macro_f1=.5,nll=.5,balanced_accuracy=.5,brier=.5,ece=.5)
    assert aggregate_rows([r])[0]['nll_sd'] is None
    with pytest.raises(ValueError):aggregate_rows([r,r])


def test_tex_escapes_identifiers_and_never_bolds_by_hidden_selection():
    a=aggregate_rows([dict(domain='d_1',method='A&B',support_seed=0,count=8,
                   macro_f1=.5,nll=.4,balanced_accuracy=.5,brier=.5,ece=.5)])[0]
    tex=table_tex([a],'nll')
    assert r'd\_1' in tex and r'A\&B' in tex
    assert 'not run' not in tex and r'\pm' not in tex
