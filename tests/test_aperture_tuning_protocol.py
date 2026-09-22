"""CPU contracts for the prespecified search and calibration-only selection."""
import json

import pytest

from aperture_tuning.protocol import candidates, select_rows


@pytest.mark.parametrize('depth', [28, 36, 34])
def test_six_distinct_valid_configs_per_method(depth):
    search = candidates(depth)
    assert set(search) == {'aperture', 'lora'}
    for method, prefix in [('aperture', 'a'), ('lora', 'l')]:
        options = search[method]
        assert [name for name, _ in options] == [f'{prefix}{i}' for i in range(6)]
        assert len({json.dumps(config, sort_keys=True) for _, config in options}) == 6
    for _, config in search['aperture']:
        assert all(0 <= layer < depth for layer in config['layers'])
        assert len(config['layers']) == len(set(config['layers']))
    assert {(c['lora_lr'], c['lora_passes']) for _, c in search['lora']} == {
        (lr, passes) for lr in (5e-5, 2e-4, 8e-4) for passes in (1, 4)
    }


def test_q25_relative_depth_deduplicates_historical_position():
    search = dict(candidates(28)['aperture'])
    assert search['a0']['layers'] == [14, 27]
    assert search['a1']['layers'] == [14]
    assert search['a2']['layers'] == [7, 21]
    for depth in (36, 34):
        assert dict(candidates(depth)['aperture'])['a2']['layers'] == [depth // 2, depth - 1]


def calibration_records(prefix='a'):
    return [dict(candidate=f'{prefix}{i}', split='calibration', macro_f1=.5, nll=1.)
            for i in range(6)]


@pytest.mark.parametrize('prefix', ['a', 'l'])
def test_selection_prioritizes_f1_then_nll_then_id_independent_of_order(prefix):
    records = calibration_records(prefix)
    records[5].update(macro_f1=.6, nll=20.)
    assert select_rows(records) == f'{prefix}5'
    records[4].update(macro_f1=.6, nll=2.)
    assert select_rows(records) == f'{prefix}4'
    records[3].update(macro_f1=.6, nll=2.)
    assert select_rows(list(reversed(records))) == f'{prefix}3'


@pytest.mark.parametrize('split', ['query', 'support', 'test', None])
def test_rejects_any_noncalibration_record(split):
    records = calibration_records()
    records[5]['split'] = split
    with pytest.raises(ValueError):
        select_rows(records)


@pytest.mark.parametrize('field', ['macro_f1', 'nll'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_rejects_nonfinite_even_for_losing_candidate(field, value):
    records = calibration_records()
    records[5][field] = value
    with pytest.raises(ValueError):
        select_rows(records)


@pytest.mark.parametrize('count', [0, 5, 7])
def test_rejects_incomplete_or_extra_candidates(count):
    records = calibration_records()
    if count == 7:
        records.append(dict(records[0], candidate='a6'))
    else:
        records = records[:count]
    with pytest.raises(ValueError):
        select_rows(records)


@pytest.mark.parametrize('replacement', ['a0', 'a6', 'l5'])
def test_rejects_duplicate_unknown_or_mixed_method_candidates(replacement):
    records = calibration_records()
    records[5]['candidate'] = replacement
    with pytest.raises(ValueError):
        select_rows(records)


@pytest.mark.parametrize('field,value', [('macro_f1', -.01), ('macro_f1', 1.01), ('nll', -.01)])
def test_rejects_out_of_range_metrics(field, value):
    records = calibration_records()
    records[5][field] = value
    with pytest.raises(ValueError):
        select_rows(records)
