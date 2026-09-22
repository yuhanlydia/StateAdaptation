"""Mask isolation must cover the requested sites, including single-layer lenses."""
import pytest

from aperture_table.audit import visual_mask_isolation


def observations(sites):
    return {
        f'{layer}:{kind}': {'outside_max_abs': 0., 'inside_norm': 1.}
        for layer, kind, _ in sites
    }


@pytest.mark.parametrize('layers', [(14,), (14, 27)])
def test_exact_selected_sites_pass(layers):
    sites = [(layer, kind, None) for layer in layers for kind in ('o_proj', 'down_proj')]
    assert visual_mask_isolation(observations(sites), sites)


def test_missing_site_fails():
    sites = [(14, 'o_proj', None), (14, 'down_proj', None)]
    direct = observations(sites)
    del direct['14:down_proj']
    assert not visual_mask_isolation(direct, sites)


def test_wrong_site_fails_even_with_four_observations():
    sites = [(layer, kind, None) for layer in (14, 27) for kind in ('o_proj', 'down_proj')]
    direct = observations(sites)
    direct['26:down_proj'] = direct.pop('27:down_proj')
    assert not visual_mask_isolation(direct, sites)


def test_extra_site_fails():
    sites = [(14, 'o_proj', None), (14, 'down_proj', None)]
    direct = observations(sites)
    direct['27:o_proj'] = {'outside_max_abs': 0., 'inside_norm': 1.}
    assert not visual_mask_isolation(direct, sites)


@pytest.mark.parametrize('leakage', [1e-30, float('nan'), float('inf')])
def test_nonzero_or_nonfinite_outside_fails(leakage):
    sites = [(14, 'o_proj', None), (14, 'down_proj', None)]
    direct = observations(sites)
    direct['14:down_proj']['outside_max_abs'] = leakage
    assert not visual_mask_isolation(direct, sites)


def test_empty_sites_fail():
    assert not visual_mask_isolation({}, [])
