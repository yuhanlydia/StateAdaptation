"""Incomplete experiments must never become exported or pushed results."""
import pytest

from aperture_tuning import publish


def test_incomplete_results_abort_before_creating_export_directory(tmp_path, monkeypatch):
    def incomplete(root):
        raise FileNotFoundError('missing selected evaluation DONE.json')
    monkeypatch.setattr(publish, 'summarize', incomplete)
    destination = tmp_path/'repository/results/aperture_tuning_v1'
    with pytest.raises(FileNotFoundError, match='missing selected evaluation'):
        publish.export(tmp_path/'run', destination)
    assert not destination.exists()


def test_push_path_never_adds_commits_or_pushes_incomplete_results(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(publish.subprocess, 'run', lambda command, **kw: commands.append(command))
    monkeypatch.setattr(publish.subprocess, 'check_output',
                        lambda *a, **kw: 'aperture-support-tuning-20260922\n')
    def incomplete(root):
        raise ValueError('incomplete result seal')
    monkeypatch.setattr(publish, 'summarize', incomplete)
    monkeypatch.setattr('sys.argv', ['publish', '--root', str(tmp_path/'run'),
                                   '--repository', str(tmp_path/'repo'), '--push'])
    with pytest.raises(ValueError, match='incomplete result seal'):
        publish.main()
    assert commands == [['git', 'diff', '--quiet'], ['git', 'diff', '--cached', '--quiet']]
    assert not (tmp_path/'repo/results/aperture_tuning_v1').exists()
