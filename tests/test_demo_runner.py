"""Check demo discovery, notebook output mode, and Slurm submissions offline."""
from pathlib import Path
from types import SimpleNamespace
import json
import shlex

from buhito import demo_runner


def test_all_demos_submit_independently_and_notebooks_run_inplace(tmp_path, monkeypatch):
    root = tmp_path / 'examples'
    root.mkdir()
    (root / 'one.py').write_text('print(1)')
    notebook = root / 'nested' / 'two.ipynb'
    notebook.parent.mkdir()
    notebook.write_text(json.dumps({'cells': [{'cell_type': 'markdown', 'source': ['<!-- HPC_RESOURCES: cpus=4 mem=16G -->']}]}))
    checkpoints = root / '.ipynb_checkpoints'
    checkpoints.mkdir()
    (checkpoints / 'hidden.ipynb').write_text('{}')
    (root / 'old_executed.ipynb').write_text('{}')
    (root / 'old.executed.ipynb').write_text('{}')
    assert set(demo_runner.discover_demos(tmp_path, 'examples')) == {root / 'one.py', notebook}
    command = demo_runner.execution_command(notebook, [])
    assert '--inplace' in command
    assert not any('output-dir' in value or 'executed' in value for value in command)
    submitted = []
    def fake_submit(command, **kwargs):
        submitted.append(command)
        return SimpleNamespace(stdout='123\n', returncode=0)
    monkeypatch.setattr(demo_runner.subprocess, 'run', fake_submit)
    demo_runner.main(['--project', str(tmp_path), '--demo-root', 'examples', '--all', '--slurm'])
    assert len(submitted) == 2
    notebook_job = next(command for command in submitted if str(notebook.relative_to(tmp_path)) in shlex.split(command[-1]))
    assert notebook_job[notebook_job.index('--cpus-per-task') + 1] == '4'
    assert notebook_job[notebook_job.index('--mem') + 1] == '16G'


def test_forward_script_arguments(tmp_path, monkeypatch):
    root = tmp_path / 'demos'
    root.mkdir()
    script = root / 'one.py'
    script.write_text('pass')
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(demo_runner.subprocess, 'run', fake_run)
    demo_runner.main(['--project', str(tmp_path), '--demo-root', 'demos', '--demo', 'demos/one.py', '--', '--data', 'input.csv'])
    assert calls[0][0][-2:] == ['--data', 'input.csv']
    assert calls[0][1]['cwd'] == tmp_path


def test_missing_notebook_resources_stops_all_submissions(tmp_path, monkeypatch):
    import pytest
    root = tmp_path / 'examples'
    root.mkdir()
    for name, source in [('a', '<!-- HPC_RESOURCES: cpus=4 mem=8G -->'), ('b', 'No resources')]:
        (root / f'{name}.ipynb').write_text(json.dumps({'cells': [{'cell_type': 'markdown', 'source': [source]}]}))
    def forbidden(*args, **kwargs):
        raise AssertionError('No jobs should be submitted before all resources validate')
    monkeypatch.setattr(demo_runner.subprocess, 'run', forbidden)
    with pytest.raises(SystemExit, match='2'):
        demo_runner.main(['--project', str(tmp_path), '--demo-root', 'examples', '--all', '--slurm'])


def test_worker_caps_joblib_to_slurm_allocation(tmp_path, monkeypatch):
    root = tmp_path / 'examples'
    root.mkdir()
    (root / 'one.py').write_text('pass')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    captured = []
    def fake_run(command, **kwargs):
        captured.append(kwargs['env'])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(demo_runner.subprocess, 'run', fake_run)
    demo_runner.main(['--project', str(tmp_path), '--demo-root', 'examples', '--all'])
    assert captured[0]['LOKY_MAX_CPU_COUNT'] == '4'
    assert captured[0]['OMP_NUM_THREADS'] == '1'
