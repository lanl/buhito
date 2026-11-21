"""Exercise real repository shell scripts with local uv/sbatch stand-ins."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

BUHITO = Path(__file__).resolve().parents[1]


def repository(name):
    root = BUHITO if name == 'buhito' else BUHITO.parent / 'minervachem'
    if not (root / 'scripts').is_dir():
        pytest.skip('Sibling MinervaChem checkout is unavailable')
    return root


def setup_project(tmp_path, name):
    source = repository(name)
    folder = 'examples' if name == 'buhito' else 'demos'
    project = tmp_path / name
    shutil.copytree(source / 'scripts', project / 'scripts')
    expected = {}
    for notebook in (source / folder).rglob('*.ipynb'):
        if any(part.startswith('.') for part in notebook.relative_to(source).parts):
            continue
        if notebook.stem.endswith(('_executed', '.executed')):
            continue
        data = json.loads(notebook.read_text())
        markdown = next(cell for cell in data['cells'] if cell['cell_type'] == 'markdown')
        marker = re.search(r'HPC_RESOURCES:\s*cpus=(\d+)\s+mem=(\d+[KMGTP])', ''.join(markdown['source']))
        assert marker, f'{notebook} needs an explicit resource configuration'
        relative = notebook.relative_to(source)
        expected[str(relative)] = marker.groups()
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({'cells': [markdown]}))
    (project / '.env').write_text('MAIL_USER=test@example.invalid\n')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    fake_uv = bindir / 'uv'
    fake_uv.write_text('#!/usr/bin/env bash\nset -euo pipefail\nwhile [[ "$1" != "python" ]]; do shift; done\nshift\nexec "$DEMO_TEST_PYTHON" "$@"\n')
    fake_sbatch = bindir / 'sbatch'
    fake_sbatch.write_text('#!' + sys.executable + '\nimport json, os, sys\nwith open(os.environ["DEMO_TEST_SUBMISSIONS"], "a") as handle:\n    handle.write(json.dumps(sys.argv[1:]) + "\\n")\nprint("123")\n')
    fake_scontrol = bindir / 'scontrol'
    fake_scontrol.write_text('#!/usr/bin/env bash\necho "MailUser=test@example.invalid MailType=BEGIN,END,FAIL"\n')
    for path in [fake_uv, fake_sbatch, fake_scontrol]:
        path.chmod(0o755)
    record = tmp_path / 'submissions.jsonl'
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               PYTHONPATH=str(BUHITO / 'src'), DEMO_TEST_PYTHON=sys.executable,
               DEMO_TEST_SUBMISSIONS=str(record))
    for key in list(env):
        if key.startswith('SLURM_'):
            del env[key]
    return project, expected, record, env


def assert_submissions(record, expected, legacy=False):
    submitted = [json.loads(line) for line in record.read_text().splitlines()]
    actual = {}
    for command in submitted:
        if legacy:
            exports = next(arg for arg in command if arg.startswith('--export='))
            path = exports.split('MINERVACHEM_NOTEBOOK=', 1)[1]
            root = exports.split('MINERVACHEM_PROJECT_DIR=', 1)[1].split(',', 1)[0]
            relative = str(Path(path).relative_to(root))
            cores = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--cpus-per-task='))
            memory = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--mem='))
        else:
            worker = shlex.split(command[command.index('--wrap') + 1])
            relative = worker[worker.index('--demo') + 1]
            cores = command[command.index('--cpus-per-task') + 1]
            memory = command[command.index('--mem') + 1]
        assert relative not in actual, f'Duplicate job for {relative}'
        actual[relative] = (cores, memory)
    assert actual == expected


@pytest.mark.parametrize('name', ['buhito', 'minervachem'])
def test_all_notebooks_via_repository_all_demo_script(tmp_path, name):
    project, expected, record, env = setup_project(tmp_path, name)
    assert expected
    result = subprocess.run(['bash', str(project / 'scripts/run_all_demos.sh'), '--slurm'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_submissions(record, expected)


def test_legacy_all_notebooks_uses_each_notebooks_resources(tmp_path):
    project, expected, record, env = setup_project(tmp_path, 'minervachem')
    result = subprocess.run(['bash', str(project / 'scripts/run_demo_notebooks.sh')],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_submissions(record, expected, legacy=True)


def test_legacy_single_notebook_uses_each_notebooks_resources(tmp_path):
    project, expected, record, env = setup_project(tmp_path, 'minervachem')
    for relative in expected:
        result = subprocess.run(['bash', str(project / 'scripts/run_demo_notebook.sh'), '--notebook', relative],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    assert_submissions(record, expected, legacy=True)
