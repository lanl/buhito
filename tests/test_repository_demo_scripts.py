"""Exercise real repository shell scripts with local uv/sbatch stand-ins."""
import json
import os
from pathlib import Path
import re
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
    runner = tmp_path / 'buhito/src/buhito/demo_runner.py'
    runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BUHITO / 'src/buhito/demo_runner.py', runner)
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
            relative = command[command.index('--demo') + 1]
            cores = command[command.index('--cpus-per-task') + 1]
            memory = command[command.index('--mem') + 1]
        assert relative not in actual, f'Duplicate job for {relative}'
        actual[relative] = (cores, memory)
    assert actual == expected


@pytest.mark.parametrize('name', ['buhito', 'minervachem'])
@pytest.mark.parametrize('flags', [[], ['--slurm']])
def test_all_notebooks_via_repository_all_demo_script(tmp_path, name, flags):
    project, expected, record, env = setup_project(tmp_path, name)
    assert expected
    folder = 'examples' if name == 'buhito' else 'demos'
    script = project / folder / 'must_only_run_on_compute_node.py'
    sentinel = project / 'demo-executed'
    script.write_text(f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n')
    expected[str(script.relative_to(project))] = ('1', '8G')
    result = subprocess.run(['bash', str(project / 'scripts/run_all_demos.sh'), *flags],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not sentinel.exists()
    assert_submissions(record, expected)


@pytest.mark.parametrize('name', ['buhito', 'minervachem'])
def test_frontend_uses_uv_to_submit_without_importing_demo_packages(tmp_path, name):
    project, expected, record, env = setup_project(tmp_path, name)
    uv = tmp_path / 'bin/uv'
    uv.write_text(uv.read_text().replace('set -euo pipefail\n', 'set -euo pipefail\ntouch "$DEMO_TEST_UV_CALLED"\n'))
    marker = tmp_path / 'uv-called'
    env['DEMO_TEST_UV_CALLED'] = str(marker)
    (tmp_path / 'buhito/src/buhito/__init__.py').write_text('raise AssertionError("frontend imported demo packages")\n')
    result = subprocess.run(['bash', str(project / 'scripts/run_all_demos.sh')],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists()
    assert_submissions(record, expected)


@pytest.mark.parametrize('name', ['buhito', 'minervachem'])
def test_single_demo_submits_by_default(tmp_path, name):
    project, expected, record, env = setup_project(tmp_path, name)
    relative = next(iter(expected))
    result = subprocess.run(['bash', str(project / 'scripts/run_demo.sh'), '--demo', relative],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_submissions(record, {relative: expected[relative]})


@pytest.mark.parametrize('all_demos,overrides,resources', [
    (True, [], ('32', '64G')),
    (False, [], ('32', '64G')),
    (False, ['--cpus', '16', '--mem', '32G'], ('16', '32G')),
])
def test_reddit_demo_requests_parallel_resources(tmp_path, all_demos, overrides, resources):
    project, expected, record, env = setup_project(tmp_path, 'buhito')
    relative = 'examples/reddit_example/reddit_graphlets_example.py'
    target = project / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BUHITO / relative, target)
    if all_demos:
        command = ['bash', str(project / 'scripts/run_all_demos.sh')]
        expected[relative] = resources
    else:
        command = ['bash', str(project / 'scripts/run_demo.sh'), '--demo', relative]
        expected = {relative: resources}
    result = subprocess.run([*command, *overrides], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_submissions(record, expected)


@pytest.mark.parametrize('name', ['buhito', 'minervachem'])
def test_failed_submission_never_executes_demo(tmp_path, name):
    project, _, record, env = setup_project(tmp_path, name)
    folder = 'examples' if name == 'buhito' else 'demos'
    script = project / folder / 'must_only_run_on_compute_node.py'
    sentinel = project / 'demo-executed'
    script.write_text(f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n')
    (tmp_path / 'bin/sbatch').write_text('#!/usr/bin/env bash\nexit 42\n')
    result = subprocess.run(['bash', str(project / 'scripts/run_all_demos.sh')],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert not record.exists()
    assert not sentinel.exists()


@pytest.mark.parametrize('name', ['buhito', 'minervachem'])
def test_python_demo_executes_only_in_batch_worker(tmp_path, name):
    project, _, record, env = setup_project(tmp_path, name)
    folder = 'examples' if name == 'buhito' else 'demos'
    script = project / folder / 'python_demo.py'
    sentinel = project / 'worker-result.json'
    script.write_text('import json, os, sys\nfrom pathlib import Path\n'
                      f'Path({str(sentinel)!r}).write_text(json.dumps({{'
                      '"job": os.environ["SLURM_JOB_ID"], "args": sys.argv[1:], '
                      '"cpus": os.environ["LOKY_MAX_CPU_COUNT"], '
                      '"runtime": os.environ["JUPYTER_RUNTIME_DIR"]}))\n')
    result = subprocess.run(['bash', str(project / 'scripts/run_demo.sh'), '--demo',
                             str(script.relative_to(project)), '--', '--data', 'input with spaces.csv'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not sentinel.exists()
    command = json.loads(record.read_text())
    worker_index = command.index(str(project / 'scripts/demo_job.sh'))
    worker = ['bash', *command[worker_index:]]
    rejected = subprocess.run(worker, env=env, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    assert not sentinel.exists()
    env.update(SLURM_JOB_ID='123', SLURM_CPUS_PER_TASK='4', SLURM_TMPDIR=str(tmp_path / 'scratch'))
    result = subprocess.run(worker, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(sentinel.read_text())
    assert outcome['job'] == '123'
    assert outcome['args'] == ['--data', 'input with spaces.csv']
    assert outcome['cpus'] == '4'
    assert Path(outcome['runtime']).is_relative_to(tmp_path / 'scratch')


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
