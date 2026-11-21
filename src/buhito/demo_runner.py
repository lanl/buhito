"""Run repository demos locally or submit independent Slurm jobs.

This module uses only the standard library. Notebook execution writes outputs
back to the source notebook after successful execution.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def discover_demos(project, demo_root):
    root = (project / demo_root).resolve()
    return sorted(
        path for path in root.rglob('*')
        if path.suffix in {'.py', '.ipynb'} and path.name != '__init__.py'
        and not any(part.startswith('.') for part in path.relative_to(root).parts)
        and not path.stem.endswith(('_executed', '.executed'))
    )


def notebook_resources(path):
    if path.suffix != '.ipynb':
        return None
    notebook = json.loads(path.read_text())
    markdown = next((''.join(cell.get('source', [])) for cell in notebook.get('cells', [])
                     if cell.get('cell_type') == 'markdown'), '')
    match = re.search(r'<!--\s*HPC_RESOURCES:\s*cpus=([1-9]\d*)\s+mem=([1-9]\d*[KMGTP])\s*-->', markdown)
    if match is None:
        raise ValueError(f'{path}: missing or invalid HPC_RESOURCES marker in the first Markdown cell')
    return match[1], match[2]


def execution_command(path, args):
    if path.suffix == '.ipynb':
        if args:
            raise ValueError('Notebook demos do not accept script arguments')
        return [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
                '--inplace', '--ExecutePreprocessor.kernel_name=python3',
                '--ExecutePreprocessor.timeout=-1', str(path)]
    return [sys.executable, str(path), *args]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--demo-root', choices=['demos', 'examples'], required=True)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--demo', help='Path relative to the repository root')
    select.add_argument('--all', action='store_true', help='Run every demo; default inputs must exist')
    select.add_argument('--list', action='store_true')
    parser.add_argument('--slurm', action='store_true', help='Submit one Slurm job per demo')
    parser.add_argument('--cpus', type=int, default=None)
    parser.add_argument('--mem', default=None)
    parser.add_argument('--time', default=os.environ.get('HPC_TIME', '08:00:00'))
    parser.add_argument('--partition', default=os.environ.get('HPC_PARTITION'))
    parser.add_argument('--account', default=os.environ.get('HPC_ACCOUNT'))
    parser.add_argument('--mail-user', default=os.environ.get('MAIL_USER'))
    parser.add_argument('demo_args', nargs=argparse.REMAINDER, help='Arguments following -- are forwarded to the Python demo')
    opts = parser.parse_args(argv)
    project = opts.project.resolve()
    demos = discover_demos(project, opts.demo_root)
    if opts.list:
        for demo in demos:
            print(demo.relative_to(project))
        return
    if opts.demo:
        chosen = (project / opts.demo).resolve()
        if chosen not in demos:
            parser.error('Choose a .py or .ipynb demo under ' + opts.demo_root)
        demos = [chosen]
    args = opts.demo_args
    if args and args[0] == '--':
        args = args[1:]
    if opts.all and args:
        parser.error('Use --demo when forwarding script arguments')
    if opts.cpus is not None and opts.cpus < 1:
        parser.error('--cpus must be positive')
    if opts.mem and not re.fullmatch(r'[1-9]\d*[KMGTP]', opts.mem):
        parser.error('--mem must be a positive size such as 8G')
    # Validate all selections before executing or submitting any jobs.
    commands = [(path, execution_command(path, args)) for path in demos]
    try:
        resources = {path: notebook_resources(path) or ('1', '8G') for path in demos} if opts.slurm else {}
    except ValueError as error:
        parser.error(str(error))
    failures = []
    for path, command in commands:
        relative = path.relative_to(project)
        if opts.slurm:
            cores, memory = resources[path]
            cores = str(opts.cpus or cores)
            memory = opts.mem or memory
            logs = project / 'logs'
            logs.mkdir(exist_ok=True)
            name = str(relative.with_suffix('')).replace('/', '__')
            # Reuse this environment on the compute node without resynchronizing
            # a shared virtualenv in every concurrent job.
            worker = [sys.executable, '-m', 'buhito.demo_runner', '--project', str(project),
                      '--demo-root', opts.demo_root, '--demo', str(relative), '--', *args]
            submit = ['sbatch', '--parsable', '--chdir', str(project), '--job-name', name,
                      '--cpus-per-task', cores, '--mem', memory, '--time', opts.time,
                      '--output', str(logs / (name + '-%j.out')),
                      '--error', str(logs / (name + '-%j.err'))]
            for flag, value in [('partition', opts.partition), ('account', opts.account),
                                ('mail-user', opts.mail_user)]:
                if value:
                    submit += ['--' + flag, value]
            if opts.mail_user:
                submit += ['--mail-type', 'BEGIN,END,FAIL']
            submit += ['--wrap', shlex.join(worker)]
            result = subprocess.run(submit, check=True, capture_output=True, text=True)
            print(f'Submitted {relative}: {result.stdout.strip()}', flush=True)
        else:
            print(f'Running {relative}', flush=True)
            env = os.environ.copy()
            env['PYTHONNOUSERSITE'] = '1'
            if 'SLURM_CPUS_PER_TASK' in env:
                env['LOKY_MAX_CPU_COUNT'] = env['SLURM_CPUS_PER_TASK']
            for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                env[key] = '1'
            if path.suffix == '.ipynb':
                env['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
            result = subprocess.run(command, cwd=path.parent if path.suffix == '.ipynb' else project, env=env)
            if result.returncode:
                failures.append(str(relative))
    if failures:
        parser.exit(1, 'Failed demos: ' + ', '.join(failures) + '\n')


if __name__ == '__main__':
    main()
