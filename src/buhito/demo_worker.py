"""Execute a single demo inside a Slurm batch job."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from .demo_runner import discover_demos


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
    parser.add_argument('--demo', required=True)
    parser.add_argument('demo_args', nargs=argparse.REMAINDER)
    opts = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Demo execution requires a Slurm allocation')
    project = opts.project.resolve()
    path = (project / opts.demo).resolve()
    if path not in discover_demos(project, opts.demo_root):
        parser.error('Choose a .py or .ipynb demo under ' + opts.demo_root)
    args = opts.demo_args
    if args and args[0] == '--':
        args = args[1:]
    command = execution_command(path, args)
    relative = path.relative_to(project)
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
        parser.exit(result.returncode, 'Failed demo: ' + str(relative) + '\n')


if __name__ == '__main__':
    main()
