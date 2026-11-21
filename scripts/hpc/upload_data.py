#!/usr/bin/env python3
"""Upload repository input datasets with rsync; preserve existing remote files."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys


@dataclass(frozen=True)
class Transfer:
    source: Path
    destination: str


def first_file(*paths):
    return next((path for path in paths if path.is_file()), paths[0])


def transfer_plan(repo, project, upstream):
    transfers = []
    missing = []
    if repo == 'buhito':
        dataset = project / 'data/REDDIT-MULTI-5K'
        for suffix in ['edges', 'graph_idx', 'graph_labels']:
            relative = f'data/REDDIT-MULTI-5K/REDDIT-MULTI-5K.{suffix}'
            source = project / relative
            if source.is_file():
                transfers.append(Transfer(source, relative))
            else:
                missing.append(str(source))
        relative = 'examples/reddit_example/reddit_graphs_benchmarks'
        graphs = sorted((project / relative).glob('*.pkl'))
        transfers.extend(Transfer(path, f'{relative}/{path.name}') for path in graphs)
        if not graphs:
            missing.append(str(project / relative / '*.pkl'))
    else:
        files = {
            'demos/qm9_processed.csv': [project / 'demos/qm9_processed.csv', upstream / 'demos/qm9_processed.csv'],
            'demos/qm9/qm9data/qm9_processed.csv': [project / 'demos/qm9/qm9data/qm9_processed.csv', upstream / 'demos/qm9/qm9data/qm9_processed.csv'],
            'demos/data/lipophilicity_astrazeneca.tab': [project / 'demos/data/lipophilicity_astrazeneca.tab', upstream / 'demos/data/lipophilicity_astrazeneca.tab', upstream / 'data/lipophilicity_astrazeneca.tab'],
        }
        for relative, candidates in files.items():
            source = first_file(*candidates)
            if source.is_file():
                transfers.append(Transfer(source, relative))
            else:
                missing.append(str(source))
        # Include raw QM9 inputs and additional upstream input datasets. Never
        # traverse environments, output/results folders, code, or SSH material.
        for relative in ['demos/qm9_data_files', 'demos/data', 'Datasets', 'Datasets_no_split', 'Datasets_Agg']:
            for source_root in [project / relative, upstream / relative]:
                if not source_root.is_dir():
                    continue
                for path in sorted(source_root.rglob('*')):
                    if path.is_file() and not path.is_symlink() and not any(part.startswith('.') for part in path.relative_to(source_root).parts):
                        transfers.append(Transfer(path, str(PurePosixPath(relative) / path.relative_to(source_root).as_posix())))
        # The original upstream checkout stores the PyTDC cache at its root.
        if (upstream / 'data').is_dir():
            for path in sorted((upstream / 'data').rglob('*')):
                if path.is_file() and not path.is_symlink() and not any(part.startswith('.') for part in path.relative_to(upstream / 'data').parts):
                    transfers.append(Transfer(path, 'demos/data/' + path.relative_to(upstream / 'data').as_posix()))
        present = {item.destination for item in transfers}
        for name in ['dsgdb9nsd.xyz.tar.bz2', 'uncharacterized.txt', 'atomref.txt']:
            if f'demos/qm9_data_files/{name}' not in present:
                missing.append(str(upstream / 'demos/qm9_data_files' / name))
    # Prefer current-checkout files over upstream copies at the same destination.
    unique = {}
    for item in transfers:
        unique.setdefault(item.destination, item)
    return list(unique.values()), missing


def default_ssh():
    release = Path('/proc/sys/kernel/osrelease')
    if release.is_file() and 'microsoft' in release.read_text().lower():
        windows = shutil.which('ssh.exe') or '/mnt/c/Windows/System32/OpenSSH/ssh.exe'
        if Path(windows).is_file():
            return windows
    return 'ssh'


def remote_directory(value):
    value = value.removeprefix('~/').rstrip('/')
    path = PurePosixPath(value)
    if not value or value in {'.', '/', '~'} or '..' in path.parts or '\n' in value or '\r' in value:
        raise argparse.ArgumentTypeError('Choose a dedicated remote repository directory')
    return str(path)


def rsync_command(transfer, args):
    target = str(PurePosixPath(args.remote_dir) / transfer.destination)
    parent = str(PurePosixPath(target).parent)
    remote_rsync = 'rsync' if args.dry_run else f'mkdir -p -- {shlex.quote(parent)} && rsync'
    command = ['rsync', '--archive', '--omit-dir-times', '--ignore-existing',
               '--partial', '--delay-updates', '--itemize-changes', '--human-readable',
               '--protect-args', '-e', args.ssh_command, '--rsync-path', remote_rsync]
    if args.dry_run:
        command.append('--dry-run')
    command += ['--', str(transfer.source), f'{args.remote}:{target}']
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', choices=['buhito', 'minervachem'], required=True)
    parser.add_argument('--source', type=Path, required=True, help='Local repository checkout')
    parser.add_argument('--upstream', type=Path, default=Path.home() / 'minervachem', help='Additional MinervaChem input data source (default: ~/minervachem)')
    parser.add_argument('--remote', default='se', help='SSH host/alias (default: se)')
    parser.add_argument('--remote-dir', type=remote_directory, help='Destination relative to remote home, or absolute (default: repository name)')
    parser.add_argument('--ssh-command', default=None, help='SSH executable/options; auto-detects Windows OpenSSH on WSL')
    parser.add_argument('--dry-run', action='store_true', help='Ask rsync to preview transfers; connects to HPC but creates no directories')
    parser.add_argument('--list', action='store_true', help='List the local upload plan without connecting')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.remote):
        parser.error('Invalid SSH host or alias')
    args.source = args.source.expanduser().resolve()
    args.upstream = args.upstream.expanduser().resolve()
    args.remote_dir = args.remote_dir or args.repo
    args.ssh_command = args.ssh_command or shlex.join([default_ssh(), '-o', 'ConnectTimeout=15'])
    transfers, missing = transfer_plan(args.repo, args.source, args.upstream)
    print(f'Destination: {args.remote}:{args.remote_dir}/ (relative paths start in remote home)', flush=True)
    print('Existing remote files are always skipped; no remote files are deleted.', flush=True)
    for transfer in transfers:
        print(f'{transfer.source} -> {args.remote_dir}/{transfer.destination}', flush=True)
    print(f'{len(transfers)} candidate files, {sum(item.source.stat().st_size for item in transfers):,} local bytes before remote skips', flush=True)
    if missing:
        print('Required local inputs not found (remote copies are left untouched):', file=sys.stderr)
        for path in missing:
            print('  ' + path, file=sys.stderr)
        if args.repo == 'buhito':
            print('See scripts/hpc/README.md and prepare_reddit_data.py for dataset preparation.', file=sys.stderr)
    if args.list:
        return 1 if missing else 0
    if not shutil.which('rsync'):
        parser.error('Install rsync locally and on the HPC server')
    if not transfers:
        return 1
    for transfer in transfers:
        subprocess.run(rsync_command(transfer, args), check=True)
    print('Preview complete.' if args.dry_run else 'Upload complete; existing remote files were preserved.')
    return 1 if missing else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, OSError) as error:
        print(f'Upload failed: {error}', file=sys.stderr)
        sys.exit(1)
