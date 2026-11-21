#!/usr/bin/env python3
"""Download REDDIT-MULTI-5K from its official TUDataset link and prepare demos.

Run from Buhito: uv run python scripts/hpc/download_reddit_data.py
Use --dry-run to print the exact URL and destinations without network access.
"""
import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

from prepare_reddit_data import DATASET, PROJECT, prepare

DOWNLOAD_URL = 'https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-5K.zip'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=PROJECT / 'data' / f'{DATASET}.zip', help='Archive cache; an existing ZIP is reused without downloading')
    parser.add_argument('--output', type=Path, default=PROJECT / 'data' / DATASET)
    parser.add_argument('--benchmark-dir', type=Path, default=PROJECT / 'examples/reddit_example/reddit_graphs_benchmarks')
    parser.add_argument('--benchmark-count', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true', help='Print the URL and destinations only; no downloads or file changes')
    args = parser.parse_args(argv)
    if args.benchmark_count < 1:
        parser.error('--benchmark-count must be positive')
    print(f'Download URL: {DOWNLOAD_URL}')
    print(f'Archive: {args.archive}\nClassification data: {args.output}\nBenchmark graphs: {args.benchmark_dir}')
    if args.dry_run:
        return
    # Check the preparation dependency before starting the download.
    import networkx  # noqa: F401
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.archive.with_name(args.archive.name + '.part')
        try:
            with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as response:
                print(f'Resolved URL: {response.geturl()}')
                with temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
            if not zipfile.is_zipfile(temporary):
                raise ValueError('Downloaded response is not a ZIP archive')
            temporary.replace(args.archive)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        print('Using existing local archive; no download')
    prepare(args.archive, args.output, args.benchmark_dir, args.benchmark_count)


if __name__ == '__main__':
    main()
