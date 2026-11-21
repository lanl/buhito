"""Offline tests: synthetic Reddit archive and upload command construction."""
import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import download_reddit_data as download
import prepare_reddit_data as preparation
import upload_data


class DataToolsTests(unittest.TestCase):
    def test_download_preview_never_connects_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'missing.zip'
            with patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
                with redirect_stdout(io.StringIO()) as output:
                    download.main(['--dry-run', '--archive', str(archive)])
            self.assertIn(download.DOWNLOAD_URL, output.getvalue())
            self.assertFalse(archive.exists())

    def test_official_zip_prepares_classification_and_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'reddit.zip'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('REDDIT-MULTI-5K/REDDIT-MULTI-5K_A.txt', '1, 2\n2, 1\n3, 4\n4, 3\n')
                handle.writestr('REDDIT-MULTI-5K/REDDIT-MULTI-5K_graph_indicator.txt', '1\n1\n2\n2\n3\n')
                handle.writestr('REDDIT-MULTI-5K/REDDIT-MULTI-5K_graph_labels.txt', '1\n2\n3\n')
                handle.writestr('../never_extract.txt', 'unrelated')
            data = root / 'classification'
            graphs = root / 'benchmark'
            preparation.prepare(archive, data, graphs)
            self.assertEqual(len(list(data.iterdir())), 3)
            self.assertFalse((root / 'never_extract.txt').exists())
            self.assertEqual(len(list(graphs.glob('*.pkl'))), 2)
            with (graphs / 'reddit_graph_0000.pkl').open('rb') as handle:
                graph = pickle.load(handle)
            self.assertEqual(set(graph), {0, 1})
            self.assertEqual(graph.number_of_edges(), 1)
            self.assertEqual(graph.nodes[0]['node_label'], 1)
            self.assertEqual(graph.edges[0, 1]['edge_label'], 'link')
            original = (graphs / 'reddit_graph_0000.pkl').read_bytes()
            preparation.prepare(archive, data, graphs)
            self.assertEqual(original, (graphs / 'reddit_graph_0000.pkl').read_bytes())

    def test_upload_never_replaces_or_deletes_remote_files(self):
        args = argparse.Namespace(remote='se', remote_dir='minervachem', dry_run=False, ssh_command='ssh.exe')
        transfer = upload_data.Transfer(Path('/local/qm9.csv'), 'demos/qm9_processed.csv')
        command = upload_data.rsync_command(transfer, args)
        self.assertIn('--ignore-existing', command)
        self.assertFalse(any(flag.startswith('--delete') for flag in command))
        self.assertEqual(command[-1], 'se:minervachem/demos/qm9_processed.csv')
        args.dry_run = True
        command = upload_data.rsync_command(transfer, args)
        self.assertIn('--dry-run', command)
        self.assertEqual(command[command.index('--rsync-path') + 1], 'rsync')

    def test_upstream_data_mapping_and_current_checkout_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, upstream = root / 'project', root / 'upstream'
            for path, text in [(project / 'demos/qm9_processed.csv', 'current'),
                               (upstream / 'demos/qm9_processed.csv', 'old'),
                               (project / 'demos/qm9/qm9data/qm9_processed.csv', 'benchmark'),
                               (upstream / 'data/lipophilicity_astrazeneca.tab', 'tdc'),
                               (upstream / 'Datasets/one.csv', 'additional')]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            for name in ['dsgdb9nsd.xyz.tar.bz2', 'atomref.txt', 'uncharacterized.txt']:
                path = upstream / 'demos/qm9_data_files' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('raw')
            transfers, missing = upload_data.transfer_plan('minervachem', project, upstream)
            self.assertFalse(missing)
            mapping = {item.destination: item.source for item in transfers}
            self.assertEqual(mapping['demos/qm9_processed.csv'].read_text(), 'current')
            self.assertEqual(mapping['demos/data/lipophilicity_astrazeneca.tab'].read_text(), 'tdc')
            self.assertIn('Datasets/one.csv', mapping)
            self.assertEqual(len(mapping), len(transfers))


if __name__ == '__main__':
    unittest.main()
