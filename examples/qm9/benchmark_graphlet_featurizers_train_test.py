
#!/usr/bin/env python3
"""
Benchmark BFS vs DFS graphlet featurizers in buhito on a QM9 dataset.

[***] How does this benchmark work? [***]
1) Loads a CSV (~/buhito/examples/qm9/qm9data/qm9_processed.csv)
2) Converts SMILES to NetworkX graphs once (can be cached to disk)
3) For each max_len (max graphlet size) and each algorithm (BFS/DFS):
   - runs `fit_transform` 
   - repeats the measurement N times
4) Writes:
   - benchmark_results.csv (one row per run)
   - system_info.json
   - benchmark.log

[***] Usage example [***]
python benchmark_graphlet_featurizers.py \
  --data qm9_processed.csv \
  --max-len 2 3 4 5 6 \
  --repeats 3 \
  --n-jobs -1 \
  --outdir bench_qm9

If you already have cached graphs:
python benchmark_graphlet_featurizers.py --data qm9_processed.csv --graph-cache qm9_graphs.pkl --max-len 5

- --n-jobs is passed into GraphletTransformer, 1 for single-core, -1 for all cores.
- The train/test split is created once and reused for all runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import tracemalloc
except Exception:
    tracemalloc = None


def _safe_import_buhito():
    try:
        from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer
        from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer
        from buhito.transformers import GraphletTransformer
        from buhito.converters import smiles_to_nx
    except Exception as e:
        raise RuntimeError(
            "Failed to import buhito. Make sure buhito is installed."
        ) from e

    return BFSGraphletFeaturizer, DFSGraphletFeaturizer, GraphletTransformer, smiles_to_nx


def _now_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging(outdir, level="INFO"):
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"benchmark_{_now_tag()}.log"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers[:] = [] 

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info("Logging to %s", log_path)
    return log_path


def _pkg_version(name):
    try:
        import importlib.metadata as im

        return im.version(name)
    except Exception:
        return None


def _system_info(extra=None):
    info: Dict[str, Any] = {
        "timestamp_local": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "networkx": _pkg_version("networkx"),
        "rdkit": _pkg_version("rdkit"),
        "joblib": _pkg_version("joblib"),
        "buhito": _pkg_version("buhito"),
    }
    if extra:
        info.update(extra)
    return info


# def _sha1_of_smiles(smiles: Sequence[str], max_items: int = 50_000) -> str:
#     """
#     A fingerprint of the dataset ordering/content (helps detect cache mismatch).
#     Uses up to max_items smiles strings to avoid huge hashing time on very large datasets.
#     """
#     h = hashlib.sha1()
#     n = min(len(smiles), max_items)
#     for s in smiles[:n]:
#         h.update(str(s).encode("utf-8", errors="ignore"))
#         h.update(b"\n")
#     h.update(f"__n={len(smiles)}__".encode())
#     return h.hexdigest()


def _load_dataset(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"CSV must contain 'smiles'.")
    return df


def _datalimit_and_sample(df, limit, sample, random_state):
    if limit is not None:
        df = df.head(int(limit)).copy()
    if sample is not None:
        sample = int(sample)
        if sample > len(df):
            raise ValueError(f"[***]Sample {sample} is larger than dataset size {len(df)}[***]")
        df = df.sample(n=sample, random_state=random_state).copy()
    df = df.reset_index(drop=True)
    return df


def _convert_smiles_to_graphs(smiles, *, add_hs, output_2d_pos):
    _, _, _, smiles_to_nx = _safe_import_buhito()

    graphs = []
    t0 = time.perf_counter()
    for i, s in enumerate(smiles):
        out = smiles_to_nx(s, add_hs=add_hs, output_2d_pos=output_2d_pos)
        G = out[0] if isinstance(out, (tuple, list)) else out
        graphs.append(G)

        if (i + 1) % 5000 == 0:
            logging.info("Converted %d/%d SMILES -> graphs (%.1fs elapsed)", i + 1, len(smiles), time.perf_counter() - t0)

    logging.info("Finished. Converted %d SMILES -> graphs in %.3fs", len(smiles), time.perf_counter() - t0)
    return graphs


def _load_or_build_graphs(df, cache_path, *, add_hs, output_2d_pos, force_recompute):
    smiles = df["smiles"].astype(str).tolist()

    meta = {
        "n_smiles": len(smiles),
        "add_hs": add_hs,
        "output_2d_pos": output_2d_pos,
    }

    if cache_path is None:
        graphs = _convert_smiles_to_graphs(smiles, add_hs=add_hs, output_2d_pos=output_2d_pos)
        return graphs, meta

    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")

    if cache_path.exists() and meta_path.exists() and not force_recompute:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            # if cached_meta.get("smiles_sha1") != smiles_sha1:
            #     logging.warning("Graph cache exists but smiles hash mismatches. Recomputing graphs.")
            # else:
                logging.info("Loading graphs from cache: %s", cache_path)
                import pickle

                with cache_path.open("rb") as f:
                    graphs = pickle.load(f)
                if len(graphs) != len(smiles):
                    logging.warning("Cached graphs length mismatch (%d != %d). Recomputing.", len(graphs), len(smiles))
                else:
                    return graphs, cached_meta
        except Exception as e:
            logging.warning("Failed to load cache (%s). Recomputing. Reason: %s", cache_path, e)

    graphs = _convert_smiles_to_graphs(smiles, add_hs=add_hs, output_2d_pos=output_2d_pos)

    try:
        import pickle

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        logging.info("Saved graphs cache to %s (+ meta %s)", cache_path, meta_path)
    except Exception as e:
        logging.warning("Could not save graph cache to %s. Reason: %s", cache_path, e)

    return graphs, meta


def _train_test_split_indices(n, train_size, random_state):
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(train_size * n))
    n_train = max(1, min(n - 1, n_train)) if n > 1 else n
    train_idx = np.sort(idx[:n_train])
    test_idx = np.sort(idx[n_train:])
    return train_idx, test_idx


def _sparse_info(X):
    info: Dict[str, Any] = {}
    shape = getattr(X, "shape", None)
    if shape is not None:
        info["X_shape"] = tuple(int(x) for x in shape)
    if hasattr(X, "nnz"):
        nnz = int(X.nnz)
        info["X_nnz"] = nnz
        if shape and len(shape) == 2 and shape[0] * shape[1] > 0:
            info["X_density"] = float(nnz / (shape[0] * shape[1]))
    return info


def _extract_transformer_info(transformer):
    """
    Pull whatever useful info is available across versions.
    """
    d: Dict[str, Any] = {}
    for attr in [
        "n_features_",
        "n_bits_",
        "bit_ids_",
        "bit_indices_",
        "feature_names_",
        "vocabulary_",
    ]:
        if hasattr(transformer, attr):
            try:
                v = getattr(transformer, attr)
                if isinstance(v, (list, tuple)):
                    d[attr] = len(v)
                elif isinstance(v, dict):
                    d[attr] = len(v)
                else:
                    d[attr] = v
            except Exception:
                pass
    if hasattr(transformer, "featurizer"):
        f = getattr(transformer, "featurizer")
        for attr in ["max_len", "max_len_", "return_nodewise", "return_nodewise_"]:
            if hasattr(f, attr):
                try:
                    d[f"featurizer.{attr}"] = getattr(f, attr)
                except Exception:
                    pass
    return d


@dataclasses.dataclass
class RunRecord:
    run_id: str
    algorithm: str
    max_len: int
    repeat: int
    n_graphs_train: int
    n_graphs_test: int
    n_jobs: int
    chunk_size: Any
    wall_fit_transform_s: float
    wall_transform_s: float
    wall_total_s: float
    cpu_total_s: float
    peak_tracemalloc_mb: Optional[float] = None
    X_train_shape: Optional[Tuple[int, int]] = None
    X_train_nnz: Optional[int] = None
    X_test_shape: Optional[Tuple[int, int]] = None
    X_test_nnz: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


def _build_transformer(algorithm: str, max_len: int, n_jobs: int, chunk_size: Any, verbose: int):
    BFSGraphletFeaturizer, DFSGraphletFeaturizer, GraphletTransformer, _ = _safe_import_buhito()

    if algorithm.lower() == "bfs":
        featurizer = BFSGraphletFeaturizer(return_nodewise=False, max_len=max_len)
    elif algorithm.lower() == "dfs":
        try:
            featurizer = DFSGraphletFeaturizer(return_nodewise=False, max_len=max_len)
        except TypeError:
            featurizer = DFSGraphletFeaturizer(max_len=max_len)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    transformer = GraphletTransformer(
        featurizer=featurizer,
        verbose=verbose,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    return transformer


def _bench_one(
    *,
    graphs: Sequence[Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    algorithm: str,
    max_len: int,
    repeat: int,
    n_jobs: int,
    chunk_size: Any,
    verbose: int,
    warmup: bool,
    track_tracemalloc: bool,
) -> RunRecord:
    run_id = f"{algorithm}_L{max_len}_r{repeat}_{_now_tag()}"

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx] if len(test_idx) else []

    transformer = _build_transformer(algorithm, max_len, n_jobs=n_jobs, chunk_size=chunk_size, verbose=verbose)

    # warm-up run not measured
    if warmup:
        try:
            _ = transformer.fit_transform(train_graphs[: min(200, len(train_graphs))])
        except Exception:
            pass
        gc.collect()

    if tracemalloc and track_tracemalloc:
        tracemalloc.start()

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    # FIT_TRANSFORM
    t1 = time.perf_counter()
    X_train = transformer.fit_transform(train_graphs)
    t2 = time.perf_counter()

    # TRANSFORM 
    if test_graphs:
        X_test = transformer.transform(test_graphs)
    else:
        X_test = None
    t3 = time.perf_counter()

    cpu_total = time.process_time() - t0_cpu
    wall_fit = t2 - t1
    wall_transform = (t3 - t2) if test_graphs else 0.0
    wall_total = time.perf_counter() - t0_wall

    peak_mb = None
    if tracemalloc and track_tracemalloc:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = float(peak) / (1024 ** 2)

    rec = RunRecord(
        run_id=run_id,
        algorithm=algorithm,
        max_len=int(max_len),
        repeat=int(repeat),
        n_graphs_train=len(train_graphs),
        n_graphs_test=len(test_graphs),
        n_jobs=int(n_jobs),
        chunk_size=chunk_size,
        wall_fit_transform_s=float(wall_fit),
        wall_transform_s=float(wall_transform),
        wall_total_s=float(wall_total),
        cpu_total_s=float(cpu_total),
        peak_tracemalloc_mb=peak_mb,
    )

    try:
        rec.extra = _extract_transformer_info(transformer)
    except Exception:
        rec.extra = {}

    try:
        tr_info = _sparse_info(X_train)
        rec.X_train_shape = tr_info.get("X_shape")
        rec.X_train_nnz = tr_info.get("X_nnz")
    except Exception:
        pass

    try:
        if X_test is not None:
            te_info = _sparse_info(X_test)
            rec.X_test_shape = te_info.get("X_shape")
            rec.X_test_nnz = te_info.get("X_nnz")
    except Exception:
        pass

    del X_train
    if X_test is not None:
        del X_test
    del transformer
    gc.collect()

    return rec


def run_benchmark(
    *,
    data_csv: Path,
    outdir: Path,
    max_lens: Sequence[int],
    repeats: int,
    train_size: float,
    random_state: int,
    n_jobs: int,
    chunk_size: Any,
    verbose: int,
    log_level: str,
    graph_cache: Optional[Path],
    force_recompute_graphs: bool,
    limit: Optional[int],
    sample: Optional[int],
    add_hs: bool,
    output_2d_pos: bool,
    warmup: bool,
    track_tracemalloc: bool,
) -> Path:
    log_path = _setup_logging(outdir, level=log_level)

    logging.info("Benchmark starting.")
    logging.info("Args: %s", {
        "data_csv": str(data_csv),
        "outdir": str(outdir),
        "max_lens": list(max_lens),
        "repeats": repeats,
        "train_size": train_size,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "chunk_size": chunk_size,
        "verbose": verbose,
        "graph_cache": str(graph_cache) if graph_cache else None,
        "force_recompute_graphs": force_recompute_graphs,
        "limit": limit,
        "sample": sample,
        "add_hs": add_hs,
        "output_2d_pos": output_2d_pos,
        "warmup": warmup,
        "track_tracemalloc": track_tracemalloc and bool(tracemalloc),
    })

    sysinfo = _system_info(extra={"command": " ".join(sys.argv)})
    (outdir / "system_info.json").write_text(json.dumps(sysinfo, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote system info to %s", outdir / "system_info.json")

    df = _load_dataset(Path(data_csv))
    df = _datalimit_and_sample(df, limit=limit, sample=sample, random_state=random_state)
    logging.info("Loaded dataset: %d rows, columns=%s", len(df), list(df.columns)[:15])

    graphs, graph_meta = _load_or_build_graphs(
        df,
        cache_path=graph_cache,
        add_hs=add_hs,
        output_2d_pos=output_2d_pos,
        force_recompute=force_recompute_graphs,
    )
    (outdir / "graph_cache_meta.json").write_text(json.dumps(graph_meta, indent=2, sort_keys=True), encoding="utf-8")

    if len(graphs) < 2 or train_size >= 0.999:
        train_idx = np.arange(len(graphs))
        test_idx = np.array([], dtype=int)
        logging.info("Using all graphs for fit_transform (no test split). n=%d", len(graphs))
    else:
        train_idx, test_idx = _train_test_split_indices(len(graphs), train_size=train_size, random_state=random_state)
        logging.info(
            "Train/test split: train=%d (%.1f%%), test=%d (%.1f%%)",
            len(train_idx), 100 * len(train_idx) / len(graphs),
            len(test_idx), 100 * len(test_idx) / len(graphs),
        )

    records = []
    algorithms = ["bfs", "dfs"]

    for max_len in max_lens:
        for alg in algorithms:
            for r in range(repeats):
                logging.info("RUN | alg=%s max_len=%s repeat=%d/%d", alg, max_len, r + 1, repeats)
                try:
                    rec = _bench_one(
                        graphs=graphs,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        algorithm=alg,
                        max_len=int(max_len),
                        repeat=r,
                        n_jobs=int(n_jobs),
                        chunk_size=chunk_size,
                        verbose=int(verbose),
                        warmup=warmup,
                        track_tracemalloc=track_tracemalloc and bool(tracemalloc),
                    )
                    row = dataclasses.asdict(rec)
                    row["extra_json"] = json.dumps(row.pop("extra") or {}, sort_keys=True)
                    records.append(row)

                    logging.info(
                        "DONE | %s | fit_transform=%.3fs transform=%.3fs total=%.3fs peak_mem=%.1fMB X_train=%s",
                        rec.run_id,
                        rec.wall_fit_transform_s,
                        rec.wall_transform_s,
                        rec.wall_total_s,
                        rec.peak_tracemalloc_mb if rec.peak_tracemalloc_mb is not None else float("nan"),
                        rec.X_train_shape,
                    )
                except Exception as e:
                    logging.exception("FAILED | alg=%s max_len=%s repeat=%d. Error: %s", alg, max_len, r, e)
                    records.append(
                        {
                            "run_id": f"{alg}_L{max_len}_r{r}_{_now_tag()}",
                            "algorithm": alg,
                            "max_len": int(max_len),
                            "repeat": int(r),
                            "n_graphs_train": int(len(train_idx)),
                            "n_graphs_test": int(len(test_idx)),
                            "n_jobs": int(n_jobs),
                            "chunk_size": chunk_size,
                            "wall_fit_transform_s": np.nan,
                            "wall_transform_s": np.nan,
                            "wall_total_s": np.nan,
                            "cpu_total_s": np.nan,
                            "peak_tracemalloc_mb": np.nan,
                            "X_train_shape": None,
                            "X_train_nnz": None,
                            "X_test_shape": None,
                            "X_test_nnz": None,
                            "extra_json": json.dumps({"error": repr(e)}),
                        }
                    )

    results_df = pd.DataFrame(records)
    results_path = outdir / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    logging.info("Wrote results CSV to %s (%d rows)", results_path, len(results_df))

    if len(results_df):
        summary = (
            results_df.groupby(["algorithm", "max_len"], as_index=False)
            .agg(
                wall_fit_mean=("wall_fit_transform_s", "mean"),
                wall_fit_std=("wall_fit_transform_s", "std"),
                wall_transform_mean=("wall_transform_s", "mean"),
                wall_transform_std=("wall_transform_s", "std"),
                wall_total_mean=("wall_total_s", "mean"),
                wall_total_std=("wall_total_s", "std"),
            )
            .sort_values(["algorithm", "max_len"])
        )
        summary_path = outdir / "benchmark_summary.csv"
        summary.to_csv(summary_path, index=False)
        logging.info("Wrote summary CSV to %s", summary_path)

    logging.info("Benchmark complete.")
    return results_path


def _parse_args(argv = None):
    p = argparse.ArgumentParser(description="Benchmark BFS vs DFS graphlet featurizers on a chemistry dataset.")
    p.add_argument("--data", dest="data_csv", required=True, type=Path, help="Path to CSV with a 'smiles' column.")
    p.add_argument("--outdir", type=Path, default=Path("bench_results"), help="Output directory for logs/CSVs.")
    p.add_argument("--max-len", dest="max_lens", type=int, nargs="+", required=True, help="One or more max_len values.")
    p.add_argument("--repeats", type=int, default=3, help="Number of repeats per (algorithm, max_len).")
    p.add_argument("--train-size", type=float, default=0.8, help="Train fraction for fit_transform (rest is transform). Use 1 to disable split.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for sampling and split.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Passed to GraphletTransformer (1=single-core, -1=all cores).")
    p.add_argument("--chunk-size", type=str, default="even", help="Passed to GraphletTransformer (e.g., 'even').")
    p.add_argument("--verbose", type=int, default=0, help="Passed to GraphletTransformer.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (INFO/DEBUG).")

    # Graph conversion/caching
    p.add_argument("--graph-cache", type=Path, default=None, help="Path to pickle cache for graphs (optional).")
    p.add_argument("--force-recompute-graphs", action="store_true", help="Ignore graph cache and recompute graphs.")
    p.add_argument("--limit", type=int, default=None, help="Use only the first N rows (debug/quick runs).")
    p.add_argument("--sample", type=int, default=None, help="Randomly sample N rows (debug/quick runs).")
    p.add_argument("--no-hs", dest="add_hs", action="store_false", help="Do NOT add hydrogens during SMILES->graph.")
    p.add_argument("--output-2d-pos", action="store_true", help="Ask smiles_to_nx for 2D positions (usually not needed for featurization).")

    # Measurement
    p.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable warm-up run per config.")
    p.add_argument("--track-mem", action="store_true", help="Track peak Python memory via tracemalloc (slower).")

    p.set_defaults(add_hs=True, warmup=True)

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    run_benchmark(
        data_csv=args.data_csv,
        outdir=args.outdir,
        max_lens=args.max_lens,
        repeats=args.repeats,
        train_size=args.train_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        log_level=args.log_level,
        graph_cache=args.graph_cache,
        force_recompute_graphs=args.force_recompute_graphs,
        limit=args.limit,
        sample=args.sample,
        add_hs=args.add_hs,
        output_2d_pos=args.output_2d_pos,
        warmup=args.warmup,
        track_tracemalloc=args.track_mem,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


