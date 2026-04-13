#!/usr/bin/env python3
"""
plotting_example.py

[***] How does this plotting example work? [***]
1) creates a NetworkX graph (Peterson)
2) uses buhito's graphlet featurizer to enumerate graphlets up to size 5
3) uses buhito.plotting.draw_graphlet to render each graphlet to a PNG
4) combines all individual graphlet PNGs into a single tiled image

[***] How to run? [***]
Run: python plotting_example.py
"""

import math
from pathlib import Path
import shutil

import networkx as nx
from PIL import Image

from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer 
from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer
import buhito.plotting as bhp


def main(out_dir="graphlets_out", max_size=5, per_row=6, remove_folder_after=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = nx.petersen_graph()

    labels = {n: f"C" for n in G.nodes}
    nx.set_node_attributes(G, labels, "atom_key")
    labels_e = {n: f"1" for n in G.edges}
    nx.set_edge_attributes(G, labels_e, "bond_key")

    bfs = BFSGraphletFeaturizer(return_nodewise=False, max_len=max_size)
    dfs = DFSGraphletFeaturizer(max_len=max_size)

    a1, b1 = dfs(G)
    a2, b2 = bfs(G)

    assert a1==a2, "BFS and DFS featurizers should produce the same bit_ids for the same graph"

    bits = list(list(a1.keys()))
    png_paths = []

    for i, bit in enumerate(bits):
        try:
            node_inds = b1[bit][0]
            P = bhp.draw_graphlet(gr=G, node_ixs=node_inds, node_label_attr="atom_key")
        except Exception:
            continue

        png_path = out_dir / f"graphlet_{i:03d}.png"
        P.write_png(str(png_path), prog="dot")
        png_paths.append(png_path)

    if not png_paths:
        print("No graphlets were drawn.")
        return

    images = [Image.open(p) for p in png_paths]
    widths, heights = zip(*(im.size for im in images))

    n = len(images)
    cols = min(per_row, n)
    rows = math.ceil(n / cols)

    max_w = max(widths)
    max_h = max(heights)

    grid_w = cols * max_w
    grid_h = rows * max_h

    combined = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * max_w
        y = r * max_h
        offset_x = x + (max_w - im.size[0]) // 2
        offset_y = y + (max_h - im.size[1]) // 2
        combined.paste(im, (offset_x, offset_y), im.convert("RGBA"))

    print(f"Saved {len(png_paths)} graphlet PNGs to {out_dir}")

    combined_path = out_dir / "combined_graphlets_peterson_graph.png" if not remove_folder_after else Path('') / "combined_graphlets_peterson_graph.png"
    combined.save(combined_path)
    print(f"Combined image saved to {combined_path}")
    if remove_folder_after:
        shutil.rmtree(out_dir)
        print(f"Removed folder {out_dir} and all individual graphlet PNGs.")

if __name__ == "__main__":
    main(per_row=3, remove_folder_after=True)
