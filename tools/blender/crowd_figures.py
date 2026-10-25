"""One colour per spectator for the Blender crowd (10/20).

The dump's crowd is one triangle soup of ~42 000 mesh pieces (about a thousand crab
spectators of ~41 parts each: a dark shell, legs and claws in their own colours);
clustered for Blender the parts of a figure fall into each other and the stands read as
a speckle. Here each spectator gets one colour -- its shell's, the other parts a
shade darker or lighter -- so the stands read as a crowd
in its clothes. The spectator of each vertex comes from the dump (crowd_b, the top body
of its geom; tools/viewer/blender_dump.py after 10/20). The crowd is the same in every
dump of the hall, so this runs once and every render uses it:

    python -I crowd_figures.py dump_with_crowd_b.npz crowd_colours.npz

(sumo_scene.py --crowd-colours FILE, or JUMPER_BLENDER_CROWD=FILE).
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

LUM = np.array([0.30, 0.59, 0.11])


def main(src: str, out: str) -> None:
    d = np.load(src)
    v, f, c = d["crowd_v"], d["crowd_f"], d["crowd_c"].astype(np.float64)
    if "crowd_b" not in d.files:
        raise SystemExit("this dump has no crowd_b (made before 10/20): dump the hall again")
    n = len(v)
    _, fig = np.unique(d["crowd_b"], return_inverse=True)
    m = fig.max() + 1
    # the parts (connected pieces) with their colour and size
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]]])
    k, part = connected_components(coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n)),
                                   directed=False)
    cnt = np.bincount(part, minlength=k).astype(np.float64)
    pcol = np.stack([np.bincount(part, weights=c[:, i], minlength=k) for i in range(3)], 1) / cnt[:, None]
    pfig = np.zeros(k, np.int64)
    pfig[part] = fig
    print(f"{n} vertices, {k} parts, {m} spectators")
    # each spectator's colour: its largest part's (the shell: the jacket or kimono)
    order = np.lexsort((-cnt, pfig))
    best = order[np.r_[True, pfig[order][1:] != pfig[order][:-1]]]
    main_col = np.zeros((m, 3))
    main_col[pfig[best]] = pcol[best]
    # the other parts in the same colour, a little darker or lighter by their own brightness:
    # the audience's simplified meshes let inner parts show through the shells in spots
    # (in Blender at full resolution: a camouflage), which in one colour no longer show
    shade = np.clip(0.8 + 0.2 * (pcol @ LUM) / np.maximum(main_col[pfig] @ LUM, 0.05), 0.8, 1.05)
    col = np.clip(main_col[pfig] * shade[:, None], 0, 1)[part].astype(np.float32)
    np.savez_compressed(out, crowd_c=col)
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
