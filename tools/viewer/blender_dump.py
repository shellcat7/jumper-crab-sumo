"""A replay's geometry for Blender (10/19): every frame render_offscreen.py draws, the
pose of every geom that moves, so tools/blender/sumo_scene.py can rebuild the bout in a
detailed hall and render it there (Cycles / EEVEE on the PC's GPU).

Set JUMPER_BLENDER_DUMP=out.npz with render_offscreen.py. Written at exit:

    dyn_*     the geoms of the robots (robot/, opponent/, gyoji/ -- the gunbai too):
              type, size, rgba, mesh, name; per frame their world position (F, G, 3)
              and rotation matrix (F, G, 9)
    static_*  the other geoms but the crowd (the dohyo, the roof, the floor ...) once,
              in the same form: Blender may use them or put its own hall in their place
    crowd_*   the spectators and judges as one triangle soup at the first frame, with a
              colour per vertex (tens of thousands of geoms: one object, not one each)
    crowdg_*  (10/20) the same crowd geom by geom -- type, size, rgba, mesh, spectator --
              and every CROWD_EVERY-th frame (JUMPER_BLENDER_CROWD_EVERY, default 2) their
              world position (float32) and orientation (quaternion, float16): the audience
              as audience.py animates it (chatting, leaning in, cheering), for Blender to
              instance clean copies of the pieces; the props (caps, towels, flags) too
    mesh_*    the vertices and faces of every mesh a dyn or static geom uses
    cam       per frame the MuJoCo free camera: lookat (3), distance, azimuth, elevation
    t         per frame the video's time; caption: the gyoji's call shown then (as
              render_offscreen.py's caption, "" for none), wins: the score so far;
              names: JUMPER_SUMO_NAMES
"""

from __future__ import annotations

import os

import numpy as np

_ROBOTS = ("robot/", "opponent/", "gyoji/")


def _entity_of(name: str) -> str | None:
    for p in _ROBOTS:
        if name.startswith(p):
            return p[:-1]
    return None


def _mat2quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 9) to unit quaternions (N, 4: w, x, y, z), vectorised."""
    m = m.reshape(-1, 3, 3).astype(np.float64)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((len(m), 4))
    w = np.sqrt(np.maximum(0.0, 1.0 + t)) / 2
    x = np.sqrt(np.maximum(0.0, 1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])) / 2
    y = np.sqrt(np.maximum(0.0, 1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2])) / 2
    z = np.sqrt(np.maximum(0.0, 1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2])) / 2
    q[:, 0] = w
    q[:, 1] = np.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    q[:, 2] = np.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    q[:, 3] = np.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class Dumper:
    def __init__(self, path: str):
        self.path = path
        self.frames_pos: list[np.ndarray] = []
        self.frames_mat: list[np.ndarray] = []
        self.cams: list[list[float]] = []
        self.times: list[float] = []
        self.captions: list[str] = []
        self.wins: list[list[int]] = []
        self.ready = False

    def _first(self, model, data, opt) -> None:
        import mujoco

        visible = np.array([opt.geomgroup[int(g)] if 0 <= g < 6 else 0 for g in model.geom_group], bool)
        body_name = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
        root = model.body_rootid
        dyn, static, crowd = [], [], []
        for g in range(model.ngeom):
            if not visible[g] or model.geom_rgba[g, 3] == 0:
                continue
            b = model.geom_bodyid[g]
            rname = body_name[root[b]]
            if _entity_of(body_name[b]) or _entity_of(rname):
                dyn.append(g)
            elif rname.startswith("decor_crowd") or body_name[b].startswith("decor_crowd"):
                crowd.append(g)
            else:
                static.append(g)
        self.dyn = np.array(dyn, int)
        self.static = np.array(static, int)
        self.crowd = np.array(crowd, int)

        def info(ids):
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{g}" for g in ids]
            rgba = model.geom_rgba[ids].copy()
            # A material's colour where the geom keeps the default grey.
            mat = model.geom_matid[ids]
            for k, m in enumerate(mat):
                if m >= 0 and np.allclose(rgba[k], [0.5, 0.5, 0.5, 1.0]):
                    rgba[k] = model.mat_rgba[m]
            bodies = [body_name[model.geom_bodyid[g]] for g in ids]
            return dict(type=model.geom_type[ids].copy(), size=model.geom_size[ids].copy(), rgba=rgba,
                        mesh=model.geom_dataid[ids].copy(), name=np.array(names), body=np.array(bodies))

        self.dyn_info = info(self.dyn)
        self.static_info = info(self.static)
        self.static_pos = data.geom_xpos[self.static].copy()
        self.static_mat = data.geom_xmat[self.static].copy()
        self.crowd_info = info(self.crowd)
        self.crowd_info["fig"] = model.geom_bodyid[self.crowd].astype(np.int32)   # decor_crowd_<tag>
        self.crowd_every = max(1, int(os.environ.get("JUMPER_BLENDER_CROWD_EVERY", "2")))
        self.crowd_pos: list[np.ndarray] = []
        self.crowd_quat: list[np.ndarray] = []
        meshes = set(int(m) for m in self.dyn_info["mesh"][self.dyn_info["type"] == 7])
        meshes |= set(int(m) for m in self.crowd_info["mesh"][self.crowd_info["type"] == 7])
        meshes |= set(int(m) for m in self.static_info["mesh"][self.static_info["type"] == 7])
        self.meshes = {}
        for m in sorted(meshes):
            va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
            fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
            self.meshes[m] = (model.mesh_vert[va:va + vn].copy(), model.mesh_face[fa:fa + fn].copy())
        # The crowd: mesh geoms in world coordinates, a colour per vertex.
        verts, faces, cols, figs, n = [], [], [], [], 0

        def top(b):  # the body right under the world: one spectator, whatever its links
            while model.body_parentid[b] != 0:
                b = model.body_parentid[b]
            return b
        for g in self.crowd:
            if model.geom_type[g] != 7:
                continue
            m = model.geom_dataid[g]
            va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
            fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
            v = model.mesh_vert[va:va + vn] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            verts.append(v)
            faces.append(model.mesh_face[fa:fa + fn] + n)
            rgba = model.geom_rgba[g]
            if model.geom_matid[g] >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
                rgba = model.mat_rgba[model.geom_matid[g]]
            cols.append(np.repeat(rgba[None, :3], vn, 0))
            figs.append(np.full(vn, model.geom_bodyid[g], np.int32))   # decor_crowd_<tag>: one spectator
            n += vn
        self.crowd_v = np.concatenate(verts).astype(np.float32) if verts else np.zeros((0, 3), np.float32)
        self.crowd_f = np.concatenate(faces).astype(np.int32) if faces else np.zeros((0, 3), np.int32)
        self.crowd_c = np.concatenate(cols).astype(np.float32) if cols else np.zeros((0, 3), np.float32)
        # 10/20: the spectator each vertex belongs to (tools/blender/crowd_figures.py colours them one each)
        self.crowd_b = np.concatenate(figs) if figs else np.zeros(0, np.int32)
        self.ready = True
        print(f"[blender_dump] {len(self.dyn)} moving geoms, {len(self.static)} static, "
              f"{len(self.crowd)} crowd geoms ({len(self.crowd_f)} triangles), {len(self.meshes)} meshes")

    def frame(self, model, data, opt, cam, t: float, caption: str = "", wins=(0, 0)) -> None:
        if not self.ready:
            self._first(model, data, opt)
        if len(self.crowd) and len(self.times) % self.crowd_every == 0:
            self.crowd_pos.append(data.geom_xpos[self.crowd].astype(np.float32))
            self.crowd_quat.append(_mat2quat(data.geom_xmat[self.crowd]).astype(np.float16))
        self.frames_pos.append(data.geom_xpos[self.dyn].astype(np.float32))
        self.frames_mat.append(data.geom_xmat[self.dyn].astype(np.float32))
        self.cams.append([*cam.lookat, cam.distance, cam.azimuth, cam.elevation])
        self.times.append(t)
        self.captions.append(caption)
        self.wins.append(list(wins))

    def write(self) -> None:
        if not self.ready:
            return
        out = dict(
            dyn_pos=np.stack(self.frames_pos), dyn_mat=np.stack(self.frames_mat),
            static_pos=self.static_pos, static_mat=self.static_mat,
            crowd_v=self.crowd_v, crowd_f=self.crowd_f, crowd_c=self.crowd_c, crowd_b=self.crowd_b,
            cam=np.array(self.cams, np.float32), t=np.array(self.times, np.float32),
            names=np.array(os.environ.get("JUMPER_SUMO_NAMES", "")),
            caption=np.array(self.captions), wins=np.array(self.wins, np.int32),
        )
        for k, v in self.dyn_info.items():
            out[f"dyn_{k}"] = v
        for k, v in self.static_info.items():
            out[f"static_{k}"] = v
        if self.crowd_pos:
            for k, v in self.crowd_info.items():
                out[f"crowdg_{k}"] = v
            out["crowdg_pos"] = np.stack(self.crowd_pos)
            out["crowdg_quat"] = np.stack(self.crowd_quat)
            out["crowdg_every"] = np.array(self.crowd_every)
        ids = sorted(self.meshes)
        out["mesh_ids"] = np.array(ids, int)
        for m in ids:
            out[f"mesh_{m}_v"], out[f"mesh_{m}_f"] = self.meshes[m]
        np.savez_compressed(self.path, **out)
        print(f"[blender_dump] {len(self.times)} frames written to {self.path}")
