"""Fixed-difficulty Perlin height field and ground queries for RSS 2021 RMA."""

import numpy as np
import trimesh
import warp as wp

from .terrain_appearance import RoughReplayTerrainImporter

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp import convert_to_warp_mesh


def _perlin(permutation, x, y):
    """Two-dimensional slice (z=0) of improved 3-D Perlin gradient noise."""
    ix, iy = np.floor(x).astype(int) % 256, np.floor(y).astype(int) % 256
    fx, fy = x - np.floor(x), y - np.floor(y)
    ux = fx**3 * (fx * (6 * fx - 15) + 10)
    uy = fy**3 * (fy * (6 * fy - 15) + 10)

    def corner(dx, dy):
        h = permutation[permutation[permutation[ix + dx] + iy + dy]] % 16
        xx, yy = fx - dx, fy - dy
        u = np.where(h < 8, xx, yy)
        v = np.where(h < 4, yy, np.where((h == 12) | (h == 14), xx, 0.0))
        return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)

    near = (1 - ux) * corner(0, 0) + ux * corner(1, 0)
    far = (1 - ux) * corner(0, 1) + ux * corner(1, 1)
    return (1 - uy) * near + uy * far


class PaperFractalTerrain:
    """A continuous collision mesh with independent spawn locations for parallel episodes."""

    def __init__(self, cfg, device="cpu"):
        # The CMS authors' add_random_terrain() supplies the patch size, frequency and
        # sampling density omitted from RSS. Its z-scale is replaced by RSS's 0.27.
        nx, ny = cfg.num_rows * 200, cfg.num_cols * 200
        width, length = cfg.num_rows * cfg.size[0], cfg.num_cols * cfg.size[1]
        x, y = np.meshgrid(np.linspace(0, width, nx), np.linspace(0, length, ny), indexing="ij")
        rng = np.random.default_rng(cfg.seed if cfg.seed is not None else np.random.randint(2**31))
        p = np.tile(rng.permutation(256), 2)
        px, py = np.meshgrid(np.arange(nx) * width * 10 / nx, np.arange(ny) * length * 10 / ny, indexing="ij")
        # RaiSim's [0,1] fBm normalization, two octaves, lacunarity 2 and gain 0.25.
        h = 0.27 * (0.5 + 0.5 * (_perlin(p, px, py) + 0.25 * _perlin(p, 2 * px, 2 * py)) / 1.25)
        x -= width / 2
        y -= length / 2
        points = np.column_stack((x.ravel(), y.ravel(), h.ravel()))
        i = np.arange(nx * ny, dtype=np.int32).reshape(nx, ny)[:-1, :-1].ravel()
        faces = np.vstack((np.column_stack((i, i + ny, i + ny + 1)), np.column_stack((i, i + ny + 1, i + 1))))
        self.terrain_mesh = trimesh.Trimesh(points, faces, process=False)
        xs = np.rint((np.arange(cfg.num_rows) + 0.5) * nx / cfg.num_rows).astype(int)
        ys = np.rint((np.arange(cfg.num_cols) + 0.5) * ny / cfg.num_cols).astype(int)
        patch = np.ix_(xs, ys)
        self.terrain_origins = np.stack([field[patch] for field in (x, y, h)], axis=-1)
        self.flat_patches = {}


@wp.kernel(enable_backward=False)
def _cast_vertical_rays(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    start_z: float,
    direction: wp.vec3,
    heights: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    position = positions[i]
    start = wp.vec3(position[0], position[1], start_z)
    hit = wp.mesh_query_ray(mesh, start, direction, 1.0e6)
    if hit.result:
        heights[i] = (start + hit.t * direction)[2]
    else:
        heights[i] = wp.inf


class PaperTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        self.ground_query_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, self.device)
        if self.ground_query_mesh.device.is_cuda:
            # Finish building the BVH before it can be queried on any Torch stream.
            wp.synchronize_stream(self.ground_query_mesh.device.stream)
        self.ray_start_z = float(mesh.bounds[1, 2]) + 1.0

    def heights(self, positions):
        """Query the collision mesh on the same stream as the tensors that consume the heights."""
        mesh = self.ground_query_mesh
        xyz = positions.to(wp.device_to_torch(mesh.device)).reshape(-1, 3).contiguous()
        heights = xyz.new_empty(xyz.shape[0])
        if xyz.shape[0]:
            # Passing the direction at launch preserves the general raycaster's CUDA rounding.
            wp.launch(
                _cast_vertical_rays,
                dim=xyz.shape[0],
                inputs=[mesh.id, wp.from_torch(xyz, dtype=wp.vec3), self.ray_start_z, wp.vec3(0.0, 0.0, -1.0)],
                outputs=[wp.from_torch(heights)],
                device=mesh.device,
                stream=wp.stream_from_torch(xyz.device) if xyz.is_cuda else None,
            )
        return heights.view(positions.shape[:-1]).to(positions.device)


class PaperReplayTerrainImporter(PaperTerrainImporter, RoughReplayTerrainImporter):
    """Color the same collision mesh while retaining the teacher's ray queries."""
