"""Relief shading for the SYSU219 rough-terrain replay."""

import numpy as np
import trimesh
from pxr import Sdf, UsdGeom, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter


class RoughReplayTerrainImporter(TerrainImporter):
    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        super().import_mesh(name, mesh)
        path = f"{self.cfg.prim_path}/{name}"
        stage = sim_utils.get_current_stage()
        ground = UsdGeom.Mesh.Get(stage, f"{path}/mesh")

        # Use each corner's true elevation. Face-varying color keeps stair risers
        # distinct from treads even where both faces share the same point.
        heights = mesh.vertices[mesh.faces, 2]
        levels = [-0.40, -0.15, 0.0, 0.18, 0.40]
        palette = np.array([
            [0.21, 0.38, 0.45],
            [0.30, 0.48, 0.51],
            [0.42, 0.60, 0.56],
            [0.66, 0.71, 0.54],
            [0.88, 0.83, 0.64],
        ])
        colors = np.stack([np.interp(heights, levels, channel) for channel in palette.T], axis=-1)
        normals = mesh.face_normals
        # Smooth the lighting across the small height-field triangles. Keep the
        # box treads flat and their edges sharp.
        slopes = 0.8 * normals[:, 0] - 0.6 * normals[:, 1]
        smooth_slopes = 0.8 * mesh.vertex_normals[:, 0] - 0.6 * mesh.vertex_normals[:, 1]
        grid_spacing = self.cfg.terrain_generator.horizontal_scale
        height_field = (mesh.area_faces < grid_spacing ** 2) & (normals[:, 2] > 0.5)
        shading = np.where(height_field[:, None], smooth_slopes[mesh.faces], slopes[:, None])
        colors *= np.clip(1.0 + 2.4 * shading, 0.55, 1.45)[:, :, None]
        colors[normals[:, 2] < 0.5] = (0.045, 0.085, 0.105)
        display_color = ground.CreateDisplayColorPrimvar(UsdGeom.Tokens.faceVarying)
        display_color.SetInterpolation(UsdGeom.Tokens.faceVarying)
        display_color.Set(Vt.Vec3fArray.FromNumpy(colors.reshape(-1, 3).astype(np.float32)))

        material = UsdShade.Material.Get(stage, f"{path}/visualMaterial")
        shader, _, _ = material.ComputeSurfaceSource()
        reader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("TerrainColor"))
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        color_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(color_output)
