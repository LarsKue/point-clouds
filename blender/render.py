
try:
    import bpy
    import bmesh
except ModuleNotFoundError:
    raise RuntimeError(f"Must run from Blender.")

import pathlib
import subprocess as sub
import sys
import os


try:
    # FOR SOME REASON this loads all the modules
    help("modules")
    import torch
    import numpy as np
except ModuleNotFoundError:
    import ensurepip

    # install pip
    ensurepip.bootstrap(upgrade=True)

    target = pathlib.Path(sys.executable).parents[1] / "lib" / "python3.10" / "site-packages"

    # install requirements
    args = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]# , f"--target={target}"]
    sub.run(args)

    # reimport requirements
    help("modules")
    import torch
    import numpy as np

sys.path.append(".")
torch.autograd.set_grad_enabled(False)

from data import MeshDataset

# name clash with blender utils so use some hacky import stuff
from importlib.machinery import SourceFileLoader
pc_path = pathlib.Path(__file__).parents[1]
utils = SourceFileLoader("utils", str(pc_path / "utils.py")).load_module()


def load_material(path):
    bpy.ops.wm.append(
        filepath=str(path / "Material" / path.stem),
        directory=str(path / "Material"),
        filename=path.stem,
        active_collection=False,
        set_fake=True,
    )
    material = bpy.data.materials.get(path.stem)

    return material


def make_node_group():
    """
    Create a new empty node group that can be used in a GeometryNodes modifier
    """
    bpy.ops.mesh.primitive_ico_sphere_add(
        radius=0.05,
    )
    icosphere = bpy.data.objects.get("Icosphere")
    bpy.ops.object.shade_smooth()
    icosphere.hide_set(True)
    icosphere.hide_render = True

    node_group = bpy.data.node_groups.new("GeometryNodes", "GeometryNodeTree")

    in_node = node_group.nodes.new("NodeGroupInput")
    in_node.location = (0, 0)

    out_node = node_group.nodes.new("NodeGroupOutput")
    out_node.location = (600, 0)

    points_node = node_group.nodes.new("GeometryNodeInstanceOnPoints")
    points_node.location = (200, 0)

    object_info = node_group.nodes.new("GeometryNodeObjectInfo")
    object_info.location = (0, -200)
    object_info.inputs[0].default_value = icosphere

    realize = node_group.nodes.new("GeometryNodeRealizeInstances")
    realize.location = (400, 0)

    in_node.outputs.new("NodeSocketGeometry", "Geometry")
    node_group.links.new(in_node.outputs["Geometry"], points_node.inputs["Points"])

    node_group.links.new(object_info.outputs["Geometry"], points_node.inputs["Instance"])
    node_group.links.new(object_info.outputs["Scale"], points_node.inputs["Scale"])

    node_group.links.new(points_node.outputs["Instances"], realize.inputs["Geometry"])

    out_node.inputs.new("NodeSocketGeometry", "Geometry")
    node_group.links.new(realize.outputs["Geometry"], out_node.inputs["Geometry"])

    return node_group


def render(path, resolution=(800, 600), device="gpu", samples=16):
    """
    Render the current scene to an image
    Parameters
    ----------
    path: pathlib.Path to the file
    resolution: tuple of image (width, height)
    device: gpu or cpu
    samples: number of render samples to use. higher values improve quality, at the cost of computation time
    """
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = device.upper()
    bpy.context.scene.cycles.samples = samples
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
    bpy.context.scene.render.film_transparent = True

    # turn on for improved performance when rendering multiple similar images
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.render.filepath = str(path)
    bpy.context.scene.render.image_settings.file_format = path.suffix[1:].upper()
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.ops.render.render(write_still=True)


def main():
    ds = MeshDataset("data/processed", shapes=["chair"], split="train", samples=8192)
    # (32, 2048, 3)
    points = torch.stack([ds[i] for i in range(16)])
    points = utils.normalize(points, dim=1)

    # perm = torch.randperm(points.shape[1])[0:128]
    # points = points[:, perm]

    # remove the default cube
    default_cube = bpy.data.objects.get("Cube")
    bpy.data.objects.remove(default_cube, do_unlink=True)

    # add floor
    bpy.ops.mesh.primitive_plane_add(
        size=100,
        location=(0, 0, -3),
    )
    floor = bpy.data.objects.get("Plane")
    floor.is_shadow_catcher = True

    # move the camera a little further out
    camera = bpy.data.objects.get("Camera")
    camera.location = (13.518, -12.731, 9.1674)

    light = bpy.data.objects.get("Light")
    light.data.type = "SUN"
    light.data.angle = np.deg2rad(30)
    light.data.energy = 2.0
    light.rotation_euler = tuple(np.deg2rad((34.9, -14.1, 76)))

    rainbow = load_material(pc_path / "materials" / "rainbow.blend")

    node_group = make_node_group()

    for i, shape in enumerate(points):
        mesh = bpy.data.meshes.new(f"mesh {i}")
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        bm = bmesh.new()

        for j, point in enumerate(shape):
            print(f"\rShape {i}: Object {j:04d}", end="")
            # obj = bpy.data.objects.new(f"empty_{i}_{j}", None)
            # bpy.context.scene.collection.objects.link(obj)
            bm.verts.new(point.tolist())

        bm.to_mesh(mesh)
        bm.free()

        modifier = obj.modifiers.new("GeometryNodes", "NODES")
        modifier.node_group = node_group

        # select the object
        bpy.context.view_layer.objects.active = obj

        # apply the modifier
        bpy.ops.object.modifier_apply(modifier="GeometryNodes")

        # add the material
        obj.data.materials.append(rainbow)

        path = pathlib.Path(os.getcwd()) / "plots" / "renders" / f"shape_{i}.png"
        render(path, resolution=(1920, 1080), samples=16)

        obj.hide_set(True)
        obj.hide_render = True

        print()


if __name__ == "__main__":
    main()
