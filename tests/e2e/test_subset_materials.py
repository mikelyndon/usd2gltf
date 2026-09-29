from os import path
from unittest import TestCase

from gltflib import GLTF
from pxr import Usd

import usd2gltf.converter as converter

from ..util import TEMP_DIR, setup_temp_dir

# A quad of two faces: one in a subset bound to "red", then one in a subset
# bound to "blue".
STAGE = """#usda 1.0
def Mesh "quad"
{
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

    def GeomSubset "a" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        uniform token elementType = "face"
        int[] indices = [0]
        rel material:binding = </red>
    }

    def GeomSubset "b" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        uniform token elementType = "face"
        int[] indices = [1]
        rel material:binding = </blue>
    }
}

def Material "red"
{
    token outputs:surface.connect = </red/surface.outputs:surface>

    def Shader "surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        token outputs:surface
    }
}

def Material "blue"
{
    token outputs:surface.connect = </blue/surface.outputs:surface>

    def Shader "surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        token outputs:surface
    }
}
"""


class TestSubsetMaterials(TestCase):
    def test_each_subset_keeps_its_material(self):
        setup_temp_dir()
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(STAGE)
        output = path.join(TEMP_DIR, "subset_materials.gltf")
        converter.Converter().process(stage, output)

        model = GLTF.load(output).model
        bound = [model.materials[p.material].name for p in model.meshes[0].primitives]
        self.assertEqual(bound, ["red", "blue"])
