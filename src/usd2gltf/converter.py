from pxr import Usd, UsdGeom

import zipfile
import tempfile
import os

from gltflib import *

import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

from usd2gltf import common
from pathlib import Path

from pprint import pprint

from usd2gltf.converters import usd_mesh, usd_xform

class Converter:
    used_images = []

    # Settings
    interpolation = "LINEAR"

    convert_meshes = True
    convert_normals = True
    convert_texcoords = True
    convert_colors = True
    normalize_normals = True
    use_mesh_indices = True

    convert_xform_animation = True
    convert_skinned_animation = True

    convert_materials = True
    convert_instancers = True
    convert_hierarchy = True

    flatten_xform_animation = False

    normalize_weights = True

    maindata_bytearray = bytearray()
    # TODO: Remove these
    currentByteOffset = 0
    totalByteLen = 0

    animated_xforms_bytearray = bytearray()
    ibt_bytearray = bytearray()

    animated_xforms_bufferview_id = -1
    animated_xforms_bufferview = None

    # Components
    animations = {}
    textures = []
    images = []
    materials = []
    skins = []
    cameras = []
    bufferViews = []
    lights = []
    texts = []
    samplers = []

    # Root nodes
    nodes = []

    nodeMap = {}        # Primpath -> gltf node
    heirachyMap = {}    # Primpath -> node id

    textureMap = {}     # filepath -> texture id
    materialMap = {}    # Primpath -> material_id

    joint_map = {}      # [skelpath] joint -> (node id, node)
    skin_map = {}       # skinpath -> (skin_id, gltfSkin)
    mesh_map = {}       # primpath -> (mesh_id, gltfMesh)
    skeleton_map = {}   # [skeleton][scene_id] -> skeleton joint id
    text_map = {}       # primpath -> (text_id, gltfText)

    # Need to track which TEXCOORD_X to use with 2 uvsets
    # Probably a better way to handle this. Tricky because
    # materials handled before uv primvars. It should probably
    # be the reverse
    material_uv_indices = {}  # Primpath -> uv_name -> uv_index

    is_glb = True

    resources = []

    temp_dir = None

    # USD
    stage = None
    FPS = 24.0

    # glTF
    gltfDoc = None

    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        logger.debug("Created converter")

    def _traverse(self, prim):
        children = prim.nameChildren
        for i, child in enumerate(children):
            self._traverse(child)

    def _gatherChildren(self, prim):
        children_array = []
        for c in prim.GetChildren():
            # if c.IsA(UsdGeom.Xformable) or c.GetTypeName() == "Preliminary_Text":
            if c.IsA(UsdGeom.Xformable):
                if c.GetPrimPath() in self.heirachyMap:
                    child_idx = self.heirachyMap[c.GetPrimPath()]

                    children_array.append(child_idx)
            else:
                children_array += self._gatherChildren(c)
        return children_array

    def _findChildMesh(self, prim):
        for proto_child in prim.GetAllChildren():
            if proto_child.IsA(UsdGeom.Mesh):
                return UsdGeom.Mesh(proto_child)
            elif len(proto_child.GetChildren()) > 0:
                m = self._findChildMesh(proto_child)
                if m:
                    return m

        return None

    def load_usd(self, inputUSD):
        # l = self.localize_zip(inputUSD)
        return Usd.Stage.Open(inputUSD)

    def add_extension(self, extension):
        if not self.gltfDoc.extensionsUsed:
            self.gltfDoc.extensionsUsed = []

        self.gltfDoc.extensionsUsed.append(extension)

    def process(self, stage, outputGLTF):

        logger.debug("== USD -> glTF Settings ==")
        logger.debug(" - Interpolation: " + self.interpolation)

        logger.debug("== Stage Contents ==")

        self.is_glb = outputGLTF.endswith('.glb')

        self.stage = stage

        self.FPS = self.stage.GetFramesPerSecond()

        self.gltfDoc = GLTFModel(
            asset=Asset(version='2.0'),
            scenes=[],
            nodes=[],
            meshes=[],
            buffers=[],
            bufferViews=[],
            accessors=[],
            extensions={},
        )

        # Primitive Ingestion
        logger.debug("Primitives: ")

        def traversePrims(parent, parent_vis=True):
            for prim in parent.GetAllChildren():

                pname = prim.GetName()
                ppath = prim.GetPrimPath()
                logger.debug("prim: {}".format(ppath))

                gltfNode = None
                node_id = -1

                is_visible = parent_vis
                is_pi_prototype = False

                # Handle transforms
                if prim.IsA(UsdGeom.Xformable) and is_visible:
                    node_id, gltfNode = usd_xform.convert(self,
                                                          UsdGeom.Xformable(prim))
                    gltfNode.extensions = {}

                    self.heirachyMap[ppath] = node_id
                    self.nodeMap[ppath] = gltfNode

                # Handle meshes
                if prim.IsA(UsdGeom.Mesh):
                    mesh_id, gltfMesh = usd_mesh.convert(
                        self, UsdGeom.Mesh(prim))

                    if not gltfMesh:
                        continue

                    if gltfNode:
                        gltfNode.mesh = mesh_id

                    self.mesh_map[ppath] = (mesh_id, gltfMesh)

                # Handle children
                if len(prim.GetAllChildren()) > 0:
                    traversePrims(prim, parent_vis=is_visible)

        traversePrims(self.stage.GetPseudoRoot())

        if self.convert_hierarchy:
            # Assign heirachy

            for prim in self.stage.Traverse():
                ppath = prim.GetPrimPath()

                if prim.IsA(UsdGeom.Xformable):
                    children_array = self._gatherChildren(prim)

                    if len(children_array) > 0:
                        if ppath in self.nodeMap:
                            if self.nodeMap[ppath].children:
                                self.nodeMap[ppath].children += children_array
                            else:
                                self.nodeMap[ppath].children = children_array

                    # Store root nodes
                    if prim.GetParent().IsPseudoRoot() and ppath in self.heirachyMap:
                        self.nodes += [self.heirachyMap[ppath]]

        # Add necessary Components

        if len(self.samplers) > 0:
            self.gltfDoc.samplers = self.samplers

        # Buffer data

        self.gltfDoc.buffers.append(Buffer(byteLength=len(
            self.maindata_bytearray), uri='geometry.bin'))
        self.resources.append(FileResource(
            'geometry.bin', data=self.maindata_bytearray))

        if len(self.animations) > 0:
            self.gltfDoc.buffers.append(Buffer(byteLength=len(
                self.animated_xforms_bytearray), uri='animation.bin'))
            self.resources.append(FileResource(
                'animation.bin', data=self.animated_xforms_bytearray))

        self.gltfDoc.buffers.append(Buffer(byteLength=len(
            self.ibt_bytearray), uri='bindTransforms.bin'))
        self.resources.append(FileResource(
            'bindTransforms.bin', data=self.ibt_bytearray))

        # Add scene

        self.gltfDoc.scenes.append(Scene(nodes=self.nodes))

        # Export GLB
        gltfOutput = GLTF(model=self.gltfDoc, resources=self.resources)

        # Export doesnt seem to pack more than 2 .bin resources
        if self.is_glb:
            for res in self.resources:
                logger.debug("Embeded res: " + str(res))
                gltfOutput.embed_resource(res)

        gltfOutput.export(outputGLTF)

        # Delete temp dirs
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            pass


