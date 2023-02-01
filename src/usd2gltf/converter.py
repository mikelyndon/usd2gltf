from pxr import Usd, UsdGeom, UsdShade, UsdLux, UsdSkel
import zipfile
import tempfile
import os
from gltflib import (
    GLTF,
    GLTFModel,
    Asset,
    Buffer,
    FileResource,
    Scene,
    BufferView,
    Animation,
)
import logging
from pathlib import Path
from usd2gltf.converters import (
        usd_mesh,
        usd_xform,
        usd_material,
        usd_camera,
        usd_lux,
        usd_skel
        )

logger = logging.getLogger(__name__)


class Converter:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.used_images = []

        # Settings
        self.interpolation = "LINEAR"

        self.convert_meshes = True
        self.convert_normals = True
        self.convert_texcoords = True
        self.convert_colors = True
        self.normalize_normals = True
        self.use_mesh_indices = True

        self.convert_xform_animation = True
        self.convert_skinned_animation = True

        self.convert_materials = True
        self.convert_instancers = True
        self.convert_hierarchy = True

        self.flatten_xform_animation = False

        self.normalize_weights = True

        self.maindata_bytearray = bytearray()
        # TODO: Remove these
        self.currentByteOffset = 0
        self.totalByteLen = 0

        self.animated_xforms_bytearray = bytearray()
        self.ibt_bytearray = bytearray()

        self.animated_xforms_bufferview_id = -1
        self.animated_xforms_bufferview = None

        # Components
        self.animations = {}
        self.textures = []
        self.images = []
        self.materials = []
        self.skins = []
        self.cameras = []
        self.bufferViews = []
        self.lights = []
        self.texts = []
        self.samplers = []

        # Root nodes
        self.nodes = []

        self.nodeMap = {}  # Primpath -> gltf node
        self.heirachyMap = {}  # Primpath -> node id

        self.textureMap = {}  # filepath -> texture id
        self.materialMap = {}  # Primpath -> material_id

        self.joint_map = {}  # [skelpath] joint -> (node id, node)
        self.skin_map = {}  # skinpath -> (skin_id, gltfSkin)
        self.mesh_map = {}  # primpath -> (mesh_id, gltfMesh)
        self.skeleton_map = {}  # [skeleton][scene_id] -> skeleton joint id
        self.text_map = {}  # primpath -> (text_id, gltfText)

        # Need to track which TEXCOORD_X to use with 2 uvsets
        # Probably a better way to handle this. Tricky because
        # materials handled before uv primvars. It should probably
        # be the reverse
        self.material_uv_indices = {}  # Primpath -> uv_name -> uv_index

        self.is_glb = True
        self.dirname = ""
        self.basename_without_ext = ""

        self.resources = []

        # USD
        self.stage = None
        self.FPS = 24.0

        # glTF
        self.gltfDoc = None

        logger.debug("Created converter")

    def _traverse(self, prim):
        children = prim.nameChildren
        for i, child in enumerate(children):
            self._traverse(child)

    def _gatherChildren(self, prim):
        children_array = []
        for c in prim.GetChildren():
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

    def localize_zip(self, input_path):
        # Unzip USDZ
        if ".usdz" in input_path:
            with zipfile.ZipFile(input_path, "r") as zip_ref:
                zip_name = os.path.basename(input_path)
                zip_ext = os.path.splitext(zip_name)[0]

                temp_name = os.path.join(self.temp_dir.name, zip_ext)

                os.mkdir(temp_name)

                zip_ref.extractall(temp_name)

                for path in Path(temp_name).rglob("*.usd*"):
                    path = str(path)
                    logger.debug("Unzipped source file: " + path)
                    return path
        else:
            return input_path

    def load_usd(self, inputUSD):
        path = self.localize_zip(inputUSD)
        return Usd.Stage.Open(path)

    def animated_buffer_view(self):
        if (
            self.animated_xforms_bufferview_id == -1
            and self.animated_xforms_bufferview is None
        ):
            # Setup buffers
            self.animated_xforms_bufferview_id = len(self.gltfDoc.bufferViews)

            self.animated_xforms_bufferview = BufferView(
                buffer=1, byteOffset=0, byteLength=0
            )

            # Always add, subsequent bufferViews need the proper id
            # Remove later if empty
            self.gltfDoc.bufferViews.append(self.animated_xforms_bufferview)

        return (self.animated_xforms_bufferview_id, self.animated_xforms_bufferview)

    def get_named_animation(self, animation_name):
        if self.flatten_xform_animation:
            animation_name = "main_animation"

        # Create animation if doesnt exist
        if animation_name not in self.animations:
            _animation = Animation(
                name=animation_name,
                channels=[],
                samplers=[],
            )
            self.animations[animation_name] = _animation
            logger.debug("Created anim: " + animation_name)

        # Return animation object
        return self.animations[animation_name]

    def add_extension(self, extension):
        if not self.gltfDoc.extensionsUsed:
            self.gltfDoc.extensionsUsed = []

        self.gltfDoc.extensionsUsed.append(extension)

    def process(self, stage, outputGLTF):

        logger.debug("== USD -> glTF Settings ==")
        logger.debug(" - Interpolation: " + self.interpolation)

        logger.debug("== Stage Contents ==")

        self.is_glb = outputGLTF.endswith(".glb")

        self.stage = stage

        # TODO: Is this being used?
        # Unzip all sublayers
        for i, l in enumerate(self.stage.GetRootLayer().subLayerPaths):
            if "usdz" in l:
                new_path = self.localize_zip(str(l))

        self.FPS = self.stage.GetFramesPerSecond()

        self.gltfDoc = GLTFModel(
            asset=Asset(version="2.0"),
            scenes=[],
            nodes=[],
            meshes=[],
            buffers=[],
            bufferViews=[],
            accessors=[],
            # extensions={},
        )

        self.dirname = os.path.dirname(outputGLTF)
        self.basename_without_ext = os.path.splitext(os.path.basename(outputGLTF))[0]

        # Material Ingestion
        logger.debug("Materials: ")
        # Must be done first to populate materialMap for mesh assignment
        for prim in self.stage.Traverse():
            ppath = prim.GetPrimPath()

            if self.convert_materials:
                # Handle materials
                if prim.IsA(UsdShade.Material):
                    material_id, gltfMaterial = usd_material.convert(
                        self, UsdShade.Material(prim)
                    )

                    self.materialMap[ppath] = material_id

            # Gather for later
            if prim.IsA(UsdGeom.PointInstancer):
                pi = UsdGeom.PointInstancer(prim)
                point_instancers.append(pi)

                point_instancer_prototypes += pi.GetPrototypesRel().GetTargets()

        # Primitive Ingestion
        logger.debug("Primitives: ")

        def traversePrims(parent, parent_vis=True):
            for prim in parent.GetAllChildren():

                ppath = prim.GetPrimPath()
                logger.debug("prim: {}".format(ppath))

                gltfNode = None
                node_id = -1

                is_visible = parent_vis

                if prim.IsA(UsdLux.DomeLight) and is_visible:
                    logger.warning(
                        "Dome lights are not supported. {} will not be exported.".format(
                            ppath
                        )
                    )
                    continue

                # Handle transforms
                if prim.IsA(UsdGeom.Xformable) and is_visible:
                    node_id, gltfNode = usd_xform.convert(self, UsdGeom.Xformable(prim))
                    # gltfNode.extensions = {}

                    self.heirachyMap[ppath] = node_id
                    self.nodeMap[ppath] = gltfNode

                # Handle lights
                _lightBase = None
                try:
                    _lightBase = UsdLux.BoundableLightBase
                except Exception:
                    _lightBase = UsdLux.Light

                if _lightBase:
                    if (
                        prim.IsA(_lightBase)
                        or prim.IsA(UsdLux.DistantLight)
                        and is_visible
                    ):
                        gltfLight = usd_lux.convert(self, _lightBase(prim))

                        light_id = len(self.lights)
                        self.lights.append(gltfLight)
                        gltfNode.extensions = {}
                        gltfNode.extensions["KHR_lights_punctual"] = {"light": light_id}

                # Handle cameras
                if prim.IsA(UsdGeom.Camera) and is_visible:
                    camera_id, gltfCamera = usd_camera.convert(
                        self, UsdGeom.Camera(prim)
                    )

                    gltfNode.camera = camera_id
                    self.cameras.append(gltfCamera)

                # Handle meshes
                if prim.IsA(UsdGeom.Mesh):
                    mesh_id, gltfMesh = usd_mesh.convert(self, UsdGeom.Mesh(prim))

                    if not gltfMesh:
                        continue

                    if gltfNode:
                        gltfNode.mesh = mesh_id

                    self.mesh_map[ppath] = (mesh_id, gltfMesh)

                # Handle children
                if len(prim.GetAllChildren()) > 0:
                    traversePrims(prim, parent_vis=is_visible)

        traversePrims(self.stage.GetPseudoRoot())

        # Handle skinned animation
        if self.convert_skinned_animation:
            for prim in self.stage.TraverseAll():
                ppath = prim.GetPrimPath()

                if prim.IsA(UsdSkel.Root):
                    skelApi = UsdSkel.BindingAPI(prim)
                    if skelApi.GetSkeletonRel():
                        targets = skelApi.GetSkeletonRel().GetTargets()
                        if len(targets) > 0:
                            skel_rel = skelApi.GetSkeletonRel().GetTargets()[0]
                            # For some reason in Houdini the skel and animation rels are null

                if prim.IsA(UsdSkel.Skeleton):
                    skel_prim = UsdSkel.Skeleton(prim)
                    skel_id, gltfSkelRoot, joint_node_ids = usd_skel.add_skeleton_rig(
                        self, skel_prim
                    )

                    skin_id, gltfSkin = usd_skel.add_skin(
                        self, skel_prim, skel_id, joint_node_ids
                    )

                    # TODO: Could be mapped better
                    # Map the scene object ids to skeleton local index
                    self.skeleton_map[ppath] = {}

                    for i, j in enumerate(joint_node_ids):
                        self.skeleton_map[ppath][j] = i

                    self.skins.append(gltfSkin)
                    self.skin_map[ppath] = (skin_id, gltfSkin)

                    self.nodeMap[ppath] = gltfSkelRoot
                    self.heirachyMap[ppath] = skel_id

                    skelApi = UsdSkel.BindingAPI(prim)
                    if skelApi.GetAnimationSourceRel():
                        targets = skelApi.GetAnimationSourceRel().GetTargets()
                        if len(targets) > 0:
                            animation_rel = skelApi.GetAnimationSource()
                            usd_skel._addSkeletonAnimation(
                                self, skel_prim, UsdSkel.Animation(animation_rel)
                            )

            idx = 0
            for prim in self.stage.TraverseAll():
                ppath = prim.GetPrimPath()

                if prim.IsA(UsdGeom.Mesh):
                    mesh_id, gltfMesh = self.mesh_map[ppath]

                    skelBinding = UsdSkel.BindingAPI(prim)

                    # TODO: Make nicer
                    if not skelBinding.GetJointWeightsAttr().IsValid():
                        continue

                    # Weights

                    usd_skel.add_weights(self, gltfMesh, UsdGeom.Mesh(prim))

                    skelApi = UsdSkel.BindingAPI(prim)
                    if skelApi.GetSkeletonRel():
                        targets = skelApi.GetSkeletonRel().GetTargets()
                        if len(targets) > 0:
                            skel_rel = skelApi.GetSkeletonRel().GetTargets()[0]

                            if skel_rel not in self.skin_map:
                                continue

                            skin_id, gltfSkin = self.skin_map[skel_rel]

                            mnode = self.nodeMap[ppath]
                            mnode.skin = skin_id

                idx += 1

        if self.convert_hierarchy:
            # Assign hierachy

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

        if len(self.animations) > 0:
            self.gltfDoc.animations = list(self.animations.values())

        if len(self.images) > 0:
            self.gltfDoc.images = self.images

        if len(self.textures) > 0:
            self.gltfDoc.textures = self.textures

        if len(self.materials) > 0:
            self.gltfDoc.materials = self.materials

        if len(self.skins) > 0:
            self.gltfDoc.skins = self.skins

        if len(self.cameras) > 0:
            self.gltfDoc.cameras = self.cameras

        if len(self.lights) > 0:
            if self.gltfDoc.extensions is None:
                self.gltfDoc.extensions = {}
            self.gltfDoc.extensions["KHR_lights_punctual"] = {"lights": self.lights}
            self.add_extension("KHR_lights_punctual")

        if len(self.samplers) > 0:
            self.gltfDoc.samplers = self.samplers

        # Buffer data
        self.gltfDoc.buffers.append(
            Buffer(
                byteLength=len(self.maindata_bytearray),
                uri="{}_geometry.bin".format(self.basename_without_ext),
            )
        )
        self.resources.append(
            FileResource(
                "{}_geometry.bin".format(self.basename_without_ext),
                data=self.maindata_bytearray,
            )
        )

        if len(self.animations) > 0:
            self.gltfDoc.buffers.append(
                Buffer(
                    byteLength=len(self.animated_xforms_bytearray),
                    uri="{}_animation.bin".format(self.basename_without_ext),
                )
            )
            self.resources.append(
                FileResource(
                    "{}_animation.bin".format(self.basename_without_ext),
                    data=self.animated_xforms_bytearray,
                )
            )
        if len(self.skins) > 0:
            self.gltfDoc.buffers.append(
                Buffer(
                    byteLength=len(self.ibt_bytearray),
                    uri="{}_bindTransforms.bin".format(self.basename_without_ext),
                )
            )
            self.resources.append(
                FileResource(
                    "{}_bindTransforms.bin".format(self.basename_without_ext),
                    data=self.ibt_bytearray,
                )
            )

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
            logger.error(e)
