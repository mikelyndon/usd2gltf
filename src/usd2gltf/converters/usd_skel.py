from pxr import UsdGeom, Sdf, UsdSkel

import struct
import operator
import numpy as np

from gltflib import (
        BufferView,
        BufferTarget,
        Accessor,
        ComponentType,
        AccessorType,
        Node,
        Skin,
        AnimationSampler,
        Channel,
        Target,
        )

from usd2gltf import common
import logging
import math

logger = logging.getLogger(__name__)

# TODO: Optimize
def _normalize_weight(arr):
    f = 1.0 / sum(arr)

    out = []
    for i in arr:
        out.append(i * f)
    return out


def add_weights(converter, gltfMesh, usd_mesh):

    # Some duplication here from mesh, but unavoidable with current architecture

    faces_attr = usd_mesh.GetFaceVertexCountsAttr()
    faces = common._GetStaticValue(faces_attr)

    can_use_mesh_indices = converter.use_mesh_indices

    # Check if any primvar has faceVarying interpolation
    # GLTF Cannot support per face accessors with index based geometry

    if can_use_mesh_indices:
        if converter.convert_normals:
            if "faceVarying" in usd_mesh.GetNormalsInterpolation():
                print("Mesh has faceVarying Normals. Forcing to NOT use indices")
                can_use_mesh_indices = False

    # Needed so we can index point indices from anywhere
    face_offset = []
    amt = 0
    for f in faces:
        face_offset.append(amt)
        amt += f

    subsets = UsdGeom.Subset.GetAllGeomSubsets(usd_mesh)

    remaining_idcs = UsdGeom.Subset.GetUnassignedIndices(subsets, len(faces))
    has_remainder = False
    if len(remaining_idcs) > 0:
        has_remainder = True

    num_subsets = len(subsets)

    if has_remainder:
        num_subsets += 1  # All subsets plus remainder

    subset_idcs = [[]] * num_subsets
    for i, s in enumerate(subsets):
        subset_idcs[i] = s.GetIndicesAttr().Get()

    # TODO: Doesnt work if subset indices are not faces
    if has_remainder:
        subset_idcs[-1] = remaining_idcs

    # Handle each subset

    for i, gltfPrimitive in enumerate(gltfMesh.primitives):
        subset_idc = subset_idcs[i]

        gltfAttributes = gltfPrimitive.attributes

        joints = []

        skelBinding = UsdSkel.BindingAPI(usd_mesh)

        jointWeightsPrimvar = UsdGeom.Primvar(skelBinding.GetJointWeightsAttr())
        jointWeights = jointWeightsPrimvar.Get()
        jointWeightsInterpolation = jointWeightsPrimvar.GetInterpolation()

        num_weights_influences = jointWeightsPrimvar.GetElementSize()

        jointIdcsPrimvar = UsdGeom.Primvar(skelBinding.GetJointIndicesAttr())
        jointIdcs = jointIdcsPrimvar.Get()
        jointIndicesInterpolation = jointIdcsPrimvar.GetInterpolation()

        meshJointsAttr = UsdGeom.Primvar(skelBinding.GetJointsAttr())
        meshJoints = meshJointsAttr.Get()

        if not skelBinding.GetSkeleton():
            # Early out for when primvars/weights exist, but no skeleton is provided
            return

        skeleton_path = skelBinding.GetSkeleton().GetPrim().GetPrimPath()

        orientation = usd_mesh.GetOrientationAttr().Get()
        isLeftHanded = orientation == UsdGeom.Tokens.leftHanded

        faces = usd_mesh.GetFaceVertexCountsAttr().Get()
        idcs = usd_mesh.GetFaceVertexIndicesAttr().Get()

        arr_offset_1 = 0
        arr_offset_2 = 1

        if isLeftHanded:
            arr_offset_1 = 1
            arr_offset_2 = 0

        # - Joints

        num_joints_influences = jointIdcsPrimvar.GetElementSize()
        limit_influence = min(num_joints_influences, 4)

        needs_normalization = False

        if num_joints_influences > 4:
            logger.warning(
                "Mesh limited to max of 4 weights per vertex. Joint influences were {0}".format(
                    num_joints_influences
                )
            )
            if converter.normalize_weights:
                logger.warning(
                    " - Normalize weights: True, normalizing weights to max 4 influences"
                )
            needs_normalization = True

        convertedIdcs = []

        if can_use_mesh_indices:
            for i in range(len(jointIdcs)):
                p1 = [0] * 4
                for k in range(0, limit_influence):
                    _joint_id = i

                    # GLTF Doesnt like indices to 0 weights
                    w = 0.0

                    if jointWeightsInterpolation == UsdGeom.Tokens.vertex:
                        w = jointWeights[(_joint_id * num_weights_influences) + k]
                    elif jointWeightsInterpolation == UsdGeom.Tokens.constant:
                        w = jointWeights[0]
                    if w == 0.0:
                        continue

                    _mesh_joint_index = -1
                    if jointIndicesInterpolation == UsdGeom.Tokens.vertex:
                        _mesh_joint_index = jointIdcs[
                            (_joint_id * num_joints_influences) + k
                        ]
                    elif jointIndicesInterpolation == UsdGeom.Tokens.constant:
                        # TODO: Should check exists
                        _mesh_joint_index = jointIdcs[0]

                    if meshJoints:
                        mesh_joint_path = Sdf.Path(
                            meshJoints[_mesh_joint_index].split("/")[-1]
                        )
                        gltfIdx, gltfBone = converter.joint_map[skeleton_path][
                            mesh_joint_path
                        ]

                        gltfIdx = converter.skeleton_map[skeleton_path][gltfIdx]
                        p1[k] = gltfIdx

                    convertedIdcs.append(p1)
        else:
            for si in subset_idc:
                f = faces[si]
                amt = face_offset[si]

                for j in range(1, f - 1):

                    p1 = [0] * 4
                    for k in range(0, limit_influence):
                        _joint_id = idcs[amt]

                        # GLTF Doesnt like indices to 0 weights
                        w = 0.0

                        if jointWeightsInterpolation == UsdGeom.Tokens.vertex:
                            w = jointWeights[(_joint_id * num_weights_influences) + k]
                        elif jointWeightsInterpolation == UsdGeom.Tokens.constant:
                            w = jointWeights[0]
                        if w == 0.0:
                            continue

                        _mesh_joint_index = -1
                        if jointIndicesInterpolation == UsdGeom.Tokens.vertex:
                            _mesh_joint_index = jointIdcs[
                                (_joint_id * num_joints_influences) + k
                            ]
                        elif jointIndicesInterpolation == UsdGeom.Tokens.constant:
                            # TODO: Should check exists
                            _mesh_joint_index = jointIdcs[0]

                        if meshJoints:
                            mesh_joint_path = Sdf.Path(
                                meshJoints[_mesh_joint_index].split("/")[-1]
                            )
                            gltfIdx, gltfBone = converter.joint_map[skeleton_path][
                                mesh_joint_path
                            ]

                            gltfIdx = converter.skeleton_map[skeleton_path][gltfIdx]
                            p1[k] = gltfIdx

                    convertedIdcs.append(p1)

                    p2 = [0] * 4
                    for k in range(0, limit_influence):
                        _joint_id = idcs[amt + j + arr_offset_1]

                        # GLTF Doesnt like indices to 0 weights
                        w = 0.0

                        if jointWeightsInterpolation == UsdGeom.Tokens.vertex:
                            w = jointWeights[(_joint_id * num_weights_influences) + k]
                        elif jointWeightsInterpolation == UsdGeom.Tokens.constant:
                            w = jointWeights[0]

                        if w == 0.0:
                            continue

                        _mesh_joint_index = -1
                        if jointIndicesInterpolation == UsdGeom.Tokens.vertex:
                            _mesh_joint_index = jointIdcs[
                                (_joint_id * num_joints_influences) + k
                            ]
                        elif jointIndicesInterpolation == UsdGeom.Tokens.constant:
                            # TODO: Should check exists
                            _mesh_joint_index = jointIdcs[0]

                        if meshJoints:
                            mesh_joint_path = Sdf.Path(
                                meshJoints[_mesh_joint_index].split("/")[-1]
                            )
                            gltfIdx, gltfBone = converter.joint_map[skeleton_path][
                                mesh_joint_path
                            ]
                            gltfIdx = converter.skeleton_map[skeleton_path][gltfIdx]
                            p2[k] = gltfIdx

                    convertedIdcs.append(p2)

                    p3 = [0] * 4
                    for k in range(0, limit_influence):
                        _joint_id = idcs[amt + j + arr_offset_2]

                        # GLTF Doesnt like indices to 0 weights
                        w = 0.0

                        if jointWeightsInterpolation == UsdGeom.Tokens.vertex:

                            w = jointWeights[(_joint_id * num_weights_influences) + k]
                        elif jointWeightsInterpolation == UsdGeom.Tokens.constant:
                            w = jointWeights[0]

                        if w == 0.0:
                            continue

                        _mesh_joint_index = -1
                        if jointIndicesInterpolation == UsdGeom.Tokens.vertex:
                            _mesh_joint_index = jointIdcs[
                                (_joint_id * num_joints_influences) + k
                            ]
                        elif jointIndicesInterpolation == UsdGeom.Tokens.constant:
                            # TODO: Should check exists
                            _mesh_joint_index = jointIdcs[0]

                        if meshJoints:
                            mesh_joint_path = Sdf.Path(
                                meshJoints[_mesh_joint_index].split("/")[-1]
                            )
                            gltfIdx, gltfBone = converter.joint_map[skeleton_path][
                                mesh_joint_path
                            ]
                            gltfIdx = converter.skeleton_map[skeleton_path][gltfIdx]
                            p3[k] = gltfIdx

                    convertedIdcs.append(p3)

        start_byte_len = len(converter.maindata_bytearray)

        for idx in convertedIdcs:
            for value in idx:
                converter.maindata_bytearray.extend(struct.pack("h", value))

        bytelen = len(converter.maindata_bytearray) - start_byte_len

        mins = [
            min([operator.itemgetter(i)(idx) for idx in convertedIdcs])
            for i in range(4)
        ]
        maxs = [
            max([operator.itemgetter(i)(idx) for idx in convertedIdcs])
            for i in range(4)
        ]

        bufferview_id = len(converter.gltfDoc.bufferViews)
        joints_accessor_id = len(converter.gltfDoc.accessors)

        converter.gltfDoc.bufferViews.append(
            BufferView(
                buffer=0,
                byteOffset=start_byte_len,
                byteLength=bytelen,
                target=BufferTarget.ARRAY_BUFFER.value,
            )
        )
        converter.gltfDoc.accessors.append(
            Accessor(
                bufferView=bufferview_id,
                byteOffset=0,
                componentType=ComponentType.UNSIGNED_SHORT.value,
                count=len(convertedIdcs),
                type=AccessorType.VEC4.value,
                min=mins,
                max=maxs,
            )
        )

        converter.currentByteOffset += bytelen
        converter.totalByteLen += bytelen

        gltfAttributes.JOINTS_0 = joints_accessor_id

        # - Weights

        limit_influence = min(num_weights_influences, 4)
        if num_weights_influences > 4:
            logger.warning(
                "Mesh limited to max of 4 weights per vertex. Weight influences were {0}".format(
                    num_weights_influences
                )
            )
        weight_interpolation = jointWeightsPrimvar.GetInterpolation()

        convertedWeights = []
        if can_use_mesh_indices:
            if weight_interpolation == UsdGeom.Tokens.vertex:
                for i in range(len(jointWeights)):

                    p1 = [0] * 4
                    for k in range(0, limit_influence):
                        p1[k] = jointWeights[(i * num_weights_influences) + k]
                    # p1[0] = jointWeights[i]

                    if needs_normalization and converter.normalize_weights:
                        p1 = _normalize_weight(p1)
                    convertedWeights.append(p1)

            elif weight_interpolation == UsdGeom.Tokens.constant:
                for i in range(len(jointIdcs)):
                    p1 = [0] * 4
                    for k in range(0, limit_influence):
                        p1[k] = jointWeights[0]

                    if needs_normalization and converter.normalize_weights:
                        p1 = _normalize_weight(p1)
                    convertedWeights.append(p1)
        else:
            if weight_interpolation == UsdGeom.Tokens.vertex:
                for si in subset_idc:
                    f = faces[si]
                    amt = face_offset[si]
                    for j in range(1, f - 1):

                        p1 = [0] * 4
                        for k in range(0, limit_influence):
                            p1[k] = jointWeights[
                                (idcs[amt] * num_weights_influences) + k
                            ]

                        if needs_normalization and converter.normalize_weights:
                            p1 = _normalize_weight(p1)
                        convertedWeights.append(p1)

                        p2 = [0] * 4
                        for k in range(0, limit_influence):
                            p2[k] = jointWeights[
                                (idcs[amt + j + arr_offset_1] * num_weights_influences)
                                + k
                            ]

                        if needs_normalization and converter.normalize_weights:
                            p2 = _normalize_weight(p2)
                        convertedWeights.append(p2)

                        p3 = [0] * 4
                        for k in range(0, limit_influence):
                            p3[k] = jointWeights[
                                (idcs[amt + j + arr_offset_2] * num_weights_influences)
                                + k
                            ]

                        if needs_normalization and converter.normalize_weights:
                            p3 = _normalize_weight(p3)
                        convertedWeights.append(p3)
            elif weight_interpolation == UsdGeom.Tokens.constant:
                for si in subset_idc:
                    f = faces[si]
                    amt = face_offset[si]
                    for j in range(1, f - 1):

                        p1 = [0] * 4
                        for k in range(0, limit_influence):
                            p1[k] = jointWeights[0]

                        if needs_normalization and converter.normalize_weights:
                            p1 = _normalize_weight(p1)
                        convertedWeights.append(p1)

                        p2 = [0] * 4
                        for k in range(0, limit_influence):
                            p2[k] = jointWeights[0]

                        if needs_normalization and converter.normalize_weights:
                            p2 = _normalize_weight(p2)
                        convertedWeights.append(p2)

                        p3 = [0] * 4
                        for k in range(0, limit_influence):
                            p3[k] = jointWeights[0]

                        if needs_normalization and converter.normalize_weights:
                            p3 = _normalize_weight(p3)
                        convertedWeights.append(p3)

        start_byte_len = len(converter.maindata_bytearray)

        for idx in convertedWeights:
            for value in idx:
                converter.maindata_bytearray.extend(struct.pack("f", value))

        bytelen = len(converter.maindata_bytearray) - start_byte_len

        mins = [
            min([operator.itemgetter(i)(idx) for idx in convertedWeights])
            for i in range(4)
        ]
        maxs = [
            max([operator.itemgetter(i)(idx) for idx in convertedWeights])
            for i in range(4)
        ]

        bufferview_id = len(converter.gltfDoc.bufferViews)
        weights_accessor_id = len(converter.gltfDoc.accessors)

        converter.gltfDoc.bufferViews.append(
            BufferView(
                buffer=0,
                byteOffset=start_byte_len,
                byteLength=bytelen,
                target=BufferTarget.ARRAY_BUFFER.value,
            )
        )
        converter.gltfDoc.accessors.append(
            Accessor(
                bufferView=bufferview_id,
                byteOffset=0,
                componentType=ComponentType.FLOAT.value,
                count=len(convertedWeights),
                type=AccessorType.VEC4.value,
                min=mins,
                max=maxs,
            )
        )

        converter.currentByteOffset += bytelen
        converter.totalByteLen += bytelen

        gltfAttributes.WEIGHTS_0 = weights_accessor_id


def add_skeleton_rig(converter, usd_skeleton):

    skeleton_name = usd_skeleton.GetPrim().GetName()
    skeleton_path = usd_skeleton.GetPrim().GetPrimPath()

    # Create nodes
    joints = usd_skeleton.GetJointsAttr().Get()
    jointNames = usd_skeleton.GetJointNamesAttr().Get()
    restTransforms = usd_skeleton.GetRestTransformsAttr().Get()

    # Skeleton root
    # TODO, technically the xform pass adds one... use that
    skel_root_id = len(converter.gltfDoc.nodes)

    gltfSkelRootNode = Node()
    gltfSkelRootNode.name = "{0}_root".format(skeleton_name)

    converter.gltfDoc.nodes.append(gltfSkelRootNode)

    logger.debug(
        " - skeleton[{0}]: {1} : {2}".format(skel_root_id, skeleton_path, skeleton_name)
    )

    # Joints

    joint_node_ids = []
    joint_nodes = []

    root_joints_ids = []

    # joint_map = {}

    converter.joint_map[skeleton_path] = {}

    for idx, joint in enumerate(joints):
        node_id = len(converter.gltfDoc.nodes)

        joint_path = Sdf.Path(jointNames[idx])

        gltfNode = Node()
        gltfNode.name = "{0}_joint_{1}".format(skeleton_name, joint_path.name)

        components = UsdSkel.DecomposeTransform(restTransforms[idx])

        gltfNode.translation = common._Vec3ToArray(components[0])
        gltfNode.rotation = common._QuatToArray(components[1])
        gltfNode.scale = common._Vec3ToArray(components[2])

        converter.joint_map[skeleton_path][joint_path] = (node_id, gltfNode)
        converter.skeleton_map[Sdf.Path(joint)] = (node_id, gltfNode)

        converter.gltfDoc.nodes.append(gltfNode)

        joint_node_ids.append(node_id)
        joint_nodes.append(gltfNode)

        # TODO: Improve this
        if "/" not in joint:
            root_joints_ids.append(node_id)

        logger.debug("    - joint: " + gltfNode.name)

    # Handle heirachy

    for idx, joint in enumerate(joints):
        node_id = len(converter.gltfDoc.nodes)

        joint_path = Sdf.Path(joint)

        # TODO: This is iterated too many times
        # TODO: We can probably store the children earlier and apply them
        # instead of looping through this again.
        for joint_parent in joint_path.GetAncestorsRange():

            jp_name = Sdf.Path(joint_parent.name)
            jpp_name = Sdf.Path(joint_parent.GetParentPath().name)

            if joint_parent.GetParentPath().name == ".":
                continue
            j_p_node_id, j_p_node = converter.joint_map[skeleton_path][jpp_name]
            j_node_id, j_node = converter.joint_map[skeleton_path][jp_name]

            if j_p_node.children == None:
                j_p_node.children = []

            if j_node_id not in j_p_node.children:
                j_p_node.children += [j_node_id]

    gltfSkelRootNode.children = root_joints_ids

    return (skel_root_id, gltfSkelRootNode, joint_node_ids)


def add_skin(converter, usd_skeleton, skel_id, joint_nodes):

    skin_name = "{0}_skin".format(usd_skeleton.GetPrim().GetName())
    skin_id = len(converter.skins)

    ibm_accessor_id = -1

    bind_transforms_attr = usd_skeleton.GetBindTransformsAttr()

    if bind_transforms_attr.HasValue():

        rawBT = bind_transforms_attr.Get()

        bindTransforms = []

        for bt in rawBT:
            bindTransforms.append(common._ConvertMatrix(bt.GetInverse()))

        start_byte_len = len(converter.ibt_bytearray)
        logger.debug("bindTransforms: {}".format(bindTransforms))

        for norm in bindTransforms:
            for value in norm:
                converter.ibt_bytearray.extend(struct.pack("f", value))

        bytelen = len(converter.ibt_bytearray) - start_byte_len

        mins = [
            min([operator.itemgetter(i)(norm) for norm in bindTransforms])
            for i in range(16)
        ]
        maxs = [
            max([operator.itemgetter(i)(norm) for norm in bindTransforms])
            for i in range(16)
        ]

        bufferview_id = len(converter.gltfDoc.bufferViews)
        ibm_accessor_id = len(converter.gltfDoc.accessors)

        converter.gltfDoc.bufferViews.append(
            BufferView(buffer=2, byteOffset=start_byte_len, byteLength=bytelen)
        )
        converter.gltfDoc.accessors.append(
            Accessor(
                bufferView=bufferview_id,
                byteOffset=0,
                componentType=ComponentType.FLOAT.value,
                count=len(bindTransforms),
                type=AccessorType.MAT4.value,
                min=mins,
                max=maxs,
            )
        )

        converter.currentByteOffset += bytelen
        converter.totalByteLen += bytelen

    gltfSkin = Skin(
        skeleton=skel_id,
        joints=joint_nodes,
        name=skin_name,
        inverseBindMatrices=ibm_accessor_id,
    )

    return skin_id, gltfSkin


def _addSkeletonAnimation(converter, usd_skeleton, usd_skel_animation):

    # Get joints
    joints = usd_skel_animation.GetJointsAttr().Get()

    skeleton_path = usd_skeleton.GetPrim().GetPrimPath()

    # Get TRS

    translations_attr = usd_skel_animation.GetTranslationsAttr()
    rotations_attr = usd_skel_animation.GetRotationsAttr()
    scales_attr = usd_skel_animation.GetScalesAttr()

    # Time

    times_accessor_id = -1

    if translations_attr.ValueMightBeTimeVarying():
        positions_ts = translations_attr.GetTimeSamples()
        start_byte_len = len(converter.animated_xforms_bytearray)

        min_time = positions_ts[0] / converter.FPS
        max_time = positions_ts[0] / converter.FPS

        for sample in positions_ts:
            gltf_time = sample / converter.FPS

            min_time = min(min_time, gltf_time)
            max_time = max(max_time, gltf_time)

            converter.animated_xforms_bytearray.extend(struct.pack("f", gltf_time))

        # Times
        bytelen = len(converter.animated_xforms_bytearray) - start_byte_len
        bufferview_id, bufferview = converter.animated_buffer_view()
        bufferview.byteLength += bytelen

        # Times accessor
        times_accessor_id = len(converter.gltfDoc.accessors)

        converter.gltfDoc.accessors.append(
            Accessor(
                bufferView=bufferview_id,
                byteOffset=0,
                componentType=ComponentType.FLOAT.value,
                count=len(positions_ts),
                type=AccessorType.SCALAR.value,
                min=[min_time],
                max=[max_time],
            )
        )

    joint_channels = []
    joint_samplers = []

    # Start storing data
    _animation_name = common._GetAnimationNameFromUSD(usd_skel_animation.GetPrim())
    if not _animation_name:
        _animation_name = "{0}_anim".format("skeleton")

    _anim = converter.get_named_animation(_animation_name)

    # - GLTF Extras
    if not _anim.extras:
        _anim.extras = {}

    common._HandleExtras(usd_skel_animation.GetPrim(), _anim)

    # Joints

    for i, joint in enumerate(joints):
        joint_path = Sdf.Path(joint)
        joint_name = Sdf.Path(joint_path.name)

        if joint_name not in converter.joint_map[skeleton_path]:
            continue

        joint_id, joint_node = converter.joint_map[skeleton_path][joint_name]

        if translations_attr.ValueMightBeTimeVarying():
            positions_ts = translations_attr.GetTimeSamples()

            # Animation

            start_byte_len = len(converter.animated_xforms_bytearray)

            joint_translations = []

            joint_rotations = []

            joint_scales = []

            # TODO: Assumes all TRS are same sample count
            for sample in positions_ts:
                # Tra
                tra = translations_attr.Get(sample)
                joint_translations.append(tra[i])

                # Rot
                rot = rotations_attr.Get(sample)
                joint_rotations.append(common._QuatToArray(rot[i]))

                # Scale
                scale = scales_attr.Get(sample)
                joint_scales.append(scale[i])

            translations = joint_translations
            rotations = joint_rotations
            scales = joint_scales

            # Convert translations
            bytes_translation_start = len(converter.animated_xforms_bytearray)
            for loc in translations:
                for value in loc:
                    converter.animated_xforms_bytearray.extend(struct.pack("f", value))

            minstra = [
                min([operator.itemgetter(i)(vertex) for vertex in translations])
                for i in range(3)
            ]
            maxstra = [
                max([operator.itemgetter(i)(vertex) for vertex in translations])
                for i in range(3)
            ]

            # Initial translation TODO: Should be rest pose?
            joint_node.translation = common._Vec3ToArray(translations[0])

            # Convert rotations
            bytes_rotation_start = len(converter.animated_xforms_bytearray)
            for rot in rotations:
                for value in rot:
                    converter.animated_xforms_bytearray.extend(struct.pack("f", value))

            minsrot = [
                min([operator.itemgetter(i)(rot) for rot in rotations])
                for i in range(4)
            ]
            maxsrot = [
                max([operator.itemgetter(i)(rot) for rot in rotations])
                for i in range(4)
            ]

            # Initial rotation TODO: Should be rest pose?
            joint_node.rotation = rotations[0]

            # Convert scales
            bytes_scale_start = len(converter.animated_xforms_bytearray)
            for scale in scales:
                for value in scale:
                    converter.animated_xforms_bytearray.extend(struct.pack("f", value))

            min_scale = [
                min([operator.itemgetter(i)(scale) for scale in scales])
                for i in range(3)
            ]
            max_scale = [
                max([operator.itemgetter(i)(scale) for scale in scales])
                for i in range(3)
            ]

            # Initial scale TODO: Should be rest pose?
            joint_node.scale = common._Vec3ToArray(scales[0])

            # Store
            bytelen = len(converter.animated_xforms_bytearray) - start_byte_len
            bufferview_id = len(converter.gltfDoc.bufferViews)

            converter.gltfDoc.bufferViews.append(
                BufferView(buffer=1, byteOffset=start_byte_len, byteLength=bytelen)
            )

            # Translation accessor
            pos_accessor_id = len(converter.gltfDoc.accessors)
            converter.gltfDoc.accessors.append(
                Accessor(
                    bufferView=bufferview_id,
                    byteOffset=bytes_translation_start - start_byte_len,
                    componentType=ComponentType.FLOAT.value,
                    count=len(translations),
                    type=AccessorType.VEC3.value,
                    min=minstra,
                    max=maxstra,
                )
            )

            # Rotation accessor
            rot_accessor_id = len(converter.gltfDoc.accessors)
            converter.gltfDoc.accessors.append(
                Accessor(
                    bufferView=bufferview_id,
                    byteOffset=bytes_rotation_start - start_byte_len,
                    componentType=ComponentType.FLOAT.value,
                    count=len(rotations),
                    type=AccessorType.VEC4.value,
                    min=minsrot,
                    max=maxsrot,
                )
            )

            # Scale accessor
            sca_accessor_id = len(converter.gltfDoc.accessors)
            converter.gltfDoc.accessors.append(
                Accessor(
                    bufferView=bufferview_id,
                    byteOffset=bytes_scale_start - start_byte_len,
                    componentType=ComponentType.FLOAT.value,
                    count=len(scales),
                    type=AccessorType.VEC3.value,
                    min=min_scale,
                    max=max_scale,
                )
            )

            if times_accessor_id < 0:
                print("TODO: We should prevent this")

            sampler_offset = len(_anim.samplers)

            # Translations
            sampler_id = len(joint_samplers) + sampler_offset
            joint_samplers.append(
                AnimationSampler(
                    input=times_accessor_id,
                    interpolation=converter.interpolation,
                    output=pos_accessor_id,
                )
            )
            joint_channels.append(
                Channel(
                    sampler=sampler_id, target=Target(node=joint_id, path="translation")
                )
            )

            # Rotations
            sampler_id = len(joint_samplers) + sampler_offset
            joint_samplers.append(
                AnimationSampler(
                    input=times_accessor_id,
                    interpolation=converter.interpolation,
                    output=rot_accessor_id,
                )
            )
            joint_channels.append(
                Channel(
                    sampler=sampler_id, target=Target(node=joint_id, path="rotation")
                )
            )

            # Rotations
            sampler_id = len(joint_samplers) + sampler_offset
            joint_samplers.append(
                AnimationSampler(
                    input=times_accessor_id,
                    interpolation=converter.interpolation,
                    output=sca_accessor_id,
                )
            )
            joint_channels.append(
                Channel(sampler=sampler_id, target=Target(node=joint_id, path="scale"))
            )

    if len(joint_channels) > 0:
        _anim.channels += joint_channels
        _anim.samplers += joint_samplers

    return (-1, None)
