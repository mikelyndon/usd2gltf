from pxr import Usd, UsdGeom, UsdShade, Sdf
import numpy as np

import struct
import operator
import math
import logging

from gltflib import (
    Primitive,
    Mesh,
    Attributes,
    BufferView,
    BufferTarget,
    Accessor,
    AccessorType,
    ComponentType,
)

from usd2gltf import common

logger = logging.getLogger(__name__)


def _process_mesh_attribute(
    converter,
    data_array,
    target=BufferTarget.ARRAY_BUFFER.value,
    componentType=ComponentType.FLOAT.value,
    type=AccessorType.VEC3.value,
    min=-1,
    max=1,
):
    """
    helper for the sausage making of gltf buffers and offsets
    """
    byte_offset = len(converter.maindata_bytearray)
    index_binary = data_array.tobytes()
    index_buffer_view = BufferView(
        buffer=0,
        byteOffset=byte_offset,
        byteLength=len(index_binary),
        target=target,
    )

    idx_accessor = Accessor(
        bufferView=len(converter.gltfDoc.bufferViews),
        componentType=componentType,
        count=len(data_array),
        type=type,
        min=min,  # data_array.min(axis=0).tolist(),
        max=max,  # data_array.max(axis=0).tolist(),
    )

    converter.maindata_bytearray.extend(index_binary)
    converter.gltfDoc.bufferViews.append(index_buffer_view)
    converter.gltfDoc.accessors.append(idx_accessor)


def _get_triangulated_attribute(
    count_array, index_array, is_index=False, isLeftHanded=True
):
    return_array = []
    j = 0
    for count in count_array:
        poly = index_array[j: j + count]

        for i in range(count - 2):  # This is a trick for triangulating ngons
            tmp = []
            if isLeftHanded:
                tmp = [poly[0], poly[i + 2], poly[i + 1]]
            else:
                tmp = [poly[0], poly[i + 1], poly[i + 2]]
            if is_index:
                return_array.append(tmp)
            else:
                return_array.extend(tmp)

        j += count

    return return_array


def convert(converter, usd_mesh):
    points_attr = usd_mesh.GetPointsAttr()
    points = common._GetStaticValue(points_attr)

    if not points:
        logger.debug("No points in mesh")
        return (None, None)

    if len(points) <= 0:
        logger.debug("No points in mesh")
        return

    # Get Face Vertex Counts
    faces_attr = usd_mesh.GetFaceVertexCountsAttr()
    faces = common._GetStaticValue(faces_attr)

    # Get Face Vertex Indices
    idcs_attr = usd_mesh.GetFaceVertexIndicesAttr()
    idcs = common._GetStaticValue(idcs_attr)

    # Check for uv, color and tangent attributes

    pv_api = UsdGeom.PrimvarsAPI(usd_mesh)

    primvars = pv_api.GetPrimvars()

    texcoords = []
    colors = []
    tangents = None

    for p in primvars:
        if p.HasAuthoredValue():
            if p.GetTypeName().role == Sdf.ValueRoleNames.TextureCoordinate:
                texcoords.append(p)
            elif p.GetTypeName().role == Sdf.ValueRoleNames.Color:
                color_name = p.GetName().split(":")[1]
                if color_name == "displayColor":
                    colors.insert(0, p)
                else:
                    colors.append(p)

            if "tangents" in p.GetName():
                tangents = p

    # Mesh Attributes

    orientation = usd_mesh.GetOrientationAttr().Get()

    isLeftHanded = orientation == UsdGeom.Tokens.leftHanded

    isDoubleSided = usd_mesh.GetDoubleSidedAttr().Get(Usd.TimeCode.Default())

    # Prim and mesh names/paths

    prim = usd_mesh.GetPrim()
    ppath = prim.GetPrimPath()
    mesh_name = prim.GetName()

    mesh_id = len(converter.gltfDoc.meshes)

    # Maps triangulated indices to untriangulated
    index_map = {}
    j = 0
    for i, count in enumerate(faces):
        tmp = []

        for _ in range(count - 2):  # Same logic as get_triangulated_attribute
            tmp.append(j)
            j += 1
        index_map[i] = tmp

    triangulated_index = _get_triangulated_attribute(
        faces, idcs, is_index=True, isLeftHanded=isLeftHanded
    )

    # Get subsets

    subsets = UsdGeom.Subset.GetAllGeomSubsets(usd_mesh)

    logger.debug(" - mesh[{0}]: {1} : {2}".format(mesh_id, ppath, mesh_name))
    # if len(subsets) > 0:
    #     for subset in subsets:
    #         logger.debug("  - {}".format(subset))

    remaining_idcs = UsdGeom.Subset.GetUnassignedIndices(subsets, len(faces))

    if len(remaining_idcs) > 0:
        # has_remainder = True
        subsets.append(remaining_idcs)

    num_subsets = len(subsets)

    # Create a glTF Mesh Primitive for each subset.
    primitives = []
    for i in range(num_subsets):
        primitives.append(Primitive(attributes=Attributes()))

    # Setup subset indices
    subset_idcs = [[]] * num_subsets
    for i, s in enumerate(subsets):
        if s == remaining_idcs:
            subset_idcs[i] = remaining_idcs
        else:
            subset_idcs[i] = s.GetIndicesAttr().Get()

    can_use_mesh_indices = converter.use_mesh_indices

    # Needed so we can index point indices from anywhere
    face_offset = []
    amt = 0
    for f in faces:
        face_offset.append(amt)
        amt += f

    # Loop through subsets

    for sub_idx, subset_indices in enumerate(subset_idcs):
        logger.debug("  - subset[{}]".format(list(subset_indices)))

        # Indices
        if can_use_mesh_indices:
            start_byte_len = len(converter.maindata_bytearray)

            sub_tri_indices = [index_map[index] for index in subset_indices]
            flattened = [item for sublist in sub_tri_indices for item in sublist]
            sub_tri_indices = np.array(flattened, "uint32")

            sub_index_array = [triangulated_index[index] for index in sub_tri_indices]
            sub_index_array = np.array(sub_index_array, "uint32")
            sub_index_array = sub_index_array.flatten()
            index_array_tri = np.array(list(range(len(sub_index_array))), "uint32")

            _process_mesh_attribute(
                converter,
                index_array_tri,
                min=[int(index_array_tri.min())],
                max=[int(index_array_tri.max())],
                type=AccessorType.SCALAR.value,
                componentType=ComponentType.UNSIGNED_INT.value,
                target=BufferTarget.ELEMENT_ARRAY_BUFFER.value,
            )

            primitives[sub_idx].indices = len(converter.gltfDoc.accessors) - 1

            bytelen = len(converter.maindata_bytearray) - start_byte_len

            converter.currentByteOffset += bytelen
            converter.totalByteLen += bytelen

        # Points

        world_pos = points
        world_pos = [world_pos[int(x)] for x in sub_index_array]
        world_pos = np.array(world_pos, "float32")

        _process_mesh_attribute(
            converter,
            world_pos,
            min=world_pos.min(axis=0).tolist(),
            max=world_pos.max(axis=0).tolist(),
        )
        primitives[sub_idx].attributes.POSITION = len(converter.gltfDoc.accessors) - 1

        # Normals

        if converter.convert_normals:

            normal_attr = usd_mesh.GetNormalsAttr()

            if normal_attr.HasValue():

                rawNormals = common._GetStaticValue(normal_attr)
                normals = []
                if usd_mesh.GetNormalsInterpolation() == "faceVarying":
                    n = _get_triangulated_attribute(
                        faces, rawNormals, is_index=True, isLeftHanded=isLeftHanded
                    )
                    normals = [n[int(x)] for x in sub_tri_indices]
                elif usd_mesh.GetNormalsInterpolation() == "vertex":
                    normals = [rawNormals[int(x)] for x in sub_index_array]
                normals = np.array(normals, "float32")
                normals = normals.reshape(-1, normals.shape[-1])
                _process_mesh_attribute(
                    converter,
                    normals,
                    min=normals.min(axis=0).tolist(),
                    max=normals.max(axis=0).tolist(),
                )

                primitives[sub_idx].attributes.NORMAL = (
                    len(converter.gltfDoc.accessors) - 1
                )

                converter.currentByteOffset += bytelen
                converter.totalByteLen += bytelen

        # - Texcoords
        if converter.convert_texcoords:
            if len(texcoords) <= 2:
                uvs_accessor_id = -1

                for i, stCoords in enumerate(texcoords):

                    tx_name = stCoords.GetName().split(":")[1]

                    tex_idx = 0

                    if stCoords.IsDefined():
                        rawUVS = common._GetStaticValue(stCoords)

                        start_byte_len = len(converter.maindata_bytearray)
                        uvs = []
                        if "faceVarying" in stCoords.GetInterpolation():
                            u = _get_triangulated_attribute(
                                faces, rawUVS, is_index=True, isLeftHanded=isLeftHanded
                            )
                            uvs = [u[int(x)] for x in sub_tri_indices]
                        elif "vertex" in stCoords.GetInterpolation():
                            uvs = [rawUVS[int(x)] for x in sub_index_array]
                        uvs = np.array(uvs, "float32")
                        uvs = uvs.reshape(-1, uvs.shape[-1])
                        flipy_uvs = [[uv[0], 1 - uv[1]] for uv in uvs]
                        flipy_uvs = np.array(flipy_uvs, "float32")
                        _process_mesh_attribute(
                            converter,
                            flipy_uvs,
                            type=AccessorType.VEC2.value,
                            min=flipy_uvs.min(axis=0).tolist(),
                            max=flipy_uvs.max(axis=0).tolist(),
                        )
                        if i == 0:
                            primitives[sub_idx].attributes.TEXCOORD_0 = (
                                len(converter.gltfDoc.accessors) - 1
                            )
                        else:
                            primitives[sub_idx].attributes.TEXCOORD_1 = (
                                len(converter.gltfDoc.accessors) - 1
                            )

                        converter.currentByteOffset += bytelen
                        converter.totalByteLen += bytelen

                        # if tex_idx == 0:
                        #     primitives[sub_idx].attributes.TEXCOORD_0 = uvs_accessor_id
                        # else:
                        #     primitives[sub_idx].attributes.TEXCOORD_1 = uvs_accessor_id

        # Colors

        if converter.convert_colors:
            if len(colors) > 0:

                colors_accessor_id = -1

                # GLTF only supports one color attribute, use first found (Usually displayColor)
                displayColor = colors[0]

                color_name = displayColor.GetName().split(":")[1]

                color_idx = 0

                if displayColor.IsDefined() and displayColor.HasAuthoredValue():
                    rawColors = common._GetStaticValue(displayColor)
                    convertedColors = []

                    start_byte_len = len(converter.maindata_bytearray)

                    logger.debug("   - color: {0} : {1}".format(color_name, color_idx))

                    if "constant" in displayColor.GetInterpolation():
                        convertedColors = [rawColors[0]] * len(sub_index_array)

                    elif "faceVarying" in displayColor.GetInterpolation():
                        cd = _get_triangulated_attribute(
                            faces, rawColors, is_index=True, isLeftHanded=isLeftHanded
                        )
                        convertedColors = [cd[int(x)] for x in sub_tri_indices]

                    elif "vertex" in displayColor.GetInterpolation():
                        convertedColors = [rawColors[int(x)] for x in sub_index_array]
                    convertedColors = np.array(convertedColors, "float32")
                    convertedColors = convertedColors.reshape(
                        -1, convertedColors.shape[-1]
                    )
                    _process_mesh_attribute(
                        converter,
                        convertedColors,
                        min=convertedColors.min(axis=0).tolist(),
                        max=convertedColors.max(axis=0).tolist(),
                    )
                    primitives[sub_idx].attributes.COLOR_0 = (
                        len(converter.gltfDoc.accessors) - 1
                    )

        # Material Binding

        mesh_material_id = -1

        mat_api = UsdShade.MaterialBindingAPI(usd_mesh.GetPrim())

        mat_path = mat_api.ComputeBoundMaterial()[0].GetPath()

        if mat_path in converter.materialMap:
            mesh_material_id = converter.materialMap[mat_path]

            # Handle double sidedness
            # Vochsel Note: USD handles this per mesh, GLTF per material
            # This will break if multiple meshes set sideness with same material
            # We could check for an attribute on the material, but I feel less happy about
            # this solution.
            should_set = (
                not mat_api.ComputeBoundMaterial()[0]
                .GetPrim()
                .HasAttribute("doubleSided")
            )
            if should_set:
                if mesh_material_id >= 0:
                    converter.materials[mesh_material_id].doubleSided = isDoubleSided

        # Set default mesh material for all first
        if mesh_material_id >= 0:
            for p in primitives:
                p.material = mesh_material_id

        # Set material for subset
        # Skip this while figuring out the points and indices basics.
        for sub in subsets:
            try:
                mat_api = UsdShade.MaterialBindingAPI(sub)
                mat_path = mat_api.ComputeBoundMaterial()[0].GetPath()

                if mat_path in converter.materialMap:
                    subset_mat_id = converter.materialMap[mat_path]

                    # Handle double sidedness
                    # Vochsel Note: USD handles this per mesh, GLTF per material
                    # This will break if multiple meshes set sideness with same material
                    # We could check for an attribute on the material, but I feel less happy about
                    # this solution.
                    should_set = (
                        not mat_api.ComputeBoundMaterial()[0]
                        .GetPrim()
                        .HasAttribute("doubleSided")
                    )
                    if should_set:
                        if subset_mat_id >= 0:
                            converter.materials[
                                subset_mat_id
                            ].doubleSided = isDoubleSided

                    primitives[s].material = subset_mat_id
            except Exception:
                pass

        logger.debug("    - material bound: {0}".format(mat_path))

    # end if subsets

    gltfMesh = Mesh(primitives=primitives)
    gltfMesh.name = mesh_name

    # - GLTF Extras
    common._HandleExtras(prim, gltfMesh)

    if gltfMesh.extras:
        logger.debug("extras: {}".format(str(gltfMesh.extras)))

    converter.gltfDoc.meshes.append(gltfMesh)

    return (mesh_id, gltfMesh)
