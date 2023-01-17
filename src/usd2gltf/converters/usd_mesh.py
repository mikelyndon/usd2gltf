from pxr import Usd, UsdGeom, UsdShade, Sdf

import struct
import operator
import math
import logging

from gltflib import (
    GLTF,
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


def convert(converter, usd_mesh):
    faces_attr = usd_mesh.GetFaceVertexCountsAttr()
    faces = common._GetStaticValue(faces_attr)

    subsets = UsdGeom.Subset.GetAllGeomSubsets(usd_mesh)

    remaining_idcs = UsdGeom.Subset.GetUnassignedIndices(subsets, len(faces))
    has_remainder = False
    if len(remaining_idcs) > 0:
        has_remainder = True

    num_subsets = len(subsets)

    if has_remainder:
        num_subsets += 1  # All subsets plus remainder

    primitives = []
    for i in range(num_subsets):
        primitives.append(Primitive(attributes=Attributes()))

    prim = usd_mesh.GetPrim()
    ppath = prim.GetPrimPath()
    mesh_name = prim.GetName()

    mesh_id = len(converter.gltfDoc.meshes)

    logger.debug(" - mesh[{0}]: {1} : {2}".format(mesh_id, ppath, mesh_name))

    points_attr = usd_mesh.GetPointsAttr()
    points = common._GetStaticValue(points_attr)

    idcs_attr = usd_mesh.GetFaceVertexIndicesAttr()
    idcs = common._GetStaticValue(idcs_attr)

    subset_idcs = [[]] * num_subsets
    for i, s in enumerate(subsets):
        subset_idcs[i] = s.GetIndicesAttr().Get()

    # TODO: Doesnt work if subset indices are not faces
    if has_remainder:
        subset_idcs[-1] = remaining_idcs

    if not points:
        logger.debug("No points in mesh")
        return (None, None)

    if len(points) <= 0:
        logger.debug("No points in mesh")
        return

    # Primvars
    # Done first because we need to know if any are faceVarying to force NO indices

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

    can_use_mesh_indices = converter.use_mesh_indices

    # Check if any primvar has faceVarying interpolation
    # GLTF Cannot support per face accessors with index based geometry

    if can_use_mesh_indices:
        if converter.convert_normals:
            if "faceVarying" in usd_mesh.GetNormalsInterpolation():
                logger.info("Mesh has faceVarying Normals. Forcing to NOT use indices")
                # print("Mesh has faceVarying Normals. Forcing to NOT use indices")
                can_use_mesh_indices = False

    if can_use_mesh_indices:
        if converter.convert_texcoords:
            if len(texcoords) <= 2:
                for stCoords in texcoords:
                    if stCoords.IsDefined():
                        if "faceVarying" in stCoords.GetInterpolation():
                            can_use_mesh_indices = False
                            logger.info(
                                "Mesh has faceVarying UVs. Forcing to NOT use indices"
                            )
                            # print("Mesh has faceVarying UVs. Forcing to NOT use indices")
                            break

    # Needed so we can index point indices from anywhere
    face_offset = []
    amt = 0
    for f in faces:
        face_offset.append(amt)
        amt += f

    arr_offset_1 = 0
    arr_offset_2 = 1

    if isLeftHanded:
        arr_offset_1 = 1
        arr_offset_2 = 0

    for sub_idx in range(num_subsets):
        subset_idc = subset_idcs[sub_idx]
        logger.debug("  - subset[{0}]".format(sub_idx))
        # Points

        vertices = []

        start_byte_len = len(converter.maindata_bytearray)

        if can_use_mesh_indices:
            vertices = points
        else:
            for si in subset_idc:
                f = faces[si]
                amt = face_offset[si]
                if isLeftHanded:
                    for i in range(1, f - 1):
                        vertices.append(points[idcs[amt]])
                        vertices.append(points[idcs[amt + i + 1]])
                        vertices.append(points[idcs[amt + i + 0]])

                else:
                    for i in range(1, f - 1):
                        vertices.append(points[idcs[amt]])
                        vertices.append(points[idcs[amt + i + 0]])
                        vertices.append(points[idcs[amt + i + 1]])

        for vertex in vertices:
            for value in vertex:
                converter.maindata_bytearray.extend(struct.pack("f", value))

        bytelen = len(converter.maindata_bytearray) - start_byte_len

        mins = [
            min([operator.itemgetter(i)(vertex) for vertex in vertices])
            for i in range(3)
        ]
        maxs = [
            max([operator.itemgetter(i)(vertex) for vertex in vertices])
            for i in range(3)
        ]

        bufferview_id = len(converter.gltfDoc.bufferViews)
        pos_accessor_id = len(converter.gltfDoc.accessors)

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
                count=len(vertices),
                type=AccessorType.VEC3.value,
                min=mins,
                max=maxs,
            )
        )

        converter.currentByteOffset += bytelen
        converter.totalByteLen += bytelen

        primitives[sub_idx].attributes.POSITION = pos_accessor_id
        # Indices
        if can_use_mesh_indices:
            start_byte_len = len(converter.maindata_bytearray)

            indices = []

            # TODO: Dont we need orientation swizzle?
            for si in subset_idc:
                f = faces[si]
                amt = face_offset[si]
                for i in range(1, f - 1):
                    indices.append(idcs[amt])
                    if isLeftHanded:
                        indices.append(idcs[amt + i + 1])
                        indices.append(idcs[amt + i + 0])
                    else:
                        indices.append(idcs[amt + i + 0])
                        indices.append(idcs[amt + i + 1])

            for index in indices:
                # for value in index:
                converter.maindata_bytearray.extend(struct.pack("H", index))

            # TODO: FIX
            # Hack so that byte offset is happy....
            if len(indices) % 2 == 1:
                converter.maindata_bytearray.extend(struct.pack("H", 0))

            bytelen = len(converter.maindata_bytearray) - start_byte_len

            mins = indices[0]
            maxs = indices[0]
            for index in indices:
                mins = min(index, mins)
                maxs = max(index, maxs)

            mins = [mins]
            maxs = [maxs]

            bufferview_id = len(converter.gltfDoc.bufferViews)
            indices_accessor_id = len(converter.gltfDoc.accessors)

            converter.gltfDoc.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=start_byte_len,
                    byteLength=bytelen,
                    target=BufferTarget.ELEMENT_ARRAY_BUFFER.value,
                )
            )
            converter.gltfDoc.accessors.append(
                Accessor(
                    bufferView=bufferview_id,
                    byteOffset=0,
                    componentType=5123,
                    count=len(indices),
                    type=AccessorType.SCALAR.value,
                    min=mins,
                    max=maxs,
                )
            )

            converter.currentByteOffset += bytelen
            converter.totalByteLen += bytelen

            primitives[sub_idx].indices = indices_accessor_id

        # Normals

        if converter.convert_normals:

            normals_accessor_id = -1

            normal_attr = usd_mesh.GetNormalsAttr()

            if normal_attr.HasValue():

                rawNormals = common._GetStaticValue(normal_attr)

                norms = []

                start_byte_len = len(converter.maindata_bytearray)

                if not can_use_mesh_indices:
                    for si in subset_idc:
                        f = faces[si]
                        amt = face_offset[si]
                        for i in range(1, f - 1):
                            if "vertex" in usd_mesh.GetNormalsInterpolation():
                                nx = rawNormals[idcs[amt]]
                                ny = rawNormals[idcs[amt + i + arr_offset_1]]
                                nz = rawNormals[idcs[amt + i + arr_offset_2]]

                                if converter.normalize_normals:
                                    nx = common._Normalize(nx)
                                    ny = common._Normalize(ny)
                                    nz = common._Normalize(nz)

                                norms.append(nx)
                                norms.append(ny)
                                norms.append(nz)
                            elif "faceVarying" in usd_mesh.GetNormalsInterpolation():
                                nx = rawNormals[amt]
                                ny = rawNormals[amt + i + arr_offset_1]
                                nz = rawNormals[amt + i + arr_offset_2]

                                if converter.normalize_normals:
                                    nx = common._Normalize(nx)
                                    ny = common._Normalize(ny)
                                    nz = common._Normalize(nz)

                                norms.append(nx)
                                norms.append(ny)
                                norms.append(nz)
                else:
                    if "vertex" in usd_mesh.GetNormalsInterpolation():
                        for i in range(len(rawNormals)):
                            nx = rawNormals[i]
                            if converter.normalize_normals:
                                nx = common._Normalize(nx)
                            norms.append(nx)
                    elif "faceVarying" in usd_mesh.GetNormalsInterpolation():
                        # TODO: Apparently, GLTF cannot have indexed points/vertices
                        # and element array buffers.... We will need to know this before hand
                        # and split up points manually (removing perf from indices...
                        # but apparently no other way)
                        logger.warning(
                            "GLTF Doesnt support faceVarying normals with indices turned on... Approximating"
                        )
                        # print("GLTF Doesnt support faceVarying normals with indices turned on... Approximating")
                        norms = [0] * len(points)

                        for si in subset_idc:
                            f = faces[si]
                            amt = face_offset[si]
                            for i in range(1, f - 1):
                                nx = rawNormals[amt]
                                ny = rawNormals[amt + i + arr_offset_1]
                                nz = rawNormals[amt + i + arr_offset_2]
                                ix = idcs[amt]
                                iy = idcs[amt + i + arr_offset_1]
                                iz = idcs[amt + i + arr_offset_2]

                                if converter.normalize_normals:
                                    nx = common._Normalize(nx)
                                    ny = common._Normalize(ny)
                                    nz = common._Normalize(nz)

                                norms[ix] = nx
                                norms[iy] = ny
                                norms[iz] = nz

                for norm in norms:
                    for value in norm:
                        converter.maindata_bytearray.extend(struct.pack("f", value))

                bytelen = len(converter.maindata_bytearray) - start_byte_len

                mins = [
                    min([operator.itemgetter(i)(norm) for norm in norms])
                    for i in range(3)
                ]
                maxs = [
                    max([operator.itemgetter(i)(norm) for norm in norms])
                    for i in range(3)
                ]

                bufferview_id = len(converter.gltfDoc.bufferViews)
                normals_accessor_id = len(converter.gltfDoc.accessors)

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
                        count=len(norms),
                        type=AccessorType.VEC3.value,
                        min=mins,
                        max=maxs,
                    )
                )

                converter.currentByteOffset += bytelen
                converter.totalByteLen += bytelen

                primitives[sub_idx].attributes.NORMAL = normals_accessor_id

        # Material Binding
        # Done before primvars for texcoord lookup

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
        for s in range(num_subsets):
            if s < len(subsets):
                sub = subsets[s]
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

        # TODO: (bjs: 3/May/22) Unsure why this was added, but it breaks subsets... Removed for now
        # print( "num_subsets = {}".format(num_subsets) )
        # print( "sub_idx = {}".format(sub_idx) )
        # if num_subsets > 1:
        # print( "len_subset = {}".format(len(subsets)) )
        # sub = subsets[sub_idx]
        # mat_api = UsdShade.MaterialBindingAPI(sub)
        # mat_path = mat_api.ComputeBoundMaterial()[0].GetPath()

        logger.debug("    - material bound: {0}".format(mat_path))

        # - Texcoords
        if converter.convert_texcoords:
            if len(texcoords) <= 2:
                uvs_accessor_id = -1

                for stCoords in texcoords:

                    tx_name = stCoords.GetName().split(":")[1]

                    tex_idx = 0

                    if (
                        mat_path in converter.material_uv_indices
                        and tx_name in converter.material_uv_indices[mat_path]
                    ):
                        tex_idx = converter.material_uv_indices[mat_path][tx_name]
                    else:
                        continue

                    if stCoords.IsDefined():
                        rawUVS = common._GetStaticValue(stCoords)
                        convertedUVS = []

                        start_byte_len = len(converter.maindata_bytearray)

                        logger.debug(
                            "   - texcoord: {0} : {1}".format(tx_name, tex_idx)
                        )

                        if not can_use_mesh_indices:
                            for si in subset_idc:
                                f = faces[si]
                                amt = face_offset[si]
                                for i in range(1, f - 1):
                                    if "vertex" in stCoords.GetInterpolation():
                                        convertedUVS.append(
                                            common._ConvertUVs(rawUVS[idcs[amt]])
                                        )
                                        convertedUVS.append(
                                            common._ConvertUVs(
                                                rawUVS[idcs[amt + i + arr_offset_1]]
                                            )
                                        )
                                        convertedUVS.append(
                                            common._ConvertUVs(
                                                rawUVS[idcs[amt + i + arr_offset_2]]
                                            )
                                        )
                                    elif "faceVarying" in stCoords.GetInterpolation():
                                        convertedUVS.append(
                                            common._ConvertUVs(rawUVS[amt])
                                        )
                                        convertedUVS.append(
                                            common._ConvertUVs(
                                                rawUVS[amt + i + arr_offset_1]
                                            )
                                        )
                                        convertedUVS.append(
                                            common._ConvertUVs(
                                                rawUVS[amt + i + arr_offset_2]
                                            )
                                        )
                        else:
                            if "vertex" in stCoords.GetInterpolation():
                                for i in range(len(rawUVS)):
                                    convertedUVS.append(common._ConvertUVs(rawUVS[i]))
                            elif "faceVarying" in stCoords.GetInterpolation():
                                # TODO: Apparently, GLTF cannot have indexed points/vertices and element array buffers.... We will need to know this before hand
                                # and split up points manually (removing perf from indices... but apparently no other way)
                                logger.warning("GLTF Doesnt support faceVarying uvs with indices turned on... Approximating")
                                # print(
                                #     "GLTF Doesnt support faceVarying uvs with indices turned on... Approximating"
                                # )
                                convertedUVS = [0] * len(points)

                                for si in subset_idc:
                                    f = faces[si]
                                    amt = face_offset[si]
                                    for i in range(1, f - 1):
                                        ix = idcs[amt]
                                        iy = idcs[amt + i + arr_offset_1]
                                        iz = idcs[amt + i + arr_offset_2]
                                        convertedUVS[ix] = common._ConvertUVs(
                                            rawUVS[amt]
                                        )
                                        convertedUVS[iy] = common._ConvertUVs(
                                            rawUVS[amt + i + arr_offset_1]
                                        )
                                        convertedUVS[iz] = common._ConvertUVs(
                                            rawUVS[amt + i + arr_offset_2]
                                        )

                        for vertex in convertedUVS:
                            for value in vertex:
                                converter.maindata_bytearray.extend(
                                    struct.pack("f", value)
                                )

                        bytelen = len(converter.maindata_bytearray) - start_byte_len

                        mins = [
                            min(
                                [
                                    operator.itemgetter(i)(vertex)
                                    for vertex in convertedUVS
                                ]
                            )
                            for i in range(2)
                        ]
                        maxs = [
                            max(
                                [
                                    operator.itemgetter(i)(vertex)
                                    for vertex in convertedUVS
                                ]
                            )
                            for i in range(2)
                        ]

                        bufferview_id = len(converter.gltfDoc.bufferViews)
                        uvs_accessor_id = len(converter.gltfDoc.accessors)

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
                                count=len(convertedUVS),
                                type=AccessorType.VEC2.value,
                                min=mins,
                                max=maxs,
                            )
                        )

                        converter.currentByteOffset += bytelen
                        converter.totalByteLen += bytelen

                        if tex_idx == 0:
                            primitives[sub_idx].attributes.TEXCOORD_0 = uvs_accessor_id
                        else:
                            primitives[sub_idx].attributes.TEXCOORD_1 = uvs_accessor_id

    # end if subsets

    gltfMesh = Mesh(primitives=primitives)
    gltfMesh.name = mesh_name

    # - GLTF Extras
    # gltfMesh.extras = {}

    common._HandleExtras(prim, gltfMesh)

    if gltfMesh.extras:
        logger.debug("extras: {}".format(str(gltfMesh.extras)))

    converter.gltfDoc.meshes.append(gltfMesh)

    return (mesh_id, gltfMesh)
