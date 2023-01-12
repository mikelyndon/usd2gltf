from pxr import Usd, UsdGeom, UsdShade, Sdf

import struct
import operator
import math
import logging

from gltflib import *

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
        num_subsets += 1 # All subsets plus remainder
    
    primitives = []
    for i in range(num_subsets):
        primitives.append( Primitive(attributes=Attributes()) )

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
                color_name = p.GetName().split(':')[1]
                if color_name == "displayColor":
                    colors.insert(0, p)
                else:
                    colors.append(p)

            if 'tangents' in p.GetName():
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
            if 'faceVarying' in usd_mesh.GetNormalsInterpolation():
                print("Mesh has faceVarying Normals. Forcing to NOT use indices")
                can_use_mesh_indices = False

    if can_use_mesh_indices:
        if converter.convert_texcoords:
            if len(texcoords) <= 2:
                for stCoords in texcoords:
                    if stCoords.IsDefined():
                        if 'faceVarying' in stCoords.GetInterpolation():
                            can_use_mesh_indices = False
                            print("Mesh has faceVarying UVs. Forcing to NOT use indices")
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
                converter.maindata_bytearray.extend(struct.pack('f', value))

        bytelen = len(converter.maindata_bytearray) - start_byte_len

        mins = [min([operator.itemgetter(i)(vertex)
                    for vertex in vertices]) for i in range(3)]
        maxs = [max([operator.itemgetter(i)(vertex)
                    for vertex in vertices]) for i in range(3)]

        bufferview_id = len(converter.gltfDoc.bufferViews)
        pos_accessor_id = len(converter.gltfDoc.accessors)

        converter.gltfDoc.bufferViews.append(BufferView(
            buffer=0, byteOffset=start_byte_len, byteLength=bytelen, target=BufferTarget.ARRAY_BUFFER.value))
        converter.gltfDoc.accessors.append(Accessor(bufferView=bufferview_id, byteOffset=0, componentType=ComponentType.FLOAT.value, count=len(
            vertices), type=AccessorType.VEC3.value, min=mins, max=maxs))
        
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
                converter.maindata_bytearray.extend(struct.pack('H', index))

            # TODO: FIX
            # Hack so that byte offset is happy....
            if len(indices) % 2  == 1:
                converter.maindata_bytearray.extend(struct.pack('H', 0))

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

            converter.gltfDoc.bufferViews.append(BufferView(
                buffer=0, byteOffset=start_byte_len, byteLength=bytelen))
            converter.gltfDoc.accessors.append(Accessor(bufferView=bufferview_id, byteOffset=0, componentType=5123, count=len(
                indices), type=AccessorType.SCALAR.value, min=mins, max=maxs))

            converter.currentByteOffset += bytelen
            converter.totalByteLen += bytelen

            primitives[sub_idx].indices = indices_accessor_id

    # end if subsets

    gltfMesh = Mesh(primitives=primitives)
    gltfMesh.name = mesh_name

    # - GLTF Extras
    gltfMesh.extras = {}

    common._HandleExtras(prim, gltfMesh)

    logger.debug(str(gltfMesh.extras))

    converter.gltfDoc.meshes.append(gltfMesh)

    return (mesh_id, gltfMesh)
