from pxr import Usd, UsdShade

import os
import struct
import operator
import shutil
import math
import logging

from gltflib import (
    GLTF,
    Material,
    FileResource,
    Image,
    Texture,
    Sampler,
    OcclusionTextureInfo,
    NormalTextureInfo,
    TextureInfo,
    PBRMetallicRoughness,
)

from usd2gltf import common

logger = logging.getLogger(__name__)

MAT_TYPE = {
    "DIFFUSE": 0,
    "SPECROUGH": 1,
    "NORMAL": 2,
    "OCCLUSION": 3,
}

MAG_CONVERTER = {
    "nearest": 9728,
    "linear": 9729,
}

WRAP_CONVERTER = {
    "black": 33071,  # Not supported in GLTF, clamp instead
    "clamp": 33071,
    "mirror": 33648,
    "repeat": 10497,
}


def add_texture(converter, filepath, sampler=None):

    image_idx = len(converter.images)
    texture_idx = len(converter.textures)
    filename = os.path.basename(filepath)
    basepath = os.path.dirname(filepath)

    if not converter.is_glb:
        shutil.copy(filepath, converter.dirname)
        filepath = os.path.join(converter.dirname, filename)
        basepath = converter.dirname

    image_res = FileResource(filename=filename, basepath=basepath)
    converter.resources.append(image_res)

    gltfImage = None

    # Get extension to build mimeType
    ext = os.path.splitext(os.path.basename(filepath))[1][1:]
    if ext == "jpg":
        ext = "jpeg"

    gltfImage = Image(uri=filename, mimeType="image/" + ext)

    converter.images.append(gltfImage)

    # TODO: Do we need one per image
    # converter.gltfDoc.samplers.append(Sampler())

    gltfTexture = Texture(source=image_idx)

    # Add sampler
    if sampler is not None:
        sampler_idx = len(converter.samplers)

        gltfSampler = Sampler()
        gltfSampler.magFilter = MAG_CONVERTER[sampler["magFilter"]]
        gltfSampler.minFilter = MAG_CONVERTER[sampler["minFilter"]]

        gltfSampler.wrapS = WRAP_CONVERTER[sampler["wrapS"]]
        gltfSampler.wrapT = WRAP_CONVERTER[sampler["wrapT"]]

        converter.samplers.append(gltfSampler)

        gltfTexture.sampler = sampler_idx

    converter.textures.append(gltfTexture)

    return (texture_idx, gltfTexture, filepath)


def traverse_inputs(shader):
    matches = []

    for i in shader.GetInputs():
        if i.HasConnectedSource():
            for s in i.GetConnectedSources():
                if len(s) > 0:
                    source_shader = UsdShade.Shader(s[0].source.GetPrim())
                    matches.append(source_shader)
                    matches += traverse_inputs(source_shader)

    return matches


def handle_texture(
    converter, usd_material, usd_uv_texture, mat_type=MAT_TYPE["DIFFUSE"]
):

    # - Texture requirements
    uv_idx = 0  # UV Sampler index
    texture_id = -1  # Texture sampler index
    usd_texture_scale = 1.0

    transform_offset_authored = False
    uv_transform_offset = [0, 0]
    uv_transform_rotation = 0
    uv_transform_scale = [0, 0]

    # - Get all inputes (primvar reader, transform, etc)
    deps = traverse_inputs(usd_uv_texture)
    deps.append(UsdShade.Shader(usd_uv_texture.GetPrim()))

    for d in deps:
        d_id = d.GetShaderId()

        if d_id == "UsdPrimvarReader_float2":
            # handle uv channel
            uv_map_name = d.GetInput("varname").Get()

            mat_path = usd_material.GetPrim().GetPrimPath()

            if mat_path not in converter.material_uv_indices:
                converter.material_uv_indices[mat_path] = {}

            if uv_map_name in converter.material_uv_indices[mat_path]:
                uv_idx = converter.material_uv_indices[mat_path][uv_map_name]
            else:
                uv_idx = len(converter.material_uv_indices[mat_path])
                logger.debug(
                    "   - UsdPrimvarSampler: " + uv_map_name + " : " + str(uv_idx)
                )

            converter.material_uv_indices[mat_path][uv_map_name] = uv_idx

        elif d_id == "UsdTransform2d":
            # handle uv transform

            for transform_input in d.GetInputs(onlyAuthored=True):
                transform_offset_authored = True

                if transform_input.GetBaseName() == "scale":
                    uv_transform_scale = common._Vec2ToArray(d.GetInput("scale").Get())

                elif transform_input.GetBaseName() == "translation":
                    uv_transform_offset = d.GetInput("translation").Get()
                    uv_transform_offset[1] = 1.0 - (uv_transform_offset[1] - 0.5)

                    uv_transform_offset = common._Vec2ToArray(uv_transform_offset)

                elif transform_input.GetBaseName() == "rotation":
                    uv_transform_rotation = math.radians(d.GetInput("rotation").Get())

        elif d_id == "UsdUVTexture":
            # handle uv texture

            sampling = {
                "magFilter": "linear",
                "minFilter": "linear",
                "wrapS": "repeat",
                "wrapT": "repeat",
            }

            for _input in usd_uv_texture.GetInputs():
                if _input.GetBaseName() == "wrapS":
                    sampling["wrapS"] = d.GetInput("wrapS").Get()
                elif _input.GetBaseName() == "wrapT":
                    sampling["wrapT"] = d.GetInput("wrapT").Get()

            # TODO: This produces a warning about no local paths...
            # str(usd_uv_texture.GetInput("file").Get())[1:-1]
            file_input = usd_uv_texture.GetInput("file").Get()

            if not file_input:
                logger.warning(
                    "Warning texture file input was None: {0} - {1}".format(
                        mat_path, usd_uv_texture.GetPrim().GetName()
                    )
                )
                return None

            filepath = file_input.resolvedPath

            # - Extract USDZ texture
            if "[" in filepath and "]" in filepath:
                zip_path = os.path.basename(filepath.split("[")[0])
                zip_ext = os.path.splitext(zip_path)[0]
                t_dir = os.path.join(converter.temp_dir.name, zip_ext)

                filepath = os.path.join(t_dir, file_input.path)

            if len(filepath) <= 0:
                logger.warning(
                    "Warning texture path was empty: {0} - {1}".format(
                        mat_path, usd_uv_texture.GetPrim().GetName()
                    )
                )
                return

            # Prevent duplicate textures being added to glb
            if filepath in converter.textureMap:
                texture_id = converter.textureMap[filepath]
            else:
                texture_id, gltfTexture, filepath = add_texture(
                    converter, filepath, sampler=sampling
                )

                converter.used_images.append(filepath)
                converter.textureMap[filepath] = texture_id

            # TODO: USD Stores scale as float3-4. Probably need to get magnitude
            if usd_uv_texture.GetInput("scale"):
                # usd_texture_scale = usd_uv_texture.GetInput("scale").Get()[0]
                usd_texture_scale = usd_uv_texture.GetInput("scale").Get().GetLength()

    extensions = {}

    if transform_offset_authored:
        converter.add_extension("KHR_texture_transform")
        extensions["KHR_texture_transform"] = {
            "offset": uv_transform_offset,
            "rotation": uv_transform_rotation,
            "scale": uv_transform_scale,
        }

    if mat_type == MAT_TYPE["NORMAL"]:
        # Scale of normals undefined. USD usually requires scale of (2,2,2,2) and bias of (-1,-1,-1,-1).
        # we could get the multiplication value normalized around (2,2,2,2) however unsure if this is worthwhile...
        return NormalTextureInfo(
            index=texture_id, scale=1.0, texCoord=uv_idx, extensions=extensions
        )
    if mat_type == MAT_TYPE["OCCLUSION"]:
        return OcclusionTextureInfo(
            index=texture_id,
            strength=usd_texture_scale,
            texCoord=uv_idx,
            extensions=extensions,
        )

    return TextureInfo(index=texture_id, texCoord=uv_idx, extensions=extensions)


def convert(converter, usd_material):

    surf = usd_material.GetSurfaceOutput()

    material_id = len(converter.materials)
    ppath = usd_material.GetPrim().GetPrimPath()
    material_name = usd_material.GetPrim().GetName()

    logger.debug(
        " - material[{0}]: {1} : {2}".format(material_id, ppath, material_name)
    )

    gltfMaterial = Material(name=material_name)
    gltfMaterial.pbrMetallicRoughness = PBRMetallicRoughness()
    # gltfMaterial.doubleSided = True

    gltfMaterial.pbrMetallicRoughness.baseColorFactor = [1, 1, 1, 1]
    gltfMaterial.pbrMetallicRoughness.metallicFactor = 0.0
    gltfMaterial.emissiveFactor = [0, 0, 0]
    converter.materials.append(gltfMaterial)

    previewSurface = surf.GetConnectedSource()[0]

    ins = previewSurface.GetInputs()

    has_alpha = False
    has_threshold = False

    for _input in ins:
        if _input == None:
            continue

        # Convinience
        uv_texture = None
        if _input.GetConnectedSource():
            uv_texture = UsdShade.Shader(_input.GetConnectedSource()[0])

        if _input.GetBaseName() == "opacity":

            opacityAttr = _input.GetAttr()
            if opacityAttr.IsValid() and opacityAttr.Get():
                opacity_val = opacityAttr.Get()
                if opacity_val < 1.0:
                    has_alpha = True

                    gltfMaterial.pbrMetallicRoughness.baseColorFactor[3] = opacity_val

                    logger.debug("   - opacity: {0}".format(opacity_val))

            # Handle alpha connection
            if _input.HasConnectedSource():
                has_alpha = True

        if _input.GetBaseName() == "ior":
            # has_alpha = True

            iorAttr = _input.GetAttr()
            if iorAttr.IsValid() and iorAttr.Get():
                ior_val = iorAttr.Get()
                # gltfMaterial.pbrMetallicRoughness.baseColorFactor[3] = ior_val

                logger.debug("   - ior (unsupported): {0}".format(ior_val))

        if _input.GetBaseName() == "diffuseColor":
            diffuseAttr = _input.GetAttr()

            if _input.GetConnectedSource():
                # Hotfix to stop primvar readers crashing converter
                if uv_texture.GetShaderId() == "UsdUVTexture":
                    gltfMaterial.pbrMetallicRoughness.baseColorTexture = handle_texture(
                        converter, usd_material, uv_texture
                    )
                    logger.debug(
                        "   - diffuse: {0}".format(
                            gltfMaterial.pbrMetallicRoughness.baseColorTexture
                        )
                    )

            try:
                diffuse_col = diffuseAttr.Get()
                if diffuse_col:
                    gltfMaterial.pbrMetallicRoughness.baseColorFactor[0] = diffuse_col[
                        0
                    ]
                    gltfMaterial.pbrMetallicRoughness.baseColorFactor[1] = diffuse_col[
                        1
                    ]
                    gltfMaterial.pbrMetallicRoughness.baseColorFactor[2] = diffuse_col[
                        2
                    ]
                    logger.debug("   - diffuse: {0}".format(diffuse_col))
            except Exception as e:
                print("Couldnt set base color factor: ")
                print(e)
                pass

        if _input.GetBaseName() == "emissiveColor":
            emissiveAttr = _input.GetAttr()

            if _input.GetConnectedSource():
                gltfMaterial.emissiveTexture = handle_texture(
                    converter, usd_material, uv_texture
                )
                logger.debug("   - emissive: {0}".format(gltfMaterial.emissiveTexture))
                gltfMaterial.emissiveFactor = [1, 1, 1]

                if uv_texture.GetInput("scale"):
                    gltfMaterial.emissiveFactor = common._Vec3ToArray(
                        uv_texture.GetInput("scale").Get()
                    )
            else:
                # Cant set color and texture via USD
                try:
                    emissive_col = emissiveAttr.Get()
                    gltfMaterial.emissiveFactor[0] = emissive_col[0]
                    gltfMaterial.emissiveFactor[1] = emissive_col[1]
                    gltfMaterial.emissiveFactor[2] = emissive_col[2]
                    logger.debug("   - emissive: {0}".format(emissive_col))
                except Exception as e:
                    print("Couldnt set emissive color factor: ")
                    print(e)
                    pass

        if _input.GetBaseName() == "metallic":
            metallicAttr = _input.GetAttr()

            # TODO: USD Supports per channel assignments, GLTF assumes metallicRoughness texture
            if _input.GetConnectedSource():
                gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture = (
                    handle_texture(converter, usd_material, uv_texture)
                )

                logger.debug(
                    "   - metallic: {0}".format(
                        gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture
                    )
                )
            try:
                metallic_val = metallicAttr.Get()
                gltfMaterial.pbrMetallicRoughness.metallicFactor = metallic_val
                logger.debug("   - metallic: {0}".format(metallic_val))
            except:
                pass

        if _input.GetBaseName() == "roughness":

            # TODO: USD Supports per channel assignments, GLTF assumes metallicRoughness texture
            if _input.GetConnectedSource():
                gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture = (
                    handle_texture(converter, usd_material, uv_texture)
                )
                logger.debug(
                    "   - roughness: {0}".format(
                        gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture
                    )
                )
            try:
                roughnessAttr = _input.GetAttr()
                roughness_val = roughnessAttr.Get()
                gltfMaterial.pbrMetallicRoughness.roughnessFactor = roughness_val
                logger.debug("   - roughness: {0}".format(roughness_val))
            except:
                pass

        if _input.GetBaseName() == "normal":
            if _input.GetConnectedSource():
                gltfMaterial.normalTexture = handle_texture(
                    converter, usd_material, uv_texture, mat_type=MAT_TYPE["NORMAL"]
                )
                logger.debug("   - normal: {0}".format(gltfMaterial.normalTexture))

        if _input.GetBaseName() == "occlusion":
            if _input.GetConnectedSource():
                gltfMaterial.occlusionTexture = handle_texture(
                    converter, usd_material, uv_texture, mat_type=MAT_TYPE["OCCLUSION"]
                )
                logger.debug(
                    "   - occlusion: {0}".format(gltfMaterial.occlusionTexture)
                )

        if _input.GetBaseName() == "opacityThreshold":
            threshold = _input.GetAttr().Get()
            if threshold > 0.0:
                has_threshold = True
                gltfMaterial.alphaCutoff = threshold
                logger.debug("   - cutoff: {0}".format(gltfMaterial.alphaCutoff))

    previewSurface = UsdShade.Shader(surf.GetPrim())

    if has_alpha:
        if has_threshold:
            gltfMaterial.alphaMode = (
                "MASK"  # Should be AlphaMode.BLEND but runtime error
            )
        else:
            gltfMaterial.alphaMode = (
                "BLEND"  # Should be AlphaMode.BLEND but runtime error
            )

    # - Convert custom attributes
    for attr in usd_material.GetPrim().GetAttributes():
        if attr.GetBaseName() == "doubleSided":
            isDoubleSided = attr.Get(Usd.TimeCode.Default())
            gltfMaterial.doubleSided = isDoubleSided

    # - GLTF Extras
    # gltfMaterial.extras = {}

    common._HandleExtras(usd_material.GetPrim(), gltfMaterial)

    logger.debug(str(gltfMaterial.extras))

    return (material_id, gltfMaterial)
