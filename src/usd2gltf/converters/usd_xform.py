from pxr import UsdGeom
from gltflib import (
        GLTF,
        Node,
        Accessor,
        AccessorType,
        ComponentType,
        Channel,
        Target,
        AnimationSampler
        )

import struct
import operator

from usd2gltf import common
import logging

logger = logging.getLogger(__name__)


def convert(converter, usd_xform):
    node_id = len(converter.gltfDoc.nodes)

    prim = usd_xform.GetPrim()
    ppath = prim.GetPrimPath()
    prim_name = prim.GetName()

    logger.debug(" - xform[{0}]: {1} : {2}".format(node_id, ppath, prim_name))

    gltfNode = Node()
    gltfNode.name = prim_name

    # - glTF Extras
    common._HandleExtras(prim, gltfNode)
    logger.debug(str(gltfNode.extras))

    matrix = common._ConvertMatrix(usd_xform.GetLocalTransformation())

    # TODO: Check this better
    if matrix != [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]:
        gltfNode.matrix = matrix

    converter.gltfDoc.nodes.append(gltfNode)

    # Handle xform animation
    if converter.convert_xform_animation:

        xformable = UsdGeom.Xformable(usd_xform)
        time_samples = xformable.GetTimeSamples()

        if len(time_samples) > 0:

            start_byte_len = len(converter.animated_xforms_bytearray)

            # Times
            # Note: each xform may have its own amount of time samples
            time_byte_start = len(converter.animated_xforms_bytearray)

            min_time = time_samples[0] / converter.FPS
            max_time = time_samples[0] / converter.FPS

            for sample in time_samples:
                gltf_time = sample / converter.FPS

                min_time = min(min_time, gltf_time)
                max_time = max(max_time, gltf_time)

                converter.animated_xforms_bytearray.extend(struct.pack("f", gltf_time))

            translations = []
            rotations = []
            scales = []
            for sample in time_samples:

                xformVectors = UsdGeom.XformCommonAPI(usd_xform).GetXformVectors(sample)
                loc = xformVectors[0]
                rot = common._MakeQuatfFromEulerAngles(
                    xformVectors[1][0], xformVectors[1][1], xformVectors[1][2]
                )
                sca = xformVectors[2]

                translations.append(loc)
                rotations.append(common._QuatToArray(rot))
                scales.append(sca)

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

            # Set initial translation
            gltfNode.translation = common._Vec3ToArray(translations[0])

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

            # Set initial rotation
            gltfNode.rotation = rotations[0]

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

            # Set initial scale
            gltfNode.scale = common._Vec3ToArray(scales[0])

            # Store
            bytelen = len(converter.animated_xforms_bytearray) - start_byte_len
            bufferview_id, bufferview = converter.animated_buffer_view()
            bufferview.byteLength += bytelen

            # Times accessor
            times_accessor_id = len(converter.gltfDoc.accessors)
            converter.gltfDoc.accessors.append(
                Accessor(
                    name="{0}_accessor_times".format(prim_name),
                    bufferView=bufferview_id,
                    byteOffset=time_byte_start,
                    componentType=ComponentType.FLOAT.value,
                    count=len(time_samples),
                    type=AccessorType.SCALAR.value,
                    min=[min_time],
                    max=[max_time],
                )
            )

            # Translation accessor
            pos_accessor_id = len(converter.gltfDoc.accessors)
            converter.gltfDoc.accessors.append(
                Accessor(
                    name="{0}_accessor_positions".format(prim_name),
                    bufferView=bufferview_id,
                    byteOffset=bytes_translation_start,
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
                    name="{0}_accessor_rotations".format(prim_name),
                    bufferView=bufferview_id,
                    byteOffset=bytes_rotation_start,
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
                    name="{0}_accessor_scales".format(prim_name),
                    bufferView=bufferview_id,
                    byteOffset=bytes_scale_start,
                    componentType=ComponentType.FLOAT.value,
                    count=len(scales),
                    type=AccessorType.VEC3.value,
                    min=min_scale,
                    max=max_scale,
                )
            )

            # Start storing data
            _animation_name = common._GetAnimationNameFromUSD(prim)
            if not _animation_name:
                _animation_name = "{0}_xform_anim".format(prim_name)

            _anim = converter.get_named_animation(_animation_name)

            sampler_offset = len(_anim.samplers)

            # Store data

            anim_channels = [
                Channel(
                    sampler=(sampler_offset + 0),
                    target=Target(node=node_id, path="translation"),
                ),
                Channel(
                    sampler=(sampler_offset + 1),
                    target=Target(node=node_id, path="rotation"),
                ),
                Channel(
                    sampler=(sampler_offset + 2),
                    target=Target(node=node_id, path="scale"),
                ),
            ]

            anim_samplers = [
                AnimationSampler(
                    input=times_accessor_id,
                    interpolation=converter.interpolation,
                    output=pos_accessor_id,
                ),
                AnimationSampler(
                    input=times_accessor_id,
                    interpolation=converter.interpolation,
                    output=rot_accessor_id,
                ),
                AnimationSampler(
                    input=times_accessor_id,
                    interpolation=converter.interpolation,
                    output=sca_accessor_id,
                ),
            ]

            _anim.channels += anim_channels
            _anim.samplers += anim_samplers

            # Animation of a node cannot be applied to a node with matrix...
            # TODO: This might cause inconsistencies
            gltfNode.matrix = None

    return (node_id, gltfNode)
