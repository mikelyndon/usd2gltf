from pxr import UsdLux

import math
import logging

from gltflib import *

from usd2gltf import common

logger = logging.getLogger(__name__)


def lerp(v0, v1, t):
    return v0 + t * (v1 - v0)


def convert(converter, usd_light):

    light_id = len(converter.lights)
    prim = usd_light.GetPrim()
    ppath = prim.GetPrimPath()

    logger.debug(" - light[{0}]: {1} : {2}".format(light_id, ppath, prim.GetName()))

    gltfLight = {"type": "point"}

    # Type
    if usd_light.GetPrim().IsA(UsdLux.DistantLight):
        gltfLight["type"] = "directional"
    if usd_light.GetPrim().IsA(UsdLux.SphereLight):
        gltfLight["type"] = "point"

    shaping_api = UsdLux.ShapingAPI(usd_light.GetPrim())
    shaping_cone_angle = shaping_api.GetShapingConeAngleAttr().Get()
    shaping_cone_softness = shaping_api.GetShapingConeSoftnessAttr().Get()

    if shaping_cone_angle is not None:
        outer = math.radians(shaping_cone_angle)
        inner = math.radians(lerp(shaping_cone_angle, 0, shaping_cone_softness))
        if outer == inner:
            inner -= 0.000001
        gltfLight["type"] = "spot"
        gltfLight["spot"] = {
            "outerConeAngle": outer,
            "innerConeAngle": inner,
        }

    # Intensity

    intensity_attr = usd_light.GetIntensityAttr()
    intensity = intensity_attr.Get()
    gltfLight["intensity"] = intensity

    # Color

    color_attr = usd_light.GetColorAttr()
    color = common._Vec3ToArray(color_attr.Get())
    gltfLight["color"] = color

    # Name
    gltfLight["name"] = str(usd_light.GetPrim().GetName()) + "_light"

    # - GLTF Extras
    gltfLight["extras"] = {}

    common._HandleExtras(usd_light.GetPrim(), gltfLight)

    return gltfLight
