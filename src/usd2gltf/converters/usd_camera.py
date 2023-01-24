from pxr import UsdGeom, Sdf, Gf

import math
import logging

from gltflib import (Camera, PerspectiveCameraInfo, OrthographicCameraInfo)

from usd2gltf import common

logger = logging.getLogger(__name__)


def convert(converter, usd_camera):
    camera_id = len(converter.cameras)

    prim = usd_camera.GetPrim()
    ppath = prim.GetPrimPath()
    prim_name = prim.GetName()

    logger.debug(" - camera[{0}]: {1} : {2}".format(camera_id, ppath, prim_name))

    gltfCamera = Camera()
    gltfCamera.name = prim_name

    proj_attr = usd_camera.GetProjectionAttr()
    projection = proj_attr.Get()

    gf_camera = usd_camera.GetCamera()

    clipping_range = usd_camera.GetClippingRangeAttr().Get()
    fov_y = math.radians(gf_camera.GetFieldOfView(Gf.Camera.FOVVertical))

    aspect_ratio = gf_camera.aspectRatio

    ap_x = gf_camera.horizontalAperture / 10
    ap_y = gf_camera.verticalAperture / 10

    if projection == UsdGeom.Tokens.perspective:
        gltfCamera.type = "perspective"
        gltfCamera.perspective = PerspectiveCameraInfo(
            aspectRatio=aspect_ratio,
            yfov=fov_y,
            znear=clipping_range[0],
            zfar=clipping_range[1],
        )
    elif projection == UsdGeom.Tokens.orthographic:
        gltfCamera.type = "orthographic"
        gltfCamera.orthographic = OrthographicCameraInfo(
            xmag=ap_x,
            ymag=ap_y,
            znear=clipping_range[0],
            zfar=clipping_range[1],
        )
    else:
        # Unknown projection type
        return (None, None)

    # - GLTF Extras
    # gltfCamera.extras = {}

    common._HandleExtras(prim, gltfCamera)

    return (camera_id, gltfCamera)
