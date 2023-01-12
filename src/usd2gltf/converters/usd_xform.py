from pxr import UsdGeom

import struct
import operator

from gltflib import *

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

    matrix = common._ConvertMatrix(usd_xform.GetLocalTransformation())

    # TODO: Check this better
    if matrix != [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]:
        gltfNode.matrix = matrix

    converter.gltfDoc.nodes.append(gltfNode)

    return (node_id, gltfNode)
