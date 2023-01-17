from pxr import Usd, Gf
import math
import logging

logger = logging.getLogger(__name__)


def _Magnitude(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _Normalize(vec):
    m = _Magnitude(vec)

    if m <= 0.0:
        print(vec)
        logger.debug(vec)
        logger.warning("Attempted divide by zero. Normals issue!")
        return [1, 0, 0]

    return [vec[0] / m, vec[1] / m, vec[2] / m]


def _Clamp(val, minv, maxv):
    return max(min(val, maxv), minv)


def _ConvertUVs(uv):
    return [uv[0], 1 - uv[1]]


def _ConvertColor(color):
    return [_Clamp(color[0], 0, 1), _Clamp(color[1], 0, 1), _Clamp(color[2], 0, 1)]


def _ConvertMatrix(gfMat):
    lmatData = []

    for a in gfMat:
        for b in a:
            lmatData.append(b)

    return lmatData


def _MakeQuatfFromEulerAngles(rx, ry, rz):
    qd = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
    ).GetQuat()
    i = qd.imaginary
    q = Gf.Quatf(qd.real, i[0], i[1], i[2])
    q.Normalize()
    return q


def _QuatToArray(quat):
    i = quat.GetImaginary()
    return (i[0], i[1], i[2], quat.GetReal())


def _Vec2ToArray(vec):
    return (vec[0], vec[1])


def _Vec3ToArray(vec):
    return (vec[0], vec[1], vec[2])


def _HandleExtras(usd_prim, gltfObject):
    properties = usd_prim.GetProperties()
    extras = {}
    for p in properties:
        if "vmi" in p.GetNamespace() or "gltf" in p.GetNamespace():
            if p.GetTypeName().isArray:
                _o = []
                _v = p.Get()
                for i in _v:
                    _o.append(i)
                extras[p.GetBaseName()] = _o
            else:
                extras[p.GetBaseName()] = p.Get()
    if len(extras) > 0:
        gltfObject.extras = extras


# GLB cannot support changing topology so if authored in USD,
# we get first time sample
def _GetStaticValue(attribute):
    ts = attribute.GetTimeSamples()
    if len(ts) > 0:
        return attribute.Get(ts[0])
    return attribute.Get(Usd.TimeCode.Default())


def _GetAnimationNameFromUSD(prim):
    try:
        attr = prim.GetAttribute("gltf:animation:name")
        return attr.Get()
    except Exception as e:
        print(e)
        return None
