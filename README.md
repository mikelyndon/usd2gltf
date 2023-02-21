# usd2gltf

USD to GLTF/GLB converter

Current version `0.3.2`

## Requirements

- usd-core (Or custom Pixar USD build)
- gltflib
- numpy

## Features

- Mesh conversion
  - UV support (TEXCOORD_0, TEXCOORD_1)
  - displayColor support (COLORS_0)
  - Animated skeleton, weights, skinning
  - Normals and tangents supported
- Materials
  - UsdPreviewSurface -> PBRMetallicRoughness
- Point Instancer
  - Creates many objects to one mesh, not true gltf instancing
  - Animated TRS supported
- Camera conversion
- Light conversion
  - Point
  - Spot
  - Directional
- Xform conversion
  - Animated xforms supported
- Animations
  - Allows per object animation or single flat GLTF animation
- Export
  - GLB and GLTF options
- GLTF Extras
