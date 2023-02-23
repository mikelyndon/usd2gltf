# usd2gltf

USD to GLTF/GLB converter

Current version `0.3.5`

## Usage

This package can be incorporated into a DCC tool (like Houdini) or used from the command-line.

### CLI

usage: usd2gltf [-h] [-i INPUT] [-o OUTPUT] [--interpolation INTERPOLATION] [-d] [-f]

Convert incoming USD(z) file to glTF/glb

optional arguments:

- -h, --help show this help message and exit
- -i INPUT, --input INPUT
  - Input USD (.usd, .usdc, .usda, .usdz)
- -o OUTPUT, --output OUTPUT
  - Output glTF (.gltf, .glb)
- --interpolation INTERPOLATION
  - Interpolation of animation (LINEAR, STEP, CUBIC)
- -d, --debug Run in debug mode
- -f, --flatten Flatten all animations into one animation

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
