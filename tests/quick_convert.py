import usd2gltf.converter as converter
import argparse

parser = argparse.ArgumentParser(description="Convert incoming USD(z) file to glTF/glb")

parser.add_argument(
    "-i",
    "--input",
    dest="input",
    action="store",
    help="Input USD (.usd, .usdc, .usda, .usdz)",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    action="store",
    help="Output glTF (.gltf, .glb)"
)

args = parser.parse_args()

factory = converter.Converter()

stage = factory.load_usd(args.input)
factory.process(stage, args.output)
