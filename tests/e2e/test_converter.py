import json
import tempfile
import subprocess
import logging
from os import path
from unittest import TestCase
from pathlib import Path
from ..util import setup_temp_dir, SAMPLES_DIR, TEMP_DIR
from gltflib import GLTF
import usd2gltf.converter as converter

logger = logging.getLogger(__name__)


class TestConverter(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        setup_temp_dir()

    def _get_model_index(self):
        with open(path.join(SAMPLES_DIR, "model-index.json")) as f:
            return json.load(f)

    def test_converter(self):
        for info in self._get_model_index():
            asset_name = info["name"]
            inputs = info["inputs"]
            outputs = info["outputs"]

            for input in inputs:
                basename = inputs[input]
                original_filename = path.join(SAMPLES_DIR, asset_name, basename)
                abspath = path.abspath(original_filename)

            logger.debug("Converting: {}".format(abspath))

            # Load the input asset
            factory = converter.Converter()
            factory.interpolation = "LINEAR"
            factory.flatten_xform_animation = True
            stage = factory.load_usd(original_filename)

            # Convert the asset

            output_filename = path.join(TEMP_DIR, asset_name, outputs["gltf"])
            reference_filename = path.join(SAMPLES_DIR, asset_name, outputs["gltf"])

            factory.process(stage, output_filename)

            export_asset = GLTF.load(output_filename)
            reference_asset = GLTF.load(reference_filename)

            self.assertEqual(export_asset.model, reference_asset.model)
