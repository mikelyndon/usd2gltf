import os
import shutil
from os import path

# Temporary directory used for tests (cleared on every run)
TEMP_DIR = "tests/temp"

SAMPLES_DIR = "tests/samples"


def sample(model, ext="usda"):
    return path.join(SAMPLES_DIR, model, "{}.{}".format(model, ext))


def setup_temp_dir():
    """
    Creates TEMP_DIR (if it does not already exist) and ensures it is empty.
    """
    if path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
