import os

# TensorFlow 2.16+ defaults to Keras 3, but tensorflow-probability 0.23
# (pinned via tf-agents) still uses Keras 2 internals (tf.keras.__internal__).
# Route tf.keras back to the standalone tf-keras (Keras 2) package before any
# TF/TFP/tf-agents import happens transitively from our submodules.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from .paligemma_wm import *
