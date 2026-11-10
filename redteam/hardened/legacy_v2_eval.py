#!/usr/bin/env python3
"""Score with the previous hardening round's protocol ("v2"), for before/after values.

Usage: python legacy_v2_eval.py <evaluator.py> <checkpoint_root> <output.json> <details.json>

Loads the given evaluator module unchanged and restores the four protocol
constants this round changed: wall virtual mass 12 kg (equal to the nominal
body), 32 pushes per impulse level, 128 wall and 256 tracking episodes. Then
runs its main(). Used only to report the previous protocol's scores.
"""

import importlib.util
import sys

path, root, output, details = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("legacy_v2_eval_module", path)
module = importlib.util.module_from_spec(spec)
sys.modules["legacy_v2_eval_module"] = module
spec.loader.exec_module(module)
module.WALL_VIRTUAL_MASS = 12.0
module.IMPULSE_PER_LEVEL = 32
module.WALL_EPISODES = 128
module.TRACK_EPISODES = 256
sys.argv = [path, root, "--output", output, "--details", details]
sys.exit(module.main())
