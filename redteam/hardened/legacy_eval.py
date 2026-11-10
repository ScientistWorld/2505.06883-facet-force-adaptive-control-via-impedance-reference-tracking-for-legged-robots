#!/usr/bin/env python3
"""Score with an evaluator whose wall-retreat gate is disabled (pre-fix behaviour).

Usage: python legacy_eval.py <evaluator.py> <checkpoint_root> <output.json> <details.json>

Loads the given evaluator module unchanged, sets RETREAT_TOLERANCE to infinity
(the behaviour of the gym before the gate was added) and runs its main(). Used
only to report the before-fix score of the retreat shortcut.
"""

import importlib.util
import math
import sys

path, root, output, details = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("legacy_eval_module", path)
module = importlib.util.module_from_spec(spec)
sys.modules["legacy_eval_module"] = module
spec.loader.exec_module(module)
module.RETREAT_TOLERANCE = math.inf
sys.argv = [path, root, "--output", output, "--details", details]
sys.exit(module.main())
