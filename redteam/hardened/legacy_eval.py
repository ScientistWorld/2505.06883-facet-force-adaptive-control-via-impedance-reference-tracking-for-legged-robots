#!/usr/bin/env python3
"""Score with an evaluator whose wall-retreat gate is disabled (pre-fix behaviour).

Usage: python legacy_eval.py <evaluator.py> <checkpoint_root> <output.json> <details.json>

Loads the given evaluator module unchanged, disables the two wall-collision
checks added in the final hardening round (RETREAT_TOLERANCE = inf and
COMPLIANCE_REFERENCE = False, i.e. the gym's behaviour before them) and runs its
main(). Used only to report before-fix scores of the wall-retreat shortcuts.
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
module.COMPLIANCE_REFERENCE = False
sys.argv = [path, root, "--output", output, "--details", details]
sys.exit(module.main())
