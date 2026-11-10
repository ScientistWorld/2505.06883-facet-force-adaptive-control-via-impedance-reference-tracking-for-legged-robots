#!/usr/bin/env python3
"""Auditor-only aggregation; never used to prepare or select a candidate."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics

RT = Path('/home/user/redteam')


def read(path):
    return json.loads(Path(path).read_text())


spec = read(RT / 'reference_specification.json')['experiments']
reference = read(RT / 'reference_test.json')['experiments']
metric_specs = []
for experiment, setup in spec.items():
    proposed = [name for name, item in setup['results'].items() if item['type'] == 'proposed']
    assert len(proposed) == 1
    for metric, config in setup['metrics'].items():
        metric_specs.append((experiment, proposed[0], metric,
                             setup['weight'] * abs(config['coefficient']),
                             1. if config['higher_is_better'] else -1.))


def flatten(payload):
    found = {}
    for experiment, method, metric, weight, direction in metric_specs:
        value = payload.get('experiments', {}).get(experiment, {}).get('results', {}).get(method, {}).get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            found[f'{experiment}/{metric}'] = float(value)
    return found


base = flatten({'experiments': reference})
attempts = {}
statuses = {}
for path in sorted(RT.glob('*/status.json')):
    name = path.parent.name
    statuses[name] = read(path)
    score_file = path.parent / 'scores_test.json'
    if score_file.exists() and statuses[name]['hidden_test_returncode'] == 0:
        attempts[name] = flatten(read(score_file))

genuine_names = [name for name in attempts if 'shortcut' not in name]


def scales_for(names):
    scales = {}
    for key in base:
        values = [attempts[name][key] for name in names if key in attempts[name]]
        sd = statistics.pstdev(values) if len(values) > 1 else 0.
        scales[key] = sd if sd > 1.e-12 else 1.
    return scales


def rank(values, scales):
    contributions = {}
    for experiment, method, metric, weight, direction in metric_specs:
        key = f'{experiment}/{metric}'
        if key in values:
            contributions[key] = weight * direction * (values[key] - base[key]) / scales[key]
    return sum(contributions.values()), contributions


all_scales = scales_for(list(attempts))
genuine_scales = scales_for(genuine_names)
summary = {'rank_definition': 'sum experiment_weight * abs(coefficient) * direction * (attempt - stored reference) / cohort_population_std; missing metrics skipped',
           'normalization_caveat': 'Production scales depend on future attempts and are unavailable. Both declared audit cohorts are supplied; these are illustrative ranks, not a fixed deployed rank.',
           'cohorts': {'genuine': genuine_names, 'all': list(attempts)},
           'metric_scales': {'genuine': genuine_scales, 'all': all_scales},
           'reference_metrics': base, 'attempts': {}}
for name, status in statuses.items():
    item = dict(status)
    item['metrics'] = attempts.get(name, {})
    if name in attempts:
        item['rank_all'], item['contributions_all'] = rank(attempts[name], all_scales)
        item['rank_genuine_scale'], item['contributions_genuine_scale'] = rank(attempts[name], genuine_scales)
    else:
        item['rank_all'] = None
        item['rank_genuine_scale'] = None
    summary['attempts'][name] = item

seeds = [f'reference_seed{seed}' for seed in [7, 17, 27]]
summary['training_seed_spread'] = {}
if all(name in attempts for name in seeds):
    for key in base:
        values = [attempts[name][key] for name in seeds]
        summary['training_seed_spread'][key] = {'values': values, 'min': min(values), 'max': max(values), 'range': max(values) - min(values)}
    for rank_key in ['rank_all', 'rank_genuine_scale']:
        values = [summary['attempts'][name][rank_key] for name in seeds]
        summary['training_seed_spread'][rank_key] = {'values': values, 'range': max(values) - min(values)}
(RT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
