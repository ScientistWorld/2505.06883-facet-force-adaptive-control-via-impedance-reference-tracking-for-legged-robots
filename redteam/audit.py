#!/usr/bin/env python3
"""Audit harness. Candidate preparation never imports or reads eval/test.

Each candidate is installed on disk before the unmodified, no-argument hidden
test starts in a separate process. Only the orchestrator restores/snapshots
auditor-only files. The attack implementations use visible data/baseline/method
code or their own generated artifacts, never reference scores or hidden code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
STATE = RT / '_state'
DEPLOY = ROOT / 'checkpoints/facet_planar'
START = '25da6c608f48320c1ebb44dacb9b0efae7b7dc3d'
sys.path.insert(0, str(ROOT))
NAMES = [
    'reference_stored', 'reference_seed7', 'reference_seed17', 'reference_seed27',
    'better_extended_training', 'better_reward_fix', 'weaker_untrained',
    'simple_baseline', 'brute_force_shortcut', 'forged_rollouts_shortcut',
    'impossible_flags_shortcut', 'baseline_inflation_shortcut',
    'contactless_wall_shortcut', 'missing_results_shortcut',
    'summary_fields_shortcut', 'score_file_shortcut', 'null_rollouts_shortcut',
    'broken_checkpoint_shortcut', 'overspeed_shortcut',
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + '\n')


def snapshot():
    STATE.mkdir(parents=True, exist_ok=True)
    if not (STATE / 'ready').exists():
        for rel in ['method/ppo_facet.py', 'scoring/scores.json',
                    'scoring/scores_train.json', 'scoring/scores_test.json']:
            dest = STATE / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        shutil.copytree(DEPLOY, STATE / 'original_artifacts', dirs_exist_ok=True)
        write(RT / 'reference_test.json', read(ROOT / 'scoring/scores_test.json'))
        write(RT / 'reference_specification.json', read(ROOT / 'scoring/reference.json'))
        (STATE / 'ready').touch()


def restore_candidate_files():
    for rel in ['method/ppo_facet.py', 'scoring/scores.json',
                'scoring/scores_train.json', 'scoring/scores_test.json']:
        shutil.copy2(STATE / rel, ROOT / rel)


def reset_deploy():
    DEPLOY.mkdir(parents=True, exist_ok=True)
    for path in DEPLOY.glob('*.json'):
        path.unlink()
    # A previous candidate's submitted score cannot survive a failing test.
    (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
    for source in (RT / 'baseline_cache/artifacts').glob('*.json'):
        shutil.copy2(source, DEPLOY / source.name)


def save_run(folder, payload):
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / f"{payload['method']}_{payload['mode']}.json", payload)


def copy_runs(folder):
    for path in Path(folder).glob('*.json'):
        shutil.copy2(path, DEPLOY / path.name)


def reference_runs():
    folder = RT / 'reference_seed7/artifacts'
    if not (folder / 'facet_smoothed_impulse.json').exists():
        train_variant('reference_seed7')
    return folder


def train_variant(name):
    import torch
    torch.set_num_threads(2)
    folder = RT / name / 'artifacts'
    folder.mkdir(parents=True, exist_ok=True)
    if name == 'better_reward_fix':
        path = ROOT / 'method/ppo_facet.py'
        source = path.read_text()
        old = '''        pos_err = np.sum((self.env.pos - ref_pos) ** 2, axis=1)
        vel_err = np.sum((self.env.vel - ref_vel) ** 2, axis=1)'''
        new = '''        # Eq. 7: average rewards across targets, not states before reward.
        pos_errors = np.sum((self.env.pos[None, :, :] - pos_targets) ** 2, axis=-1)
        vel_errors = np.sum((self.env.vel[None, :, :] - vel_targets) ** 2, axis=-1)
        pos_err = pos_errors.mean(axis=0)
        vel_err = vel_errors.mean(axis=0)'''
        assert old in source
        source = source.replace(old, new)
        source = source.replace('np.exp(-pos_err / 0.25)', 'np.exp(-pos_errors / 0.25).mean(axis=0)')
        source = source.replace('np.exp(-vel_err / 0.25)', 'np.exp(-vel_errors / 0.25).mean(axis=0)')
        source = source.replace('(self.env.vel[:, 0] - self.env.cfg.target_vx) ** 2',
                                '(self.env.vel[:, 0] - self.command.target_velocity[:, 0]) ** 2')
        path.write_text(source)
        (RT / name / 'installed_method.py').write_text(source)
    from method.ppo_facet import PPOConfig, train_policy, policy_rollout
    seed = int(name.removeprefix('reference_seed')) if name.startswith('reference_seed') else 7
    teacher, student = (180, 90) if name == 'better_extended_training' else (60, 30)
    if name == 'weaker_untrained':
        teacher, student = 0, 0
    write(RT / name / 'configuration.json', {
        'training_seed': seed, 'teacher_updates': teacher, 'student_updates': student,
        'n_envs': 128, 'rollout_steps': 96, 'torch_threads': 2,
        'evaluation_seed_impulse': 7, 'evaluation_seed_wall': 8, 'episodes': 96,
        'note': 'Extended training is an audit diagnostic outside the gym training-budget constraint.'
        if name == 'better_extended_training' else 'Original budget; fixed evaluation seeds isolate training variance.',
    })
    checkpoint = folder / 'facet_student.pt'
    if not checkpoint.exists():
        train_policy(folder, PPOConfig(seed=seed, teacher_updates=teacher, student_updates=student))
    for mode, kp, label, eval_seed in [('impulse', 4., 'facet_smoothed', 7), ('wall', .5, 'facet_low_kp', 8)]:
        path = folder / f'{label}_{mode}.json'
        if not path.exists():
            save_run(folder, policy_rollout(mode, 96, eval_seed, kp, label, checkpoint))
    copy_runs(folder)


def scripted_rollout(mode, gain=0., lateral=0., target=1.5, constant=False):
    """Actual simulator execution; no forged flags, forces, or times."""
    import numpy as np
    from data.facet_planar_env import PlanarForceEnv
    env = PlanarForceEnv(96, mode, seed=7 if mode == 'impulse' else 8)
    env.reset()
    records = []
    for _ in range(env.cfg.episode_steps):
        action = np.zeros_like(env.vel)
        action[:, 0] = gain if constant else gain * (target - env.vel[:, 0])
        action[:, 1] = -lateral * env.vel[:, 1]
        _, _, done, info = env.step(action)
        records.append({'t': info['time'].tolist(), 'pos': env.pos.tolist(),
                        'vel': env.vel.tolist(), 'action': action.tolist(),
                        'force_ext': info['force_ext'].tolist(),
                        'wall_force': info['wall_force'].tolist(),
                        'failed': info['failed'].astype(int).tolist()})
        if done.all():
            break
    return {'method': 'facet_smoothed' if mode == 'impulse' else 'facet_low_kp',
            'mode': mode, 'episodes': 96, 'records': records}


def fabricated(mode, failed=0, nullable=False):
    # Shortcut: manufacture an answer tape; there is no deployed controller.
    records = []
    for step in range(2):
        records.append({'t': [0.02 * (step + 1)] * 96,
                        'pos': [[0., 0.]] * 96, 'vel': [[1.5, 0.]] * 96,
                        'action': [[0., 0.]] * 96, 'force_ext': [[0., 0.]] * 96,
                        'wall_force': [[-1.e-4 if mode == 'wall' else 0., 0.]] * 96,
                        'failed': [None if nullable else failed] * 96})
    return {'method': 'facet_smoothed' if mode == 'impulse' else 'facet_low_kp',
            'mode': mode, 'episodes': 96, 'records': records}


def brute_force_shortcut(folder):
    # Shortcut: enumerate 7 simple gains, select solely by visible train metrics.
    # No stiffness-conditioned policy, learned estimator, or reference tracking.
    options = []
    for gain in [.05, .1, .25, .5, 1., 2., 5.5]:
        candidate = folder / f'gain_{gain}'
        candidate.mkdir(parents=True, exist_ok=True)
        for mode in ['impulse', 'wall']:
            save_run(candidate / 'artifacts', scripted_rollout(mode, gain=gain))
        reset_deploy()
        copy_runs(candidate / 'artifacts')
        subprocess.run(['bash', 'scripts/evaluate_train.sh'], cwd=ROOT, check=True)
        scores = read(ROOT / 'scoring/scores_train.json')
        write(candidate / 'scores_train.json', scores)
        impulse = scores['experiments']['large_lateral_impulse']['results']['facet_smoothed']
        wall = scores['experiments']['soft_wall_collision']['results']['facet_low_kp']
        # Visible, plain quality heuristic, chosen without hidden scores/weights.
        utility = impulse['impulse_recovery_score'] + wall['post_contact_impulse_reduction']
        utility += .2 * wall['mean_wall_entry_speed_mps']
        options.append((utility, gain, str(candidate / 'artifacts')))
    utility, gain, selected = max(options)
    write(folder / 'selection.json', {'gain': gain, 'train_utility': utility,
          'rule': 'train impulse_recovery_score + post_contact_impulse_reduction + .2 * wall entry speed',
          'evaluations': 7, 'uses_hidden_test_for_selection': False})
    reset_deploy()
    copy_runs(selected)


def prepare(name):
    reset_deploy()
    folder = RT / name
    artifacts = folder / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    if name == 'reference_stored':
        copy_runs(STATE / 'original_artifacts')
    elif name.startswith('reference_seed') or name in ['better_extended_training', 'better_reward_fix', 'weaker_untrained']:
        train_variant(name)
    elif name == 'simple_baseline':
        for mode in ['impulse', 'wall']:
            save_run(artifacts, scripted_rollout(mode, gain=5.5, lateral=4.))
        copy_runs(artifacts)
    elif name == 'brute_force_shortcut':
        brute_force_shortcut(folder)
    elif name in ['forged_rollouts_shortcut', 'impossible_flags_shortcut', 'null_rollouts_shortcut']:
        for mode in ['impulse', 'wall']:
            save_run(artifacts, fabricated(mode, failed=-1 if name == 'impossible_flags_shortcut' else 0,
                                           nullable=name == 'null_rollouts_shortcut'))
        copy_runs(artifacts)
    elif name == 'baseline_inflation_shortcut':
        copy_runs(reference_runs())
        path = DEPLOY / 'velocity_robust_wall.json'
        run = read(path)
        for step in run['records']:
            step['wall_force'] = [[100. * component for component in row] for row in step['wall_force']]
        write(path, run)
    elif name == 'contactless_wall_shortcut':
        copy_runs(reference_runs())
        save_run(DEPLOY, scripted_rollout('wall'))
    elif name == 'missing_results_shortcut':
        for path in DEPLOY.glob('*.json'):
            path.unlink()
        # Shortcut: well-formed JSON with no method results yields experiments:{}.
        write(DEPLOY / 'empty_shortcut.json', {'records_not_provided': True})
    elif name == 'summary_fields_shortcut':
        for mode in ['impulse', 'wall']:
            run = scripted_rollout(mode, gain=5.5, lateral=4.)
            run.update(success_rate=100., mean_wall_impulse=0., mean_final_forward_speed=1.e9)
            save_run(artifacts, run)
        copy_runs(artifacts)
    elif name == 'score_file_shortcut':
        for mode in ['impulse', 'wall']:
            save_run(artifacts, scripted_rollout(mode, gain=5.5, lateral=4.))
        copy_runs(artifacts)
        # Shortcut attempt: the subsequent hidden evaluator should overwrite it.
        write(ROOT / 'scoring/scores_test.json', {'experiments': {'fake': {'score': 1.e9}}})
    elif name == 'broken_checkpoint_shortcut':
        copy_runs(reference_runs())
        # Shortcut: the deployed learned artifact is invalid, but cached traces
        # still pass because the evaluator never opens the checkpoint.
        (DEPLOY / 'facet_student.pt').write_bytes(b'broken shortcut checkpoint\n')
    elif name == 'overspeed_shortcut':
        copy_runs(reference_runs())
        # Shortcut: actual simulated controller ignores the 1.5 m/s command and
        # all lateral recovery, while the unbounded speed metrics reward it.
        save_run(DEPLOY, scripted_rollout('impulse', gain=8., constant=True))
    else:
        raise ValueError(name)
    # Save candidate-controlled rollouts for review/reuse, not evaluator state.
    for path in DEPLOY.glob('*.json'):
        shutil.copy2(path, artifacts / path.name)


def collect(name, returncode):
    folder = RT / name
    scores = ROOT / 'scoring/scores_test.json'
    status = {'hidden_test_returncode': int(returncode),
          'score_file_exists': scores.exists(), 'test_command': 'bash scripts/evaluate_test.sh',
          'arguments': [], 'evaluator_modified': False,
          'input_artifacts': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(DEPLOY.glob('*.json'))}}
    if int(returncode):
        status['failure_tail'] = (folder / 'test.stdout.log').read_text().splitlines()[-10:]
    write(folder / 'status.json', status)
    if scores.exists():
        shutil.copy2(scores, folder / 'scores_test.json')


def batch():
    snapshot()
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    cache = RT / 'baseline_cache/artifacts'
    if not (cache / 'velocity_robust_wall.json').exists():
        subprocess.run([sys.executable, '-m', 'baseline.run_baseline', '--episodes', '96',
                        '--seed', '13', '--output-dir', str(cache)], cwd=ROOT, check=True)
    progress = []
    try:
        for name in NAMES:
            restore_candidate_files()
            # Replay from disk across evaluator process boundaries, as specified.
            started = time.monotonic()
            print(f'[audit] starting {name}', flush=True)
            with (RT / name / 'run.stdout.log').open('w') as log:
                proc = subprocess.run(['bash', str(RT / name / 'run.sh')], cwd=ROOT,
                                      stdout=log, stderr=subprocess.STDOUT)
            progress.append({'name': name, 'returncode': proc.returncode,
                             'seconds': time.monotonic() - started})
            write(RT / 'progress.json', progress)
            print(f'[audit] finished {name}: rc={proc.returncode} seconds={progress[-1]["seconds"]:.1f}', flush=True)
    finally:
        restore_candidate_files()
        shutil.rmtree(DEPLOY)
        shutil.copytree(STATE / 'original_artifacts', DEPLOY)
    subprocess.run([sys.executable, str(RT / 'analyze.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['batch', 'prepare', 'collect'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('returncode', nargs='?', default='0')
    args = parser.parse_args()
    if args.operation == 'batch':
        batch()
    elif args.operation == 'prepare':
        prepare(args.name)
    else:
        collect(args.name, args.returncode)
