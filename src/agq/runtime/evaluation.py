"""Closed-loop eight-action execution and an explicit local LIBERO adapter.

This generic evaluator does not reconstruct the withheld EXP019 panel. New
evaluations must be labelled with their actual initial states and environment.
"""
import numpy as np


def rollout(env, observation, predict, prepare, *, max_steps, action_horizon=8):
    """Execute chunks until success or horizon; propagate simulator errors.

predict(prepared_observation) returns already deployed [H,7] actions.
env.step follows LIBERO's (observation, reward, success, info) interface.
The environment is already restored/settled by the caller and remains caller-owned.
"""
    if type(max_steps) is not int or max_steps <= 0 or type(action_horizon) is not int or action_horizon <= 0:
        raise ValueError("positive integer step and action horizons required")
    steps, calls = [], 0
    while len(steps) < max_steps:
        actions = np.asarray(predict(prepare(observation)), dtype=float)
        if actions.shape != (action_horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("invalid deployed action chunk")
        calls += 1
        for action in actions:
            if len(steps) >= max_steps:
                break
            observation, reward, success, _ = env.step(action.tolist())
            if not np.isfinite(reward):
                raise RuntimeError("nonfinite simulator reward")
            steps.append({"reward": float(reward), "success": bool(success)})
            if success:
                return {"success": True, "control_steps": len(steps), "policy_calls": calls, "steps": steps}
    return {"success": False, "control_steps": len(steps), "policy_calls": calls, "steps": steps}


def evaluate_libero_state(policy, task, initial_state, *, suite, seed=7, wait_steps=10, max_steps=None,
                          validate_prepared=None):
    """Evaluate a caller-selected LIBERO state through installed OFT helpers.

An optional validate_prepared callback runs BEFORE any policy action and can
enforce exact image/state comparison against the caller's frozen observation.
No state retry, replacement, or silent skipping is performed.
"""
    limits = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    if suite not in limits or type(wait_steps) is not int or wait_steps < 0:
        raise ValueError("unsupported suite or invalid settling steps")
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import prepare_observation
    env, instruction = get_libero_env(task, policy.cfg.model_family, resolution=policy.cfg.env_img_res)
    try:
        env.seed(seed)
        env.reset()
        observation = env.set_init_state(np.asarray(initial_state))
        for _ in range(wait_steps):
            observation, _, _, _ = env.step(get_libero_dummy_action(policy.cfg.model_family))
        prepare = lambda raw: prepare_observation(raw, 224)[0]
        if validate_prepared is not None:
            validate_prepared(prepare(observation))
        result = rollout(env, observation, lambda obs: policy.predict_prepared(obs, instruction, suite)["deployed"],
                         prepare, max_steps=limits[suite] if max_steps is None else max_steps)
        return {**result, "suite": suite, "seed": seed, "wait_steps": wait_steps,
                "strict_observation_validator": validate_prepared is not None,
                "panel": "caller-supplied state; not the original EXP019 panel"}
    finally:
        env.close()
