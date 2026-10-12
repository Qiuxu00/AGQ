"""Storage-constrained matrix allocation, adapted from EXP008/EXP010/EXP018.

The objective is sum_i rho_i (4-b_i)^2. Candidate byte costs are explicit,
because historical probe formats and exported row-wise storage differ.
"""
import math
from functools import reduce


def rowwise_bytes(rows, columns, bits):
    """Packed codes plus BF16 scale and uint8 zero/reserved value for each row."""
    if any(type(v) is not int or v <= 0 for v in (rows, columns)) or bits not in (1, 2, 3, 4):
        raise ValueError("positive integer dimensions and W1--W4 are required")
    return rows * ((columns * bits + 7) // 8) + 3 * rows


def joint_scores(sensitivity, overhead, lambda_value=0.5):
    """Normalize within ONE backbone. A zero-sum component contributes zero.

The measured overhead is max(T_W4 - T_BF16, 0), supplied by the caller.
The zero-sum convention makes all-zero test probes well defined; the paper's
formal inputs have positive sums for both components.
"""
    s, c = list(map(float, sensitivity)), list(map(float, overhead))
    if not s or len(s) != len(c) or not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("matching nonempty measurements and nonnegative lambda required")
    if any(not math.isfinite(v) or v < 0 for v in s + c):
        raise ValueError("measurements must be finite and nonnegative")
    ss, cs = math.fsum(s), math.fsum(c)
    return [(si / ss if ss else 0) + lambda_value * (ci / cs if cs else 0)
            for si, ci in zip(s, c)]


def allocate(records, budget_bytes, lambda_value=0.5, *, time_limit=300, mip_rel_gap=1e-6):
    """Solve one backbone's allocation using SciPy/HiGHS binary MILP.

Each record has name, rows, columns, sensitivity, overhead; optional
candidate_bytes is [W1_bytes, W2_bytes, W3_bytes, W4_bytes]. Use separate calls
for vision/language to preserve the separate budgets and normalization.
Only an independently checked feasible solution is returned. Time-limited
solutions carry optimal=False and their solver gap, never a claim of proof.
"""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    if type(budget_bytes) is not int or budget_bytes < 0 or not records:
        raise ValueError("a nonnegative integer byte budget and records are required")
    if not math.isfinite(time_limit) or time_limit <= 0 or not 0 <= mip_rel_gap < 1:
        raise ValueError("invalid solver limits")
    names = [r["name"] for r in records]
    if any(not isinstance(n, str) or not n for n in names) or len(names) != len(set(names)):
        raise ValueError("matrix names must be unique nonempty strings")
    storage = []
    for r in records:
        default = [rowwise_bytes(r["rows"], r["columns"], b) for b in (1, 2, 3, 4)]
        cost = list(r.get("candidate_bytes", default))
        if len(cost) != 4 or any(type(v) is not int or v <= 0 for v in cost):
            raise ValueError("four positive integer candidate byte counts required")
        storage.extend(cost)
    if sum(min(storage[i:i + 4]) for i in range(0, len(storage), 4)) > budget_bytes:
        raise ValueError("budget is below the minimum candidate storage")
    scores = joint_scores([r["sensitivity"] for r in records], [r["overhead"] for r in records], lambda_value)
    penalties = [score * (4 - b) ** 2 for score in scores for b in (1, 2, 3, 4)]
    divisor = reduce(math.gcd, storage + [budget_bytes])
    n, count = len(storage), len(records)
    columns = np.arange(n, dtype=np.int32)
    matrix = coo_array((np.concatenate((np.ones(n), np.asarray(storage) // divisor)),
                        (np.concatenate((columns // 4, np.full(n, count, dtype=np.int32))),
                         np.concatenate((columns, columns)))), shape=(count + 1, n)).tocsc()
    result = milp(np.asarray(penalties), integrality=np.ones(n, dtype=np.int32),
                  bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(matrix, np.r_[np.ones(count), -np.inf],
                                               np.r_[np.ones(count), budget_bytes // divisor]),
                  options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap), "presolve": True})
    if result.x is None:
        raise RuntimeError(f"No feasible solution: {result.message}")
    rounded = np.rint(result.x).astype(int).reshape(count, 4)
    if np.max(np.abs(result.x - rounded.ravel())) > 1e-6 or not np.all(rounded.sum(1) == 1) or np.any((rounded < 0) | (rounded > 1)):
        raise RuntimeError("solver returned a nonintegral selection")
    bits = (rounded.argmax(1) + 1).tolist()
    selected = [storage[4 * i + b - 1] for i, b in enumerate(bits)]
    total = sum(selected)
    objective = math.fsum(s * (4 - b) ** 2 for s, b in zip(scores, bits))
    if total > budget_bytes or not math.isclose(objective, result.fun, rel_tol=1e-8, abs_tol=1e-10):
        raise RuntimeError("independent budget/objective validation failed")
    parameters = sum(r["rows"] * r["columns"] for r in records)
    gap = getattr(result, "mip_gap", None)
    return {"records": [{"name": r["name"], "bits": b, "score": s, "selected_bytes": cost,
                         "export_rowwise_bytes": rowwise_bytes(r["rows"], r["columns"], b)}
                        for r, b, s, cost in zip(records, bits, scores, selected)],
            "budget_bytes": budget_bytes, "selected_bytes": total, "objective": objective,
            "weighted_bits": sum(r["rows"] * r["columns"] * b for r, b in zip(records, bits)) / parameters,
            "lambda_value": lambda_value, "optimal": bool(result.status == 0 and gap is not None and gap <= mip_rel_gap),
            "mip_gap": None if gap is None else float(gap), "solver_status": int(result.status),
            "solver_message": str(result.message), "independent_check_passed": True}
