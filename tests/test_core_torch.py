import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from agq.core.calibration import LearnedLinear, export_matrix, calibrate_block
from agq.core.codec import encode
from agq.runtime.linear import PackedLinear, replace_linears
from agq.runtime.blockwise import install_block_lifetime
from agq.runtime.profiling import balanced_action_mse, probe_action_sensitivity, benchmark_inference
from agq.runtime.evaluation import rollout


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_calibrated_pack_matches_dense_and_frozen_weights(bits):
    torch.manual_seed(7)
    original = nn.Linear(13, 5)
    learned = LearnedLinear(original, bits)
    x = torch.randn(3, 13)
    learned(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in learned.parameters())
    assert learned.original.grad is None
    exported = export_matrix(learned)
    packed = PackedLinear(exported, original.bias)
    assert torch.equal(packed.decode(), learned.weight())
    assert torch.equal(packed(x), learned(x))
    blob, _ = encode(exported["packed"].numpy().tobytes(), bits)
    packed.compress(blob)
    assert torch.equal(packed(x), learned(x))


def test_block_reconstruction_and_exception_cleanup():
    torch.manual_seed(8)
    teacher = nn.Sequential(nn.Linear(12, 16), nn.Tanh(), nn.Linear(16, 12))
    before = copy.deepcopy(teacher.state_dict())
    inputs = [torch.randn(2, 12) for _ in range(3)]
    student, history = calibrate_block(teacher, {"0": 2, "2": 3}, inputs, epochs=2)
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())
    assert history["epochs"] == 2 and history["trainable_scalars"] == 56
    payloads = {"block." + n: export_matrix(student.get_submodule(n)) for n in ("0", "2")}
    model = nn.Module()
    model.add_module("block", copy.deepcopy(teacher))
    replace_linears(model, payloads)
    life = install_block_lifetime(model, {"block": list(payloads)})
    assert torch.equal(model.block(inputs[0]), student(inputs[0]))
    assert life.report()["max_live_matrices"] == 2
    with pytest.raises(RuntimeError):
        model.block(torch.randn(1, 99))
    assert life.report()["entries"] == life.report()["exits"] == 2
    life.remove()
    assert torch.equal(model.block(inputs[0]), student(inputs[0]))


def test_balanced_action_error_excludes_gripper():
    reference = np.zeros((3, 8, 7))
    changed = reference.copy()
    changed[:2, :, :6] = 1
    changed[2, :, :6] = 3
    changed[:, :, 6] = 1000
    assert balanced_action_mse(reference, changed, ["a", "a", "b"], ["t1", "t1", "t2"]) == 5


def test_sensitivity_probe_restores_weight_when_predict_raises():
    model = nn.Sequential(nn.Linear(3, 56))
    before = model[0].weight.detach().clone()
    calls = []
    def predict(m, sample):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("intentional predictor failure")
        return m(sample).detach().numpy().reshape(8, 7)
    with pytest.raises(RuntimeError, match="intentional"):
        probe_action_sensitivity(model, ["0"], [torch.ones(3)], predict, ["a"], ["b"])
    assert torch.equal(before, model[0].weight)
    assert model.training


def test_rollout_chunk_boundary_success_and_time_limit():
    class Env:
        def __init__(self, success_at):
            self.calls, self.success_at = 0, success_at
        def step(self, action):
            self.calls += 1
            return self.calls, 0, self.calls == self.success_at, {}
    predict = lambda obs: np.zeros((8, 7))
    done = rollout(Env(10), 0, predict, lambda x: x, max_steps=20)
    assert done["success"] and done["control_steps"] == 10 and done["policy_calls"] == 2
    timeout = rollout(Env(100), 0, predict, lambda x: x, max_steps=9)
    assert not timeout["success"] and timeout["control_steps"] == 9 and timeout["policy_calls"] == 2


def test_cpu_benchmark_does_not_invent_cuda_memory():
    calls = []
    result = benchmark_inference(lambda: calls.append(1), warmup=2, repeats=3)
    assert len(calls) == 5 and result["peak_allocated_bytes"] is None
    assert len(result["latency_ms_samples"]) == 3


def test_target_package_roundtrip_and_tamper_rejection(tmp_path):
    from agq.runtime.package import save_payloads, load_payloads
    torch.manual_seed(42)
    payloads = {f"layer{i}": export_matrix(LearnedLinear(nn.Linear(17, 9), i)) for i in (1, 2, 3, 4)}
    folder = tmp_path / "private_targets"
    manifest = save_payloads(folder, payloads)
    restored, encoded = load_payloads(folder)
    assert manifest["non_target_weights_included"] is False
    for name, payload in payloads.items():
        for key in ("packed", "scale", "zero_or_reserved"):
            assert torch.equal(restored[name][key], payload[key])
        module = PackedLinear(restored[name]).compress(encoded[name])
        assert torch.equal(module.decode(), PackedLinear(payload).decode())
    target = folder / manifest["matrices"]["layer1"]["file"]
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        load_payloads(folder)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_compressed_runtime_drops_raw_cache(bits):
    original = nn.Linear(64, 64, bias=False)
    with torch.no_grad():
        original.weight.fill_(0.2)
    learned = LearnedLinear(original, bits)
    payload = export_matrix(learned)
    blob, report = encode(payload["packed"].numpy().tobytes(), bits)
    assert report["mode"] == "group_huffman"
    module = PackedLinear(payload).compress(blob)
    assert module.packed is None and module.encoded is not None
    assert torch.equal(module.decode(), learned.weight())


def test_openvla_action_domains_are_independent_with_inplace_gripper_helpers():
    from types import SimpleNamespace
    from agq.runtime.openvla import OpenVLAAdapter

    normalized = np.full((8, 7), 0.25)

    class Model(nn.Module):
        norm_stats = {"libero_spatial": {}}

        def _unnormalize_actions(self, actions, unnorm_key=None):
            result = actions.copy()
            result[:, :6] += 2
            result[:, 6] = 0.75
            return result

    class Robot:
        @staticmethod
        def get_action(cfg, model, observation, instruction, **kwargs):
            return model._unnormalize_actions(normalized, cfg.unnorm_key)

        @staticmethod
        def normalize_gripper_action(action, binarize=True):
            action[-1] = 2 * action[-1] - 1
            return action

        @staticmethod
        def invert_gripper_action(action):
            action[-1] *= -1
            return action

    model = Model()
    original = model._unnormalize_actions
    policy = OpenVLAAdapter(SimpleNamespace(), model, None, None, None, Robot())
    result = policy.predict_prepared({}, "synthetic instruction")
    assert model._unnormalize_actions == original
    assert np.array_equal(result["normalized"], normalized)
    assert np.all(result["unnormalized"][:, 6] == 0.75)
    assert np.all(result["deployed"][:, 6] == -0.5)
    assert np.all(result["unnormalized"][:, :6] == 2.25)
    assert not np.shares_memory(result["normalized"], result["unnormalized"])
    assert not np.shares_memory(result["unnormalized"], result["deployed"])
    result["deployed"][:] = 123
    assert np.all(result["unnormalized"][:, 6] == 0.75)
    assert np.all(result["normalized"] == 0.25)


def test_openvla_capture_is_restored_after_upstream_prediction_failure():
    from types import SimpleNamespace
    from agq.runtime.openvla import OpenVLAAdapter

    class Model(nn.Module):
        norm_stats = {"libero_spatial": {}}

        def _unnormalize_actions(self, actions, unnorm_key=None):
            return actions.copy()

    class Robot:
        @staticmethod
        def get_action(cfg, model, observation, instruction, **kwargs):
            model._unnormalize_actions(np.zeros((8, 7)), cfg.unnorm_key)
            raise RuntimeError("upstream prediction failed after capture")

    model = Model()
    original = model._unnormalize_actions
    policy = OpenVLAAdapter(SimpleNamespace(), model, None, None, None, Robot())
    with pytest.raises(RuntimeError, match="upstream prediction failed"):
        policy.predict_prepared({}, "synthetic instruction")
    assert model._unnormalize_actions == original


def test_block_lifetime_enforces_inference_with_trainable_norms_and_input():
    model = nn.Module()
    model.add_module("block", nn.Sequential(
        nn.LayerNorm(12), nn.Linear(12, 16), nn.SiLU(), nn.Linear(16, 12), nn.LayerNorm(12)
    ))
    model.eval()
    payloads = {"block." + name: export_matrix(LearnedLinear(model.block.get_submodule(name), 3))
                for name in ("1", "3")}
    modules = replace_linears(model, payloads)
    life = install_block_lifetime(model, {"block": list(payloads)})
    assert model.block[0].weight.requires_grad and model.block[4].weight.requires_grad
    x = torch.randn(2, 12, requires_grad=True)
    saved = []
    def pack_hook(tensor):
        saved.append(tensor)
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        output = model.block(x)
    assert not output.requires_grad and output.grad_fn is None
    assert not saved
    assert all(module._temporary is None for module in modules.values())
    assert life.report()["dense_weights_live_outside_block"] == 0
    assert life.report()["entries"] == life.report()["exits"] == 1
