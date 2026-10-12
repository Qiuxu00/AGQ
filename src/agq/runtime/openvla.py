"""Explicit integration with a user-installed OpenVLA-OFT checkout.

No upstream implementation or model weights are vendored. Follow the upstream
installation and put its checkout on PYTHONPATH before using from_upstream.
The combined LIBERO checkpoint must already be present locally.
"""
import copy
from pathlib import Path
import re
from types import MethodType

import numpy as np
import torch

from .linear import replace_linears
from .blockwise import install_block_lifetime


def target_groups(model, *, require_paper_shape=True):
    """Discover the 32x7 language and 51x4 vision targets of the paper model."""
    language, vision = {}, {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lm = re.fullmatch(r"(language_model\.model\.layers\.\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(up|gate|down)_proj)", name)
        vm = re.fullmatch(r"(vision_backbone\..*\.blocks\.\d+)\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))", name)
        if lm:
            language.setdefault(lm.group(1), []).append(name)
        elif vm:
            vision.setdefault(vm.group(1), []).append(name)
    if require_paper_shape and (len(language) != 32 or len(vision) != 51 or any(len(g) != 7 for g in language.values()) or any(len(g) != 4 for g in vision.values())):
        raise ValueError("model does not match the paper's 224 language + 204 vision matrices")
    return {"language": language, "vision": vision}


class OpenVLAAdapter:
    def __init__(self, cfg, model, processor, action_head, proprio_projector, robot_utils):
        self.cfg, self.model, self.processor = copy.copy(cfg), model, processor
        self.action_head, self.proprio_projector, self.robot = action_head, proprio_projector, robot_utils
        self.lifetimes = []

    @classmethod
    def from_upstream(cls, cfg):
        """Load through the installed OFT API using an existing local checkpoint.

cfg is upstream GenerateConfig (or an object with the same attributes).
This adapter is for the two-image, proprioceptive L1 policy evaluated by AGQ.
"""
        if not Path(cfg.pretrained_checkpoint).is_dir():
            raise ValueError("pretrained_checkpoint must be an existing local directory")
        if not cfg.use_l1_regression or cfg.use_diffusion or cfg.use_film or not cfg.use_proprio or cfg.num_images_in_input != 2:
            raise ValueError("AGQ adapter expects two images, proprioception and the L1 action head")
        if cfg.load_in_4bit or cfg.load_in_8bit:
            raise ValueError("load BF16 before installing AGQ target matrices")
        from experiments.robot import robot_utils
        from experiments.robot.openvla_utils import get_processor, get_action_head, get_proprio_projector
        robot_utils.set_seed_everywhere(cfg.seed)
        model = robot_utils.get_model(cfg)
        return cls(cfg, model, get_processor(cfg), get_action_head(cfg, model.llm_dim),
                   get_proprio_projector(cfg, model.llm_dim, proprio_dim=8), robot_utils)

    def install(self, payloads, *, backend="triton", encoded=None, require_paper_shape=True):
        groups = target_groups(self.model, require_paper_shape=require_paper_shape)
        expected = {n for modality in groups.values() for names in modality.values() for n in names}
        if set(payloads) != expected:
            raise ValueError("payloads must cover every discovered target exactly")
        replace_linears(self.model, payloads, backend=backend, encoded=encoded)
        self.lifetimes = [install_block_lifetime(self.model, modality) for modality in groups.values() if modality]
        return {kind: {"blocks": len(modality), "matrices": sum(map(len, modality.values()))} for kind, modality in groups.items()}

    def predict_prepared(self, observation, instruction, suite="libero_spatial"):
        """Input: upstream-prepared full_image, wrist_image and eight-value state.

Return normalized, unnormalized and simulator-deployed action chunks separately.
This temporary capture is intentionally single-threaded, like block lifetimes.
"""
        key = suite if suite in self.model.norm_stats else suite + "_no_noops"
        if key not in self.model.norm_stats:
            raise ValueError(f"normalization statistics missing: {suite}")
        self.cfg.task_suite_name, self.cfg.unnorm_key = suite, key
        original, captured = self.model._unnormalize_actions, []
        def capture(model, normalized_actions, unnorm_key=None):
            captured.append(np.asarray(normalized_actions).copy())
            return original(normalized_actions, unnorm_key)
        self.model._unnormalize_actions = MethodType(capture, self.model)
        try:
            with torch.inference_mode():
                raw = np.asarray(self.robot.get_action(self.cfg, self.model, copy.deepcopy(observation), instruction,
                                  processor=self.processor, action_head=self.action_head,
                                  proprio_projector=self.proprio_projector, noisy_action_projector=None, use_film=False)).copy()
        finally:
            self.model._unnormalize_actions = original
        if len(captured) != 1:
            raise RuntimeError("expected one normalized eight-action chunk")
        deployed = np.asarray([self.robot.invert_gripper_action(self.robot.normalize_gripper_action(row.copy(), binarize=True)) for row in raw])
        result = {"normalized": captured[0], "unnormalized": raw, "deployed": deployed}
        if any(v.shape != (8, 7) or not np.isfinite(v).all() for v in result.values()):
            raise RuntimeError("invalid policy action chunk")
        for life in self.lifetimes:
            life.report()
        return result
