"""Device/dtype normalization for CPU and CUDA actor configs."""
from pathlib import Path

import pytest
import yaml

from jev_reward_model.config import ExperimentConfig, normalize_device


def test_normalize_device_cpu_and_cuda():
    assert normalize_device("cpu") == "cpu"
    assert normalize_device("CPU:0") == "cpu"
    assert normalize_device("cuda") == "cuda:0"
    assert normalize_device("cuda:1") == "cuda:1"
    with pytest.raises(ValueError, match="cpu or cuda"):
        normalize_device("mps")
    with pytest.raises(ValueError, match="CPU device"):
        normalize_device("cpu:1")


def test_cpu_requires_float32_and_does_not_silently_keep_bf16():
    with pytest.raises(ValueError, match="dtype=float32"):
        ExperimentConfig(device="cpu", dtype="bfloat16").validate()
    cfg = ExperimentConfig(device="cpu", dtype="float32")
    cfg.validate()
    assert cfg.device == "cpu" and cfg.dtype == "float32"


def test_cuda_defaults_unchanged():
    cfg = ExperimentConfig()
    cfg.validate()
    assert cfg.device == "cuda:0" and cfg.dtype == "bfloat16"


def test_load_cpu_yaml_defaults_float32(tmp_path):
    path = tmp_path / "cpu.yaml"
    path.write_text(yaml.safe_dump({"reward": "grounded", "device": "cpu",
                                    "train_data": "t", "eval_data": "e",
                                    "output_dir": "runs/x"}))
    cfg = ExperimentConfig.load(path)
    assert cfg.device == "cpu" and cfg.dtype == "float32"


def test_load_explicit_cpu_bf16_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"device": "cpu", "dtype": "bfloat16"}))
    with pytest.raises(ValueError, match="dtype=float32"):
        ExperimentConfig.load(path)


def test_example_cpu_and_cuda_yaml_parity():
    cuda = ExperimentConfig.load("configs/grounded.yaml")
    cpu = ExperimentConfig.load("configs/grounded_cpu.yaml")
    assert cuda.device == "cuda:0" and cuda.dtype == "bfloat16"
    assert cpu.device == "cpu" and cpu.dtype == "float32"
    skip = {"device", "dtype", "output_dir"}
    assert {k: v for k, v in cuda.to_dict().items() if k not in skip} == {
        k: v for k, v in cpu.to_dict().items() if k not in skip}
    assert Path("configs/grounded.yaml").exists()
