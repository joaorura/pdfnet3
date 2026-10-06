from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import pytest

from pdfnet3.film import HIDDEN, verify_film_graph
from pdfnet3.pipeline import (
    SUPPORTED_PROVIDERS,
    PDFNet3Session,
    auto_select_provider,
    get_available_execution_providers,
    get_available_providers,
    get_recommended_provider,
    resolve_execution_providers,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_models_exist():
    assert (MODELS_DIR / "enc.onnx").is_file()
    assert (MODELS_DIR / "df_dec.onnx").is_file()
    assert (MODELS_DIR / "erb_dec.onnx").is_file()
    assert (MODELS_DIR / "config.ini").is_file()


def test_film_graph_verification():
    enc_proto = onnx.load(str(MODELS_DIR / "enc.onnx"))
    assert verify_film_graph(enc_proto), "enc.onnx must contain FiLM inputs and nodes"

    df_proto = onnx.load(str(MODELS_DIR / "df_dec.onnx"))
    assert verify_film_graph(df_proto), "df_dec.onnx must contain FiLM inputs and nodes"


def test_session_init_and_film_params():
    session = PDFNet3Session(MODELS_DIR)
    assert session.enc_sess is not None
    assert session.df_dec_sess is not None
    assert session.erb_dec_sess is not None

    gamma, beta = session.neutral_film_parameters(seq_len=4)
    assert gamma.shape == (1, 4, HIDDEN)
    assert beta.shape == (1, 4, HIDDEN)
    assert np.allclose(gamma, 1.0)
    assert np.allclose(beta, 0.0)

    dummy_emb = np.random.randn(192).astype(np.float32)
    gamma_spk, beta_spk = session.voice_film_parameters(dummy_emb, seq_len=4)
    assert gamma_spk.shape == (1, 4, HIDDEN)
    assert beta_spk.shape == (1, 4, HIDDEN)
    assert not np.allclose(gamma_spk, 1.0)


def test_get_available_execution_providers():
    providers = get_available_execution_providers()
    assert isinstance(providers, list)
    assert "CPUExecutionProvider" in providers
    # Verify alias works identically
    assert get_available_providers() == providers


def test_get_recommended_provider_real():
    recommended = get_recommended_provider()
    assert isinstance(recommended, str)
    assert recommended in SUPPORTED_PROVIDERS
    # Alias check
    assert auto_select_provider() == recommended


def test_provider_resolution_and_aliases():
    # DirectML (AMD Radeon, Intel Arc, NVIDIA on Windows DirectX 12)
    assert resolve_execution_providers("dml") == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("directml") == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("DmlExecutionProvider") == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Apple CoreML
    assert resolve_execution_providers("coreml") == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("CoreMLExecutionProvider") == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Intel OpenVINO
    assert resolve_execution_providers("openvino") == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("OpenVINOExecutionProvider") == [
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]

    # AMD ROCm / Linux
    assert resolve_execution_providers("rocm") == ["ROCMExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("amd") == ["ROCMExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("ROCMExecutionProvider") == [
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    ]

    # AMD Vitis AI / Ryzen AI
    assert resolve_execution_providers("vitis") == ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("vitisai") == ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("VitisAIExecutionProvider") == [
        "VitisAIExecutionProvider",
        "CPUExecutionProvider",
    ]

    # NVIDIA CUDA / TensorRT
    assert resolve_execution_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("CUDAExecutionProvider") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert resolve_execution_providers("tensorrt") == ["TensorrtExecutionProvider", "CPUExecutionProvider"]
    assert resolve_execution_providers("trt") == ["TensorrtExecutionProvider", "CPUExecutionProvider"]

    # CPU
    assert resolve_execution_providers("cpu") == ["CPUExecutionProvider"]
    assert resolve_execution_providers("CPUExecutionProvider") == ["CPUExecutionProvider"]

    # Explicit list
    assert resolve_execution_providers(providers=["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


@pytest.mark.parametrize(
    "mocked_available,expected_recommended",
    [
        (["CPUExecutionProvider"], "CPUExecutionProvider"),
        (["ROCMExecutionProvider", "CPUExecutionProvider"], "ROCMExecutionProvider"),
        (["DmlExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider"),
        (["CoreMLExecutionProvider", "CPUExecutionProvider"], "CoreMLExecutionProvider"),
        (["OpenVINOExecutionProvider", "CPUExecutionProvider"], "OpenVINOExecutionProvider"),
        (["VitisAIExecutionProvider", "CPUExecutionProvider"], "VitisAIExecutionProvider"),
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"),
        (["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"], "TensorrtExecutionProvider"),
    ],
)
def test_simulated_accelerator_recommendations(mocked_available, expected_recommended):
    with patch("pdfnet3.pipeline.get_available_execution_providers", return_value=mocked_available):
        assert get_recommended_provider() == expected_recommended


def test_session_init_auto_provider():
    session = PDFNet3Session(MODELS_DIR, provider="auto")
    assert session.active_provider is not None
    assert "CPUExecutionProvider" in session.active_providers


def test_session_explicit_providers_list():
    session = PDFNet3Session(MODELS_DIR, providers=["CPUExecutionProvider"])
    assert session.providers == ["CPUExecutionProvider"]
    assert session.active_provider == "CPUExecutionProvider"


def test_session_graceful_fallback_unsupported_provider():
    # Attempting to load an unavailable provider (e.g. DmlExecutionProvider on Linux)
    # With fallback_to_cpu=True, it should emit a warning and fall back to CPUExecutionProvider
    with pytest.warns(RuntimeWarning, match="Requested execution provider 'DmlExecutionProvider' is not registered"):
        session = PDFNet3Session(MODELS_DIR, provider="DmlExecutionProvider", fallback_to_cpu=True)
    assert session.active_provider == "CPUExecutionProvider"
    assert session.providers == ["CPUExecutionProvider"]


def test_session_strict_error_when_fallback_disabled():
    # With fallback_to_cpu=False, attempting an unavailable provider must raise ValueError
    with pytest.raises(ValueError, match="is not available in the current ONNX Runtime installation"):
        PDFNet3Session(MODELS_DIR, provider="DmlExecutionProvider", fallback_to_cpu=False)


def test_session_run_step_forward():
    session = PDFNet3Session(MODELS_DIR, provider="auto")
    S = 1

    feat_erb = np.random.randn(1, 1, S, 32).astype(np.float32)
    feat_spec = np.random.randn(1, 2, S, 96).astype(np.float32)

    # 1. Forward step with neutral FiLM
    out_neutral = session.run_step(feat_erb, feat_spec)
    assert "coefs" in out_neutral
    assert "mask" in out_neutral
    assert "lsnr" in out_neutral
    assert "emb" in out_neutral

    assert out_neutral["coefs"].shape == (1, S, 96, 10)
    assert out_neutral["mask"].shape == (1, 1, S, 32)
    assert out_neutral["lsnr"].shape == (1, S, 1)
    assert out_neutral["emb"].shape == (1, S, 512)

    # 2. Forward step with speaker embedding conditioning
    dummy_emb = np.random.randn(192).astype(np.float32)
    gamma_spk, beta_spk = session.voice_film_parameters(dummy_emb, seq_len=S)
    out_spk = session.run_step(feat_erb, feat_spec, gamma=gamma_spk, beta=beta_spk)

    assert out_spk["coefs"].shape == (1, S, 96, 10)
    assert out_spk["mask"].shape == (1, 1, S, 32)
    assert not np.allclose(out_neutral["coefs"], out_spk["coefs"])


def test_onnx_graphs_operator_compatibility():
    """Validates that all ONNX operators in all 3 models are standard ops supported across EPs."""
    for model_file in ("enc.onnx", "df_dec.onnx", "erb_dec.onnx"):
        proto = onnx.load(str(MODELS_DIR / model_file))
        op_types = {node.op_type for node in proto.graph.node}
        # Standard ONNX operators supported by DML, CoreML, OpenVINO, ROCm, TensorRT, CUDA, CPU:
        common_ops = {
            "Add", "Sub", "Mul", "Div", "MatMul", "Gemm", "Conv", "ConvTranspose", "Relu",
            "Sigmoid", "Tanh", "PRelu", "LeakyRelu", "Reshape", "Transpose", "Concat", "Split",
            "Slice", "Gather", "Shape", "Unsqueeze", "Squeeze", "Flatten", "GRU", "LSTM",
            "ReduceMean", "ReduceSum", "Softmax", "Constant", "ConstantOfShape", "Identity",
            "Cast", "Einsum", "Pad",
        }
        for op in op_types:
            assert op in common_ops, f"Unexpected non-standard op '{op}' in {model_file}"
