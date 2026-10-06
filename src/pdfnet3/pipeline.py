"""Inference session for pDFNet3 running on ONNX Runtime with multi-accelerator execution providers."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pdfnet3.film import HIDDEN

# Canonical hierarchy of supported execution providers
SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "VitisAIExecutionProvider",
    "CPUExecutionProvider",
)

# Friendly aliases mapped to canonical ONNX Runtime Execution Provider names
PROVIDER_ALIASES: dict[str, str] = {
    # DirectML (AMD Radeon, Intel Arc, NVIDIA on Windows DirectX 12)
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dmlexecutionprovider": "DmlExecutionProvider",
    # Apple Silicon (Neural Engine / Metal GPU on macOS)
    "coreml": "CoreMLExecutionProvider",
    "coremlexecutionprovider": "CoreMLExecutionProvider",
    # Intel OpenVINO (Intel NPU, Intel Arc GPU, Intel CPU)
    "openvino": "OpenVINOExecutionProvider",
    "openvinoexecutionprovider": "OpenVINOExecutionProvider",
    # AMD ROCm (Linux)
    "rocm": "ROCMExecutionProvider",
    "rocmexecutionprovider": "ROCMExecutionProvider",
    "amd": "ROCMExecutionProvider",
    # AMD Vitis AI / Ryzen AI NPU
    "vitis": "VitisAIExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "vitisaiexecutionprovider": "VitisAIExecutionProvider",
    # NVIDIA CUDA
    "cuda": "CUDAExecutionProvider",
    "cudaexecutionprovider": "CUDAExecutionProvider",
    "nvidia": "CUDAExecutionProvider",
    # NVIDIA TensorRT
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "tensorrtexecutionprovider": "TensorrtExecutionProvider",
    # CPU fallback
    "cpu": "CPUExecutionProvider",
    "cpuexecutionprovider": "CPUExecutionProvider",
}


def get_available_execution_providers() -> list[str]:
    """Returns the list of execution providers registered in the current ONNX Runtime installation."""
    return list(ort.get_available_providers())


get_available_providers = get_available_execution_providers


def get_recommended_provider() -> str:
    """Returns the highest priority execution provider available on the current host.

    Priority order:
    1. TensorrtExecutionProvider (NVIDIA TensorRT)
    2. CUDAExecutionProvider (NVIDIA CUDA)
    3. ROCMExecutionProvider (AMD ROCm Linux)
    4. DmlExecutionProvider (Microsoft DirectML: AMD Radeon, Intel Arc, NVIDIA on Windows)
    5. CoreMLExecutionProvider (Apple Silicon Neural Engine / Metal on macOS)
    6. OpenVINOExecutionProvider (Intel NPU / Arc GPU / CPU)
    7. VitisAIExecutionProvider (AMD Ryzen AI NPU / XDNA)
    8. CPUExecutionProvider (Universal CPU fallback)
    """
    available = set(get_available_execution_providers())
    for provider in SUPPORTED_PROVIDERS:
        if provider in available:
            return provider
    return "CPUExecutionProvider"


auto_select_provider = get_recommended_provider


def resolve_execution_providers(
    provider: str | list[str] | None = "auto",
    providers: list[str] | None = None,
) -> list[str]:
    """Resolves provider aliases and builds an execution provider priority list.

    Args:
        provider: Provider alias, exact name, or 'auto'.
        providers: Explicit list of providers (takes precedence if provided).

    Returns:
        List of canonical ONNX Runtime execution provider names ending with CPU fallback.
    """
    if providers is not None:
        raw_list = providers if isinstance(providers, list) else [providers]
        resolved: list[str] = []
        for p in raw_list:
            canonical = PROVIDER_ALIASES.get(str(p).strip().lower(), str(p))
            if canonical not in resolved:
                resolved.append(canonical)
        if "CPUExecutionProvider" not in resolved:
            resolved.append("CPUExecutionProvider")
        return resolved

    if provider is None:
        return ["CPUExecutionProvider"]

    if isinstance(provider, list):
        resolved = []
        for p in provider:
            canonical = PROVIDER_ALIASES.get(str(p).strip().lower(), str(p))
            if canonical not in resolved:
                resolved.append(canonical)
        if "CPUExecutionProvider" not in resolved:
            resolved.append("CPUExecutionProvider")
        return resolved

    p_str = str(provider).strip().lower()
    if p_str == "auto":
        rec = get_recommended_provider()
        if rec == "CPUExecutionProvider":
            return ["CPUExecutionProvider"]
        return [rec, "CPUExecutionProvider"]

    canonical = PROVIDER_ALIASES.get(p_str, str(provider))
    if canonical == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    return [canonical, "CPUExecutionProvider"]


class PDFNet3Session:
    """Manages the ONNX Runtime sessions for pDFNet3 submodels across multiple hardware accelerators.

    Supports:
    - DirectML (DmlExecutionProvider) for AMD Radeon, Intel Arc, and NVIDIA on Windows DirectX 12
    - Apple CoreML (CoreMLExecutionProvider) for Apple Silicon Neural Engine / Metal on macOS
    - Intel OpenVINO (OpenVINOExecutionProvider) for Intel NPU, GPU, and CPU
    - AMD ROCm (ROCMExecutionProvider) and Vitis AI (VitisAIExecutionProvider) on Linux / Ryzen AI
    - NVIDIA CUDA (CUDAExecutionProvider) and TensorRT (TensorrtExecutionProvider)
    - CPU (CPUExecutionProvider) universal fallback
    """

    def __init__(
        self,
        model_dir: Path | str,
        provider: str | list[str] | None = "auto",
        providers: list[str] | None = None,
        fallback_to_cpu: bool = True,
        session_options: ort.SessionOptions | None = None,
        provider_options: list[dict] | dict | None = None,
    ):
        self.model_dir = Path(model_dir)
        enc_path = self.model_dir / "enc.onnx"
        df_dec_path = self.model_dir / "df_dec.onnx"
        erb_dec_path = self.model_dir / "erb_dec.onnx"

        if not enc_path.is_file():
            raise FileNotFoundError(f"Missing encoder model: {enc_path}")
        if not df_dec_path.is_file():
            raise FileNotFoundError(f"Missing DF decoder model: {df_dec_path}")
        if not erb_dec_path.is_file():
            raise FileNotFoundError(f"Missing ERB decoder model: {erb_dec_path}")

        opts = session_options or ort.SessionOptions()
        if session_options is None:
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
        self.session_options = opts

        requested_providers = resolve_execution_providers(provider=provider, providers=providers)
        available_providers = get_available_execution_providers()

        primary_ep = requested_providers[0]
        if primary_ep not in available_providers and primary_ep != "CPUExecutionProvider":
            if fallback_to_cpu:
                warnings.warn(
                    f"Requested execution provider '{primary_ep}' is not registered in the current "
                    f"ONNX Runtime build (available: {available_providers}). "
                    "Gracefully falling back to CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                target_providers = ["CPUExecutionProvider"]
                target_prov_opts = None
            else:
                raise ValueError(
                    f"Execution provider '{primary_ep}' is not available in the current ONNX Runtime installation: "
                    f"{available_providers}"
                )
        else:
            target_providers = requested_providers
            target_prov_opts = provider_options

        formatted_options = None
        if target_prov_opts is not None:
            if isinstance(target_prov_opts, dict):
                formatted_options = [target_prov_opts] + [{}] * (len(target_providers) - 1)
            elif isinstance(target_prov_opts, list):
                formatted_options = target_prov_opts

        def _init_sessions(provs: list[str], p_opts: list[dict] | None):
            kwargs = {"sess_options": opts, "providers": provs}
            if p_opts is not None:
                kwargs["provider_options"] = p_opts
            enc = ort.InferenceSession(str(enc_path), **kwargs)
            df = ort.InferenceSession(str(df_dec_path), **kwargs)
            erb = ort.InferenceSession(str(erb_dec_path), **kwargs)
            return enc, df, erb

        try:
            self.enc_sess, self.df_dec_sess, self.erb_dec_sess = _init_sessions(
                target_providers, formatted_options
            )
            self.providers = target_providers
        except Exception as exc:
            if fallback_to_cpu and target_providers != ["CPUExecutionProvider"]:
                warnings.warn(
                    f"Execution provider initialization failed for {target_providers}: {exc}. "
                    "Gracefully falling back to CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.enc_sess, self.df_dec_sess, self.erb_dec_sess = _init_sessions(
                    ["CPUExecutionProvider"], None
                )
                self.providers = ["CPUExecutionProvider"]
            else:
                raise

        active = self.enc_sess.get_providers()
        self.active_provider: str = active[0] if active else "CPUExecutionProvider"

    @property
    def active_providers(self) -> list[str]:
        """Returns the list of active execution providers reported by ONNX Runtime."""
        return self.enc_sess.get_providers()

    def neutral_film_parameters(self, seq_len: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Returns identity FiLM parameters (gamma=1.0, beta=0.0)."""
        gamma = np.ones((1, seq_len, HIDDEN), dtype=np.float32)
        beta = np.zeros((1, seq_len, HIDDEN), dtype=np.float32)
        return gamma, beta

    def voice_film_parameters(self, embedding: np.ndarray, seq_len: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Computes affine FiLM parameters from a 192-dimensional speaker embedding."""
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, 0)
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        if norm > 0:
            embedding = embedding / norm

        # Project 192-dim embedding to 256-dim gamma and beta
        # Deterministic projection for conditioning
        gamma_weights = np.sin(np.outer(np.arange(192), np.arange(HIDDEN)) * 0.05).astype(np.float32) * 0.05
        beta_weights = np.cos(np.outer(np.arange(192), np.arange(HIDDEN)) * 0.05).astype(np.float32) * 0.05

        gamma_vec = 1.0 + np.matmul(embedding, gamma_weights)
        beta_vec = np.matmul(embedding, beta_weights)

        gamma = np.repeat(np.expand_dims(gamma_vec, 1), seq_len, axis=1).astype(np.float32)
        beta = np.repeat(np.expand_dims(beta_vec, 1), seq_len, axis=1).astype(np.float32)
        return gamma, beta

    def run_step(
        self,
        feat_erb: np.ndarray,
        feat_spec: np.ndarray,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Executes a single forward step across the encoder, ERB decoder, and DF decoder.

        Args:
            feat_erb: ERB feature tensor [1, 1, S, 32]
            feat_spec: Complex spectrogram feature tensor [1, 2, S, 96]
            gamma: Optional FiLM scale tensor [1, S, 256]. Defaults to neutral 1.0.
            beta: Optional FiLM shift tensor [1, S, 256]. Defaults to neutral 0.0.

        Returns:
            Dictionary containing 'coefs', 'mask', 'lsnr', and 'emb'.
        """
        seq_len = feat_erb.shape[2]
        if gamma is None or beta is None:
            n_gamma, n_beta = self.neutral_film_parameters(seq_len=seq_len)
            if gamma is None:
                gamma = n_gamma
            if beta is None:
                beta = n_beta

        enc_inputs = {
            "feat_erb": feat_erb.astype(np.float32),
            "feat_spec": feat_spec.astype(np.float32),
            "gamma": gamma.astype(np.float32),
            "beta": beta.astype(np.float32),
        }
        enc_outs = self.enc_sess.run(None, enc_inputs)
        out_names = [o.name for o in self.enc_sess.get_outputs()]
        enc_map = dict(zip(out_names, enc_outs))

        erb_inputs = {
            "emb": enc_map["emb"],
            "e3": enc_map["e3"],
            "e2": enc_map["e2"],
            "e1": enc_map["e1"],
            "e0": enc_map["e0"],
        }
        erb_outs = self.erb_dec_sess.run(None, erb_inputs)

        df_inputs = {
            "emb": enc_map["emb"],
            "c0": enc_map["c0"],
            "gamma": gamma.astype(np.float32),
            "beta": beta.astype(np.float32),
        }
        df_outs = self.df_dec_sess.run(None, df_inputs)

        return {
            "coefs": df_outs[0],
            "mask": erb_outs[0],
            "lsnr": enc_map["lsnr"],
            "emb": enc_map["emb"],
        }
