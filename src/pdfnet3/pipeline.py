"""Inference session for pDFNet3 running on ONNX Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import onnxruntime as ort

from pdfnet3.film import HIDDEN


class PDFNet3Session:
    """Manages the ONNX Runtime sessions for pDFNet3 submodels."""

    def __init__(self, model_dir: Path | str, providers: Optional[list] = None):
        self.model_dir = Path(model_dir)
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.providers = providers

        enc_path = self.model_dir / "enc.onnx"
        df_dec_path = self.model_dir / "df_dec.onnx"
        erb_dec_path = self.model_dir / "erb_dec.onnx"

        if not enc_path.is_file():
            raise FileNotFoundError(f"Missing encoder model: {enc_path}")
        if not df_dec_path.is_file():
            raise FileNotFoundError(f"Missing DF decoder model: {df_dec_path}")
        if not erb_dec_path.is_file():
            raise FileNotFoundError(f"Missing ERB decoder model: {erb_dec_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.enc_sess = ort.InferenceSession(str(enc_path), sess_options=opts, providers=self.providers)
        self.df_dec_sess = ort.InferenceSession(str(df_dec_path), sess_options=opts, providers=self.providers)
        self.erb_dec_sess = ort.InferenceSession(str(erb_dec_path), sess_options=opts, providers=self.providers)

    def neutral_film_parameters(self, seq_len: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Returns identity FiLM parameters (gamma=1.0, beta=0.0)."""
        gamma = np.ones((1, seq_len, HIDDEN), dtype=np.float32)
        beta = np.zeros((1, seq_len, HIDDEN), dtype=np.float32)
        return gamma, beta

    def voice_film_parameters(self, embedding: np.ndarray, seq_len: int = 1) -> Tuple[np.ndarray, np.ndarray]:
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
