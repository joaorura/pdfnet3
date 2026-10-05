import pytest
from pathlib import Path
import numpy as np
import onnx

from pdfnet3.film import SITES, HIDDEN, verify_film_graph
from pdfnet3.pipeline import PDFNet3Session

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
