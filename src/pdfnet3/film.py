"""FiLM conditioning logic for DeepFilterNet3 ONNX graphs."""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

SITES = {
    "enc": "/emb_gru/linear_in/1/Relu_output_0",
    "df_dec": "/df_gru/linear_in/linear_in.1/Relu_output_0",
}
HIDDEN = 256
_FILM_NODES = ("film_mul", "film_add")


class FilmGraphError(RuntimeError):
    """Raised when the ONNX graph does not match expected FiLM injection sites."""


def add_film(model: onnx.ModelProto, site: str) -> onnx.ModelProto:
    """Injects gamma/beta multiplicative and additive FiLM nodes at the designated site."""
    g = model.graph
    if any(i.name in ("gamma", "beta") for i in g.input):
        raise FilmGraphError("model already contains gamma/beta inputs")
    producers = [i for i, nd in enumerate(g.node) if site in nd.output]
    if len(producers) != 1:
        raise FilmGraphError(f"site {site}: expected 1 producer, found {len(producers)}")
    for name in ("gamma", "beta"):
        g.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, "S", HIDDEN]))
    scaled, filmed = site + "_gamma", site + "_film"
    for nd in g.node:
        for k, x in enumerate(nd.input):
            if x == site:
                nd.input[k] = filmed
    nodes = list(g.node)
    nodes.insert(
        producers[0] + 1,
        helper.make_node(
            "Mul",
            [site, "gamma"],
            [scaled],
            name=f"{site}/film_mul",
        ),
    )
    nodes.insert(
        producers[0] + 2,
        helper.make_node(
            "Add",
            [scaled, "beta"],
            [filmed],
            name=f"{site}/film_add",
        ),
    )
    g.ClearField("node")
    g.node.extend(nodes)
    return model


def verify_film_graph(model: onnx.ModelProto) -> bool:
    """Verifies that gamma and beta exist in graph inputs and FiLM nodes are present."""
    input_names = {i.name for i in model.graph.input}
    if not ({"gamma", "beta"}.issubset(input_names)):
        return False
    node_names = {n.name for n in model.graph.node}
    has_mul = any(n == "film_mul" or n.endswith("/film_mul") for n in node_names)
    has_add = any(n == "film_add" or n.endswith("/film_add") for n in node_names)
    return has_mul and has_add
