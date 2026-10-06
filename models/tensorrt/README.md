# TensorRT Inference Engines for pDFNet3

This directory contains pre-compiled NVIDIA TensorRT execution plans for ultra-low latency real-time speech isolation.

## Specifications

- **Target Architecture**: NVIDIA Blackwell Architecture (Compute Capability 12.0)
- **Model Topology**: Stateful DeepFilterNet3 recurrent graphs with explicit buffer states (`h_in`/`h_out`, `feat_erb_buf`, `feat_spec_buf`, `c0_buf`) for $S=1$ real-time causal frame processing.
- **Engine Files**:
  - `enc.engine` (2.23 MB): Encoder with bottleneck FiLM conditioning injection.
  - `df_dec.engine` (3.52 MB): Deep Filtering Decoder with recurrent DF-GRU state progression.
  - `erb_dec.engine` (3.57 MB): ERB Spectral Decoder computing gain masks.

## Execution

Engines can be loaded directly using NVIDIA TensorRT C++ / Python APIs or the high-performance Rust execution harness (`crates/runtime-tensorrt`):

```python
import tensorrt as trt

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open("models/tensorrt/enc.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
```
