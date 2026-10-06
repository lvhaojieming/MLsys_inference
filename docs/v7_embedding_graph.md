# V7 contextual Router and NPU graph execution

`AscendRouterRuntime` loads the V7 `frozen-contextual-expert-softmax-v2` checkpoint,
its frozen Qwen3-Embedding encoder, and its CPU MLP head. The checkpoint threshold
is selected on validation data. Embedding weights remain frozen and complete prompts
are preserved.

Enable the optional graph path in the `router` configuration:

```json
{
  "embedding_model": "/path/to/Qwen3-Embedding-0.6B",
  "embedding_graph": true,
  "graph_buckets": [64, 128, 256, 512, 1024],
  "graph_threshold_margin": 0.01
}
```

Graphs are captured during initialization before service readiness. Each bucket
retains its static token and position buffers, output, and graph. Runtime copies
the new token prefix and replays the nearest bucket. The Qwen3 SDPA causal path
prevents the padded suffix from changing the valid prefix mathematically; pooling
uses the final real encoder token, including the same encoder tokenizer behavior
used during training. Shape changes can nevertheless change BF16 rounding.

Single plain-text prompts are serialized with the canonical training tokenizer.
No content is truncated. Prompts beyond the largest graph bucket use the original
encoder. Graph probabilities within the configured margin of the checkpoint
threshold are recalculated through the original encoder. This protects observed
decision-boundary cases but is not a universal bound on unseen numerical errors.
The graph option defaults to false. Graphs use the same BF16 weights and do not
change the trained architecture or retrain the Router.

On the 2026-10-06 V7 experiment, all 742 validation and 1,677 test expert choices
matched saved predictions. The 0.01 margin caused 215 original-path recalculations;
one sample exceeded the largest bucket. Median full Router time was 22.93 ms and
P95 was 80.97 ms. Maximum output probability difference was 0.009664. The original
24-prompt serving measurement had Router median 67.35 ms and P95 75.87 ms; these
different sample cohorts are not a paired latency comparison.

The completed 24-prompt serving benchmark measured Router median 23.95 ms,
mean 35.75 ms and P95 105.11 ms with graphs. Automatic Gateway TTFT median
was 170.09 ms, mean 186.89 ms and P95 292.24 ms. Without graphs, automatic
TTFT median was 213.32 ms, mean 217.69 ms and P95 238.21 ms. Graph execution
improved typical latency but worsened the tail in this small sample. The typical
24 ms / 170 ms ratio is about 14%; the aggregate measured ratio is about 19%.

## Deployment

Copy `configs/v7_graph_ttft_cluster.json` for the current two-expert cluster,
or adapt its paths and endpoints for a new deployment. Set `router.checkpoint`,
`router.tokenizer`, `router.training_code`, `router.embedding_model`, and
`router.device` to paths/devices visible inside the Gateway process/container.
`embedding_model` overrides the checkpoint's original encoder directory;
use the same frozen weights and tokenizer that produced the training features.
When omitted, the checkpoint's encoder path is used. The training source is
still required for the matching contextual head and feature functions.
Configure expert endpoints or managed node/replica launch commands separately.

Set `embedding_graph` to false to use the original encoder. `graph_buckets`
must be strictly increasing integer lengths (at least 2), within the encoder
context limit. Larger/more buckets increase startup work and resident memory.
`graph_threshold_margin` is an absolute probability distance from the checkpoint
threshold; 0.01 means one percentage point. It does not change that threshold.
This graph path supports a single full-causal Qwen3 prompt, serialized by the
Router lock; it is not dynamic batching. The tested environment used torch 2.10
with torch_npu, Transformers 5.13 and CANN 9.0. Revalidate decisions when changing
the runtime, checkpoint or buckets. Unsupported capture fails initialization
before backend admission; no silent graph disable occurs.

Start with `moqe-serve --config /path/to/cluster.json`. Restart Gateway after
changing Router settings; config watching only applies node/replica changes.
Graph capture and Router warmup complete before backend admission. Admission
and READY-only expert scheduling retain the existing state machine.

`scripts/validate_embedding_graph.py` records every held-out choice and probability
difference. `scripts/benchmark_v7_ttft.py` compares direct expert requests, fixed
Gateway routing, and automatic routing, preserving the same prompt, budget,
template, and selected expert. First nonempty content is the TTFT boundary.
Requests run sequentially after warmup; mode order is randomized. Complete streams
are drained. Output differences between independent requests are counted without
discarding latency samples. These measurements exclude client WAN latency and
are not a load test.
