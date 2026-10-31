# Example setup — local embedding on CPU

tags: #setup #embedding #cpu

How this vault embeds notes without occupying the GPU.

## Decision

Run `Qwen/Qwen3-Embedding-0.6B` on the **CPU in float32**, lazily, one command at
a time. On the original machine the ROCm path produced NaN vectors with bf16, and
loading the model into VRAM on every invocation cost more than it saved.

## Commands

```bash
export AGENT_MEMORY_DEVICE=cpu
./memory.py reindex
./memory.py status
```

## Trade-offs

- Reindex is slower than GPU, but never blocks other GPU work.
- The model is released when the process exits, so idle VRAM is zero.

See [[decisions/example-decision]] for the reasoning.
