# Daily Summary — September 2, 2026

- Built a batch-aware persistent MoE expert cache that works with prompt and MTP-style batches.
- Added prefill overlap, density-gated full-layer streaming, and an 8-token GPU-to-GPU handoff that preserves the decode cache.
- Replaced random-token benchmarking with a persistent `llama-server` corpus test and corrected GPU selection for the real RTX 3090 and 5090.
- Qwen3.5 cache decode reached **2.19×** baseline on the 3090 and **1.73×** on the 5090.
- Qwen3.8 cache decode reached **1.96×/1.71×** baseline with pinned weights and **2.42×/2.06×** with mmap on the 3090/5090.
- Qwen3.8 mmap reduced total request time by **19.6%** on the 3090 and **16.8%** on the 5090, with roughly 15–17 second model startup.
- Main finding: selected-expert prefetch is the right default. Full-layer streaming wastes bandwidth at the observed 57–65% prompt density, while the decode cache remains valuable even with mmap-backed experts.
- Local builds, cache tests, and diff checks pass.
