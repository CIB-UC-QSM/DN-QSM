# Batch-size-10 depth capacity

The previous training run was stopped cleanly at update 2,100. The capacity search selected six symmetric encoder/decoder levels with widths `[23, 47, 90, 180, 360, 720]` and a 1,440-channel bottleneck for both networks. The measured architecture contained **410,626,887 parameters** and 33 Conv3d layers per network, compared with 18 in the previous three-level model.

The current defaults restore this measured concatenation architecture: **410,626,887 parameters**, 33 Conv3d layers in each network, raw-field concatenation before the phase output head, no Laplacian output addition, and no refinement block. The experimental Laplacian-residual and refinement variants retain their saved configurations for checkpoint loading.

Current new models use LeakyReLU with slope 0.01; the capacity measurements used CELU. This activation change preserves parameter counts. Memory measurements have not been repeated with LeakyReLU.

This is the deepest fitting model in the tested family: preserve `[23, 47, 90]`, double widths for each added level, and keep the existing pooling, padding, and normalization. It is not a maximum over every possible neural-network topology.

| Encoder levels per network | Parameters, both networks | Peak process VRAM | Result |
|---|---:|---:|---|
| 3 | 6,421,407 | 42.91 GiB | fits |
| 4 | 25,673,847 | 42.93 GiB | fits |
| 5 | 102,669,927 | 43.93 GiB | fits |
| 6 | 410,626,887 | 47.70 GiB | fits |
| 7 | 1,642,400,007 | Failed before first update | oom |

All candidates used full, uncropped **160³** batches of **10**; BF16 convolutions; FP32 Laplacian/physics; activation checkpointing; TKD 0.2; and the alternating phase/chi schedule. Six zero-LR AdamW steps initialized both networks’ optimizer moments, then exercised each branch with gradient diagnostics and both sets of moments retained. Full and partial validation batches, the shape-validation suite, COSMOS (184³), and sim2 (210³) were evaluated with the optimizer resident. No learned weights were produced by these probes.

Seven encoder levels force the existing U-Net padding to increase the internal grid from 160³ to 256³. That candidate failed on its first phase batch: an additional 7.19 GiB allocation was needed with only 4.85 GiB free. The 98% probe budget reserves a small margin; the failed allocation also exceeded the remaining physical device memory. The GPU is an NVIDIA RTX PRO 6000 Blackwell Max-Q with 94.96 GiB reported to PyTorch.

The selected six-level candidate peaked at **47.70 GiB of process VRAM** (51.21 decimal GB). The large remaining margin reflects the abrupt padding/memory jump at seven levels; it is not evidence that seven levels fit. Fitting memory does not establish reconstruction quality.

Use the portable [training profile](../configs/deep_alternating_batch10.json) to retain the latest augmentation and 10-pretrain/100-alternating-epoch settings. The architecture and batch size are also the model/training defaults. Older profiles with explicit `features` and batch settings retain those values.

Changing U-Net depth changes tensor shapes. Exact resume and weight initialization from a shallower checkpoint are incompatible; existing checkpoints are preserved and retain their recorded architectures for inference. The profile starts with fresh weights.

Reproduce the isolated search using a full TrainConfig JSON:

```sh
.venv/bin/python -m dnqsm.depth_size --config <training-config.json> \
  --directory artifacts/capacity/depth_batch10 --batch-size 10 \
  --base-features 23 47 90 --target-fraction 0.98
```

Each candidate runs in a separate CUDA process. A failed candidate releases its allocator before the next probe. A user-imposed maximum depth is labeled bounded rather than claimed as a measured maximum.

[Measured results](../artifacts/capacity/depth_batch10_20261020_225102/summary.json) contain per-branch timings, memory telemetry, and the adjacent OOM result.

Validation after applying the defaults: **108 tests passed**, plus Ruff lint, formatting, and whitespace checks. The prior run remains stopped; the capacity probes did not start a replacement training run.
