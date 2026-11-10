# Installation and runtime boundary

The reference target is Linux x86_64, Python 3.10, CUDA PyTorch 2.6.0 (cu124), and four 80GB NVIDIA GPUs. CPU RAM is also needed for offloaded parameters and optimizer states; available GPU memory alone is not a host-memory budget. Check actual CPU/RAM allocation before a long run.

Use the installation sequence in the [README](../README.md#quick-start). It creates a new environment rather than modifying the source project's environment.

The SGLang adapter comes from a preserved 0.4.x project snapshot. `sglang[srt,openai]==0.4.6.post5` declares `torch==2.6.0`, `torchvision==0.21.0`, `transformers==4.51.1`, `sgl-kernel==0.1.4`, and `flashinfer_python==0.2.5` in its [versioned upstream metadata](https://github.com/sgl-project/sglang/blob/v0.4.6.post5/python/pyproject.toml). The release requirements follow that combination, removing the old project's contradictory Transformers constraints. Remaining direct pins are a proposed portable runtime, not a recovered complete historical pip freeze. Transitive dependencies are not fully locked; save `pip freeze` after a successful target-machine installation.

FlashAttention is installed separately. `scripts/install_flash_attention.py` queries the official v2.7.4.post1 release assets, matches Python/PyTorch/C++ ABI tags, and installs the matching wheel with `--no-deps`. A missing wheel causes a clear stop. It does not silently compile CUDA sources. To inspect only the expected wheel tag on a CUDA host:

```bash
python scripts/install_flash_attention.py --dry-run
```

If the wheel cannot be fetched, use the [official release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1) through an allowed network route and install the matching wheel locally. Do not replace a binary with one compiled for a different PyTorch or C++ ABI. Source compilation is an explicit fallback and can take substantial time.

Run `python -m pip check`, then `python scripts/check_environment.py`. Use `--gpu` only on four idle assigned GPUs; this performs tiny CUDA arithmetic and FlashAttention calls, without loading model weights. `--inputs` also checks the prepared data and base-model files. The check does not call the external API. Use `--report outputs/environment-check.json` to retain the report.

Driver compatibility depends on the host driver and CUDA compatibility libraries. A successful package installation or `nvidia-smi` output is not a CUDA-kernel test. The local release preparation host has no four-GPU CUDA runtime, so a full target-host test is still required.

Training subprocesses receive per-run JIT cache paths in local temporary storage. They are not automatically deleted; no recursive cleanup is applied to model, data or checkpoint directories.
