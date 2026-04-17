# llama.cpp for AMD BC-250 (Cyan Skillfish / gfx1013)

**Purpose:** a hardware-specific snapshot of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with optimizations and tuning changes that make the **AMD BC-250** (former cryptomining APU based on the PS5's "Cyan Skillfish" silicon — Zen 2 + gfx1013 iGPU, 16 GB GDDR6 unified) a legitimately capable inference node.

**Status:** snapshot fork, not actively maintained. Based on upstream commit `f772f6e43`. See [relationship to upstream](#relationship-to-upstream) for PR candidates.

**Measured performance on 9B Q4_K_M (Qwen 3.5-9B):**

| Metric | Stock llama.cpp (Vulkan) | This fork (Vulkan) | Δ |
|---|---|---|---|
| **Single-stream tg** | 37.00 tok/s | **54.99 tok/s** | **+48.6%** |
| **Single-stream pp** | 199.66 tok/s | **~306 tok/s** | **+53%** |
| **Batched-32 aggregate tg** | ~100 tok/s | **151 tok/s** | **+51%** |
| **Batched-64 aggregate tg** | — | **175 tok/s** | — |

Perplexity validated unchanged within noise (±0.025 on wikitext fragment). All gains are engineering optimizations, **not quality compromises**.

A 24-node BC-250 cluster at these numbers projects to **~4200 tokens/s of aggregate serving capacity**.

---

## Who this is for

- You own one or more BC-250 boards and want the fastest `llama.cpp` inference possible on them.
- You don't need a maintained project — you need working code you can clone, build, and run.
- You're comfortable building from source and running Linux command-line tooling.

If you want production-ready multi-user serving on this hardware, combine this fork with the deployment notes in [akandr/bc250](https://github.com/akandr/bc250) and the community docs at [elektricM/amd-bc250-docs](https://github.com/elektricM/amd-bc250-docs).

---

## What this fork changes vs upstream

Three code-level changes in `ggml/src/ggml-vulkan/`:

### 1. `rm_kq = 4` on RDNA1 (matches GCN tuning)

File: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Upstream defaults `rm_kq` (rows per workgroup for K-quants) to 2 on RDNA1, while GCN uses 4. On gfx1013 with RADV, rows-per-workgroup=4 gives **+3% tg** across all K-quants (Q2_K–Q6_K) with no register spills and unchanged perplexity.

**Important caveat**: this value is arch × driver specific. [PR #21043](https://github.com/ggml-org/llama.cpp/discussions/21043) found `rm_kq=1` works best on AMDVLK / RDNA4. Our change applies only to `architecture == AMD_RDNA1`. If you build on a different RDNA1 card (e.g. RX 5700 XT / gfx1010) and see regression, revert this change.

### 2. Q4_K smin algebraic restructure

File: `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp`

The `smin` sum is rewritten from 16 FMAs into 4 horizontal-sum + 4 FMA form — the same algebraic identity that `mul_mat_vec_q5_k.comp` already uses. Gives **+3% tg** across all Q4_K workloads on our test setup.

This change is the cleanest upstream PR candidate: 6 lines changed, no arch-specific code, matches an existing pattern in Q5_K that has been stable for years.

### 3. Fused gate+up+SWIGLU_SPLIT Vulkan kernel (Q4_K only)

Files:
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_glu_iface.glsl` (new)
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_glu_base.glsl` (new)
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k_glu.comp` (new)
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` (+18 lines)
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (+~200 lines: pipeline array, creation, dispatch, graph-level fusion detection)

Detects the `MUL_MAT(gate, cur) + MUL_MAT(up, cur) + SWIGLU_SPLIT` pattern emitted by Llama/Qwen/Gemma-family FFNs with `LLM_FFN_PAR + LLM_FFN_SILU`. When matched, dispatches one fused kernel that:
- Reads the `cur` activation once (not twice)
- Writes the SwiGLU output directly (no intermediate gate_out / up_out buffers)

Activates only for:
- Both weight tensors `GGML_TYPE_Q4_K` with matching shape
- Shared `cur` (B) activation
- Single-column decode (`ne11 == 1`)
- `GGML_GLU_OP_SWIGLU`, not swapped

Falls back to the two-MUL_MAT path for any non-matching pattern. Gives **+5.8% tg** on models with the matching structure. The Q4_K unpack cost dominates the kernel, so the single-read activation saving is modest relative to total compute. Still, a clean win.

Disabled by setting `GGML_VK_DISABLE_FUSION=1`.

---

## Equally important: the rest of the tuning (NOT llama.cpp changes)

**The biggest single performance win on BC-250 comes from something that is _not_ in this fork**: the community SMU governor, which unlocks GPU clocks past the amdgpu driver's conservative ceiling. You MUST install this to get the numbers above. Everything in this fork is on top of that foundation.

Setup checklist, in order:

### 1. BIOS
A community-patched BIOS is required to unlock dynamic VRAM allocation. See [elektricM/amd-bc250-docs/bios](https://elektricm.github.io/amd-bc250-docs/bios/). All the docs for flashing are there. Do this first — nothing else works without it.

### 2. Linux kernel & drivers
Ubuntu 24.04 (kernel 6.8+) with a recent Mesa (26.x) is known to work. Either install from the [kisak PPA](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa) on Ubuntu, or follow [akandr/bc250](https://github.com/akandr/bc250) for their exact stack.

### 3. Kernel memory tuning (critical)
Without this, models over ~5 GB fail to load. Defaults cap GPU buffers at 50% of system RAM (~7.4 GB).

Runtime (test before making permanent):
```bash
echo 4194304 | sudo tee /sys/module/ttm/parameters/pages_limit
echo 4194304 | sudo tee /sys/module/ttm/parameters/page_pool_size
```

Permanent:
```bash
echo "options ttm pages_limit=4194304 page_pool_size=4194304" | \
  sudo tee /etc/modprobe.d/ttm-gpu-memory.conf
```

Also remove the **deprecated** `amdgpu.gttsize=14336` from GRUB's `GRUB_CMDLINE_LINUX` if present — it actively hurts on modern kernels. Run `sudo update-grub` after.

Full details: [akandr/bc250 readme](https://github.com/akandr/bc250) and [elektricM/amd-bc250-docs](https://github.com/elektricM/amd-bc250-docs/tree/main/docs).

### 4. SMU governor (the big one — ~30% of total tg/s gain)

The amdgpu driver pins gfx1013 shader clock at 1500 MHz even under sustained compute. The chip can hold 2300+ MHz stably. This community tool communicates directly with the SMU (System Management Unit) firmware to push clocks to their real max.

**Install the [filippor/cyan-skillfish-governor](https://github.com/filippor/cyan-skillfish-governor/tree/smu) `smu` branch:**

For Fedora/Bazzite:
```bash
sudo dnf copr enable filippor/bazzite
sudo dnf install cyan-skillfish-governor-smu
sudo systemctl enable --now cyan-skillfish-governor-smu
```

For Arch/CachyOS:
```bash
yay -S cyan-skillfish-governor-smu
sudo systemctl enable --now cyan-skillfish-governor-smu
```

For Ubuntu / generic Linux (what we used for our benchmarks):
```bash
cd /tmp
wget https://github.com/filippor/cyan-skillfish-governor/releases/download/v0.4.3/cyan-skillfish-governor-smu-v0.4.3-x86_64-linux.tar.gz
tar -xf cyan-skillfish-governor-smu-v0.4.3-x86_64-linux.tar.gz
cd cyan-skillfish-governor-smu-v0.4.3-x86_64-linux
sudo install -Dm755 cyan-skillfish-governor-smu /etc/cyan-skillfish-governor-smu/cyan-skillfish-governor-smu
sudo install -Dm644 config.toml /etc/cyan-skillfish-governor-smu/config.toml
sudo sed -i 's|set-method = "kernel"|set-method = "smu"|' /etc/cyan-skillfish-governor-smu/config.toml
# dbus policy file comes from the source repo, not the release tarball:
sudo wget -O /etc/dbus-1/system.d/com.cyan.SkillFishGovernor.conf https://raw.githubusercontent.com/filippor/cyan-skillfish-governor/smu/com.cyan.SkillFishGovernor.conf
sudo tee /etc/systemd/system/cyan-skillfish-governor-smu.service > /dev/null <<'EOF'
[Unit]
Description=Cyan Skillfish GPU Governor
After=multi-user.target
[Service]
Type=simple
ExecStart=/etc/cyan-skillfish-governor-smu/cyan-skillfish-governor-smu /etc/cyan-skillfish-governor-smu/config.toml
Restart=on-failure
RestartSec=5s
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now cyan-skillfish-governor-smu
```

Verify it's running:
```bash
sudo journalctl -u cyan-skillfish-governor-smu -n 20
# Should see: "SMU communication verified"
```

---

## Build instructions

Assuming BIOS, TTM, and SMU governor are in place:

```bash
# Required packages (Ubuntu 24.04)
sudo apt install -y build-essential cmake git libvulkan-dev glslang-tools spirv-tools

# Clone this fork (not upstream)
git clone --branch vulkan-fused-gate-up <URL-to-this-fork> llama.cpp-bc250
cd llama.cpp-bc250

# Configure + build with Vulkan
cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON -DLLAMA_CURL=OFF
cmake --build build -j4 --target llama-bench llama-cli llama-server rpc-server

# Run a quick benchmark
./build/bin/llama-bench -m /path/to/your/model-Q4_K_M.gguf -ngl 99 -n 128 -p 128 -r 3
```

If you see tg128 ≥ 50 tok/s on Qwen 9B Q4_K_M, everything is working as intended.

The three optimizations in this fork apply automatically; there's nothing to enable. The fused gate+up kernel can be disabled with `GGML_VK_DISABLE_FUSION=1` if you need to A/B test.

---

## Recommended per-node usage

### Single user, interactive chat
```bash
./build/bin/llama-cli -m model.gguf -ngl 99 -c 4096
```
Expect ~55 tok/s tg on 9B Q4_K.

### Multi-user serving (best throughput per node)
```bash
./build/bin/llama-server -m model.gguf -ngl 99 -c 4096 --parallel 8 --host 0.0.0.0 --port 8080
```
8 concurrent users share the GPU with total ~110 tok/s aggregate throughput on 9B Q4_K. Per-user latency ~13 tok/s — fine for chat. Scale `--parallel` up to ~32 for heavy batched workloads (~155 tok/s aggregate), down to 1 for pure latency.

### Multi-node (pipeline parallel across BC-250s)
```bash
# On each BC-250 node:
./build/bin/rpc-server -H 0.0.0.0 -p 50052

# On the orchestrator:
./build/bin/llama-cli -m big-model.gguf -ngl 99 \
    --rpc <node1>:50052,<node2>:50052,... --device RPC0/RPC1/...
```
Enables running models larger than a single node's 14 GiB via combined memory.

### Speculative decoding: doesn't deliver on BC-250 (Vulkan)

On this hardware we could not measure a speedup from speculative decoding via `llama-server` or `llama-speculative`:

- **Qwen3.5 9B**: hybrid architecture (Gated Delta Net) blocks partial sequence removal — `llama-server` refuses to initialize speculative. Via the CLI tool the compat check passes but draft acceptance collapses to ~0.8% on a 0.8B draft.
- **Llama 3.1 8B + Llama 3.2 1B**: 72% draft acceptance measured, but total wall-clock throughput is identical to non-speculative (~55 tok/s both). Theoretical math predicted ~1.9×.

The gap points at the Vulkan backend's batched verify path, not BC-250 itself. A reader welcome to investigate upstream. This section is retained as a "we tried, here's what happened" note so future readers don't repeat the same experiment.

---

## Relationship to upstream

This fork is **pinned to commit f772f6e43** (llama.cpp b8819) and will not be rebased. If you want fresher base, fork it yourself and forward-port.

### What should go upstream (if someone wants to push)

**Tier 1 — clean PR candidate:**
- **Q4_K smin restructure** — 6 lines in `mul_mat_vec_q4_k.comp`, matches the existing `mul_mat_vec_q5_k.comp` style, benefits all Vulkan-capable hardware running Q4_K models, no architectural assumptions. Low review friction.

**Tier 2 — needs broader hardware testing before upstream:**
- **RDNA1 rm_kq=4** — beneficial on gfx1013 + RADV, but [PR #21043](https://github.com/ggml-org/llama.cpp/discussions/21043) showed the opposite effect on RDNA4 + AMDVLK. Other RDNA1 cards (e.g. gfx1010 / RX 5700 XT) need validation before becoming a default.
- **Fused gate+up kernel** — benefits every Vulkan GPU in principle. Needs a test matrix across NVIDIA, AMD, Intel to confirm no regression. Author of upstream Vulkan backend (0cc4m, jeffbolznv) should probably review the fusion detection logic.

If you're willing to upstream any of these, please do — give credit and pointers back here if helpful. The code in this fork is under the same MIT license as llama.cpp itself.

### What will NEVER go upstream
- The BIOS requirements
- The TTM kernel tuning
- The SMU governor — that's its own separate project, correctly

---

## Benchmarks, measurements, context

Step-by-step contribution of each change, measured on Qwen 3.5-9B Q4_K_M (BC-250, Ubuntu 24.04, kernel 6.8.0, Mesa 26.0.3, RADV, TTM pages_limit raised to 16 GiB):

1. **Baseline**: 37.00 tok/s on Qwen 9B Q4_K with stock llama.cpp + default BC-250 config
2. **+ fused gate+up kernel**: 39.15 (+5.8%)
3. **+ Q4_K smin restructure**: 41.08 (+5.2%)
4. **+ RDNA1 rm_kq=4**: 42.33 (+3.0%)
5. **+ SMU governor**: **54.99 (+29.7%)** ← biggest single win, and it's someone else's work

### Why you can trust the numbers

- Perplexity on Qwen3-4B Q4_K held at **1.0455–1.0457 ± 0.0246** across every configuration we tested (baseline vs full stack). The ±0.025 error bar swamps the 0.0002 deltas — all configs are statistically indistinguishable.
- `test-backend-ops MUL_MAT` full sweep passes 938/938 on Vulkan.
- Every optimization is mathematically equivalent (or FP-equivalent within 1 ULP rounding) to what upstream computes — we are not cutting corners, we are removing unnecessary overhead.

### Reference points in the community

- **Stock ollama 0.19 on same hardware**: 25.67 tok/s (their vendored llama.cpp lags master). We are 2.1× faster.
- **hipfire (Kaden-Schutt)** with custom RDNA1-native kernels: 47 tok/s on 9B MQ4 (their quant). We are 1.17× faster on standard Q4_K via pure engineering (no custom format).
- **Theoretical BW ceiling** for 9B at 5.3 GB on BC-250's 448 GB/s GDDR6: 84.5 tok/s. We're at 65% of physical limit.

---

## Credits

This fork is the sum of many people's work. **Please read and credit accordingly** if you build on top:

### Core dependencies
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — the upstream project. This fork is just a thin patch on top. All the heavy lifting of inference is theirs.
- **[0cc4m](https://github.com/0cc4m)** — lead author of llama.cpp's Vulkan backend. Our changes sit on top of his architecture.
- **[jeffbolznv](https://github.com/jeffbolznv)** — major contributor to Vulkan shader optimization, including [PR #10296](https://github.com/ggml-org/llama.cpp/pull/10296) that vectorized the Q4_K GEMV kernel.

### BC-250 community
- **[akandr/bc250](https://github.com/akandr/bc250)** — the foundational setup guide for running AI workloads on BC-250. Source of the TTM tuning, Ollama configuration, Mesa version guidance, and a complete working Signal-bot + image-gen stack. **Start here for any BC-250 setup, not with this fork.**
- **[elektricM/amd-bc250-docs](https://github.com/elektricM/amd-bc250-docs)** — community documentation hub: BIOS flashing, kernel configuration, power management, governor setup. The definitive reference for BC-250 Linux users.
- **[mothenjoyer69/bc250-documentation](https://github.com/mothenjoyer69/bc250-documentation)** — the original "BC-250 as desktop" documentation effort.
- **[bc250-collective](https://github.com/bc250-collective/)** — reverse-engineered the SMU API that makes the governor possible.
- **[filippor/cyan-skillfish-governor](https://github.com/filippor/cyan-skillfish-governor/tree/smu)** — the adaptive GPU governor that unlocks gfx1013's real clock ceiling. **The single biggest performance contributor on our measurement** — larger than any change in this fork. Install this first; our llama.cpp changes are the small addition on top.
- **[Magnap/cyan-skillfish-governor](https://github.com/Magnap/cyan-skillfish-governor)** — the original governor implementation that filippor's `smu` branch builds on.

### Proof that RDNA1 inference can be fast
- **[Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire)** — showed that 47+ tok/s on gfx1013 is achievable with custom kernels. Without this existing as a benchmark target, we wouldn't have known what performance was sitting unclaimed on the silicon. The `redline` crate's DRM-ioctl approach also strongly influenced our thinking about what's possible.

### Similar fork precedent
- **[iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)** — the "snapshot fork for community of a specific AMD arch" pattern we're following here. Thank you for demonstrating that this is a valid way to share hardware-specific work.

### Hardware origin
The BC-250 silicon itself is AMD's "Cyan Skillfish" — a cut-down variant of the PS5 APU that AMD sold in bulk to a cryptocurrency mining operation. When the mining boards were liquidated, the enthusiast community found it to be a surprisingly capable Linux host. We owe the existence of this hardware-at-this-price to [community reverse-engineering of the PS5 APU lineage](https://www.phoronix.com/news/AMD-RADV-PS5-BC-250) and the [LLVM AMDGPU docs](https://llvm.org/docs/AMDGPUUsage.html#processors) that made gfx1013 a workable compile target.

---

## Known limitations

- **Q4_K only for the fused gate+up kernel.** Other quant formats (Q8_0, Q6_K, Q5_K, Q4_0, IQ variants) still use the unfused two-MUL_MAT path on the FFN.
- **Decode-only fusion.** Batched prefill (`NUM_COLS > 1`) doesn't use the fused kernel — falls back to upstream.
- **SSM / hybrid architectures** (Qwen 3.5's internal mix of transformer + SSM layers) trigger `ggml_backend_rpc_buffer_clear` aborts when used with `rpc-server`. Use standalone `llama-cli` / `llama-server` for these models; RPC with Qwen 3.5 specifically is broken upstream and not in this fork's scope to fix.
- **No AMDVLK testing.** The RDNA1 `rm_kq=4` may regress under AMDVLK — we only verified RADV.
- **No regular syncs with upstream.** Pinned to commit `f772f6e43`. Features added to upstream after that commit are not in this fork.

---

## License

This fork inherits the [MIT License](LICENSE) from upstream llama.cpp. Use freely.

If you rework and redistribute, please keep the credits section intact. Many people's unpaid effort is behind the tiny change in this fork.

---

*This fork produced by [Kato](https://github.com/PMZFX) while characterizing a 12-24 node BC-250 inference cluster. [`UPSTREAM-README.md`](UPSTREAM-README.md) is the preserved upstream llama.cpp README.*
