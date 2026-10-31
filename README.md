# Do We Need the Shadow?

**Freezing-aware weight versioning for pipeline-parallel training.**
Course project, Systems Engineering for Deep Learning (CS6886). Authors: Animesh Jain, Rupankar Podder.

The project asks whether **PipeDream-2BW**, a pipeline-parallel training scheme, needs its second weight copy (the *shadow*) to train well, and whether layer-freezing signals (**PolicyFRZ**) can decide which layers may drop it or be frozen. It also re-checks **DDG** (decoupled backprop with delayed gradients) against a modern baseline.

Everything runs on 2 GPUs. A $d$-stage pipeline is *emulated* exactly on one GPU: which weight version each stage uses depends only on the schedule, not on the number of devices. The emulator is validated bit-for-bit against a real 2-GPU pipeline.

---

## Repository layout

```
ShadowFRZ/
├── README.md
├── report/report.pdf              2-page project report (source: report/report.tex)
├── final_slides/slides.pdf        final presentation (findings)
├── proposal_slides/slides.pdf     proposal presentation
└── code_repo/
    ├── env.sh                     source first: caches + PYTHONPATH
    ├── requirements.txt
    ├── pipeemu/                   the pipeline emulator (core library)
    ├── models/unitnet.py          models expressed as sequences of "units"
    ├── data/                      dataset preparation and GPU-resident loaders
    ├── train_ddg.py               Track A: backprop vs. DDG on CIFAR ResNets
    ├── train_cnn.py               Track B: ResNet-50 under each pipeline/freezing variant
    ├── train_bert.py              Track C: BERT MLM pre-training under each variant
    ├── finetune_glue.py           SST-2/MNLI fine-tuning (downstream checks, public checkpoints)
    ├── tests/                     validation checks V0-V3
    ├── scripts/                   job queue, job generators, status
    ├── analysis/                  tables and figures from results/
    ├── jobs/final/                clean, dependency-ordered job lists for every reported run
    ├── jobs/history/              the incremental job files actually used during the study
    └── results/                   one directory per run (JSON logs, traces, summaries) + figures
```

(Model checkpoints, job logs, the planning documents and the slide sources are git-ignored; see `.gitignore`.)

---

## The codebase

### `pipeemu/`: the emulator

| File | What it does |
|---|---|
| `schedule.py` | Simulates PipeDream's 1F1B schedule. Produces the **version table** (which weight version every stage uses for every microbatch's forward and backward pass) for vanilla/Flush, 2BW and 1BW, and the closed-form 1BW rule. Also gives the steady-state time per batch for flush and flush-free pipelines. |
| `engine.py` | `PipelineEmulator`. Keeps two copies of each unit's weights (`cur`, `prev`) and, per unit, a mode: `exact` (vanilla), `shadow` (2BW: forward and backward on `prev`), `noshadow` (1BW: forward on `prev` only for microbatches in flight at an update, backward always on `cur`), `frozen`. The forward/backward mismatch is reproduced by overwriting weights in place (`p.data.copy_`) between forward and backward. |
| `freeze.py` | `FreezeController`. Per-unit PolicyFRZ statistics over a window of steps: gradient variation **GV** and mean magnitude **AM**, computed with fused GPU ops. Freezing policies: `off`, `gvam` (threshold rule), `random` (replays the counts of a `gvam` run), `freezeout` (fixed prefix schedule). Shadow policies: `all`, `none`, `gvam` (AM- or GV-based), `random`. Frozen units are periodically thawed for re-measurement. Writes a per-window trace. |
| `costmodel.py` | Profiles each unit (saved-activation bytes, forward/backward time in "training" and "frozen but gradients pass through" regimes). Contiguous DP stage partition, per-stage memory model, and 1F1B time per batch (static or rebalanced partition). |
| `runner.py` | The shared training loop for Tracks B and C. The `VARIANTS` table maps each ablation row to emulator modes and controller policies. Also handles the canonical stage partition and the cost-model summary over the trace. |

### `models/unitnet.py`
Every model is a sequence of **units**, the granularity used for stages, freezing and shadows. Interface: `net.units`, `net.prepare(batch)`, `net.unit_forward(i, x, ctx)` (the last unit returns the loss), `net.loss(batch)`.

- `ResNetUnits`: ResNet-56/110 (He et al. CIFAR) and ResNet-50 (CIFAR variant). Units are the stem, each residual block, and the head. `zero_init_residual=True` is used for ResNet-50.
- `BertPreLNMLMUnits`: Megatron-style Pre-LN BERT (Mini/Small/Medium/Base/Large configs). Units are the embedding, each layer, and the head. Used for all reported pre-training.
- `BertMLMUnits`: Hugging Face Post-LN BERT, used only by the archived edge-of-stability runs.
- `BertClsUnits`, `BertPreLNClsUnits`: classifiers built from public checkpoints, or from our pre-trained encoders.

### `data/`
- `prepare.py`: downloads CIFAR-10/100 (HF mirrors), tokenises WikiText-103 into packed 128-token sequences (`bert-base-uncased` WordPiece), and caches GLUE SST-2/MNLI and the `bert-base/large-uncased` checkpoints.
- `loaders.py`:
  - CIFAR lives entirely on the GPU, with crop and flip augmentation done there too.
  - WikiText MLM masking (15%, 80/10/10) on the GPU.
  - GLUE splits as padded GPU tensors.

### Trainers
| Script | Purpose | Key flags |
|---|---|---|
| `train_ddg.py` | Exact DDG (Huo et al., ICML'18). Each module's gradient is taken at the weights its forward pass used; weights are replaced, never modified in place, so old autograd graphs stay valid. `--K 1` is plain backprop. | `--model {resnet56,resnet110} --data {cifar10,cifar100} --K --recipe {modern,paper} --seed` |
| `train_cnn.py` | ResNet-50 under any variant; 160 epochs, SGD lr 0.1 cosine by default | `--variant --d --m --epochs` + freezing thresholds |
| `train_bert.py` | MLM pre-training under any variant | `--size {mini,small,medium,base,large} --arch {preln,postln} --variant --lr --steps --stop_at --save` + thresholds |
| `finetune_glue.py` | `--from_mlm DIR`: SST-2 downstream check of a pre-trained run (no pipeline). `--public NAME`: fine-tune a public checkpoint *under* a pipeline variant. | `--task {sst2,mnli} --epochs` |

**Variants** (`--variant`, defined in `pipeemu/runner.py`):

| Variant | Meaning |
|---|---|
| `vanilla` | no staleness (= PipeDream-Flush / GPipe semantics) |
| `2bw` | every unit keeps a shadow: uniform 1-step delay |
| `1bw` | no shadows: in-place updates, most microbatches fresh, a few mismatched |
| `frz` | vanilla + GV/AM freezing |
| `sel` | 2BW with shadows dropped only for "stable" units |
| `2bw_frz` | 2BW + freezing |
| `full` | `sel` + freezing (the proposed method) |
| `full_random` | `full`'s counts, random units (needs `--replay <trace of a full run>`) |
| `full_freezeout` | `sel` + FreezeOut-style fixed prefix freezing |
| `full_adamreset` | `full`, resetting optimizer state when a unit thaws |

**Freezing thresholds:**

| Flag | Meaning |
|---|---|
| `--frz_gv`, `--frz_am` | freeze when GV < θ1 and AM/AM(ref) < θ2 |
| `--sel_am` | drop the shadow when AM/AM(ref) < θ (use `--sel_gv` for a GV rule instead) |
| `--window` | statistics window, in steps |
| `--thaw_every` | re-measure frozen units every N windows |

The values used in the study are in `jobs/final/`. ResNet-50: `--frz_gv 0.3 --frz_am 0.5 --sel_am 0.8`. BERT: `--frz_gv 0.3 --frz_am 0.6 --sel_am 0.8`.

### Outputs
Each run writes to its `--out` directory:

| File | Contents |
|---|---|
| `config.json` | all CLI arguments |
| `log.jsonl` | training curve and periodic evaluation |
| `trace.jsonl` | per window: GV, AM, AM_rel and each unit's mode |
| `profile.json` | per-unit bytes and timings |
| `final.json` | final metrics, the weight-mismatch count, and the cost-model summary |
| `model.pt` | BERT checkpoint (with `--save`; git-ignored) |

`results/canonical/` holds the shared per-(model, depth) stage partitions. Every run that finds one uses it, so 1BW semantics and the cost model are identical across runs. A shared BERT-Base profile is also kept there, to skip a memory-hungry profiling pass.

### `scripts/`
| Script | Purpose |
|---|---|
| `queue.py` | Runs a job file (`name \| command` per line) with N jobs per GPU. Skips runs whose `final.json` exists. Starts a job only when the GPU has `--min-free-gb` free (waiting `--settle` seconds after each launch). Re-runs jobs that died of CUDA OOM (`--retries`). |
| `export_jobs.py` | Regenerates `jobs/final/*.txt` from the `config.json` of every finished run |
| `make_jobs.py`, `make_ft_jobs.py` | The generators used during the study (threshold rows, random replays, fine-tunes) |
| `status.py` | Compact progress report over all result directories |

---

## Reproduction

### 1. Check out and environment
```bash
git clone https://github.com/2coolcoder/ShadowFRZ.git
cd ShadowFRZ
```
- Tested with Python 3.12.7, PyTorch 2.9.0 + CUDA 12.8, on 2 × NVIDIA RTX 6000 Ada (48 GB each).
- Setup:

```bash
cd code_repo
pip install -r requirements.txt
export SYSDL_CACHE=/path/with/space   # optional; datasets + HF/torch caches go here (~7 GB)
source env.sh                         # sets caches, PYTHONPATH, allocator config
```

### 2. Data and checkpoints
```bash
python data/prepare.py          # CIFAR-10/100, WikiText-103 (tokenised), GLUE, bert-base/large
```

### 3. Validation (≈10 minutes)
```bash
python tests/test_schedule.py   # V0: closed-form 1BW rule == simulated 1F1B schedule (all d, m)
python tests/test_swap.py       # in-place weight swap gives the intended mismatched gradients
python tests/test_engine.py     # V1: vanilla == grad accumulation; V2: 2BW == closed form
# V3: real 2-stage pipeline on 2 GPUs vs. the emulator (expect max diff 0.0)
for v in vanilla 2bw 1bw; do
  torchrun --nproc_per_node 2 tests/realpipe_d2.py --variant $v && python tests/realpipe_d2.py --variant $v
done
```

### 4. Single experiments
```bash
python train_ddg.py  --model resnet56 --data cifar100 --K 4 --out results/demo/ddg_k4
python train_cnn.py  --variant 1bw --out results/demo/r50_1bw
python train_bert.py --size mini --lr 1e-3 --variant 1bw --save --out results/demo/mini_1bw
python finetune_glue.py --task sst2 --from_mlm results/demo/mini_1bw --out results/demo/mini_1bw_sst2
python finetune_glue.py --task sst2 --public bert-base-uncased --variant full --epochs 1 \
       --frz_gv 0.3 --frz_am 0.6 --sel_am 0.8 --window 50 --out results/demo/base_ft_full
```

### 5. The full study
Run the stages in order (later stages read earlier results). Each file lists the exact command of every reported run:

```bash
python scripts/queue.py jobs/final/1_trackA.txt        --gpus 0,1 --per-gpu 6 --min-free-gb 4
python scripts/queue.py jobs/final/2_trackC_lr.txt     --gpus 0,1 --per-gpu 3 --min-free-gb 12
python scripts/queue.py jobs/final/3_trackB.txt        --gpus 0,1 --per-gpu 2 --min-free-gb 10
python scripts/queue.py jobs/final/4_trackC.txt        --gpus 0,1 --per-gpu 3 --min-free-gb 12
python scripts/queue.py jobs/final/5_random_replay.txt --gpus 0,1 --per-gpu 1 --min-free-gb 10
python scripts/queue.py jobs/final/6_finetune.txt      --gpus 0,1 --per-gpu 2 --min-free-gb 18
python scripts/queue.py jobs/final/7_postln_edge.txt   --gpus 0,1 --per-gpu 3 --min-free-gb 8   # optional
python scripts/status.py                                # progress at any time
```

Budget and settings:
- The whole study took **≈44 h on both GPUs**.
- ResNet-50 runs are compute-bound (≈1.5 h each alone).
- BERT-Base needs ≈11–14 GB; BERT-Large fine-tuning ≈16 GB. Hence the higher `--min-free-gb` for those stages.
- Lower `--per-gpu` on smaller GPUs.

Two things the job files cannot regenerate:
- The `*_1bw_warm10_*` runs came from a since-fixed bug (1BW kept its shadow for the first 10% of training), so they are not in the job files; their result directories are kept as an extra data point.
- Runs that started before the canonical partitions existed used a per-run partition. Re-running them with the canonical one changes their 1BW mismatch pattern only marginally.

### 6. Tables and figures
```bash
python analysis/report.py all > results/report_all.md   # every table + results/figs/*.png
python analysis/final_figs.py                            # figures for the final slides
```

`report.py` recomputes every memory and time number with one canonical profile per model and the canonical partition, so variants are compared like-for-like.

---

## Implementation notes worth knowing

- **No autocast.** Runs use fp32 weights with TF32 matmuls. Under autocast, backward uses cached bf16 *copies* of the weights, so the in-place swap that emulates a no-shadow pipeline would be silently ignored. `channels_last` breaks it for convolutions for the same reason (`tests/test_swap.py` checks this).
- **BERT word embeddings and the MLM decoder are untied** in every variant. Tying would place one parameter on the first and the last stage.
- The MLM head runs only at masked positions: exact for the loss, and ~7× cheaper.
- Memory/time for pipelines deeper than 2 stages are **modelled**. Communication is not modelled. Differences between variants match a real 2-GPU pipeline exactly; absolute footprints are not claimed.

---

## Results at a glance

Summary in `report/report.pdf`; details and all numbers in `final_slides/slides.pdf` and `code_repo/results/report_all.md`.

- **DDG vs. a modern baseline:** DDG's own backprop baseline is 1–4 points below today's recipes. Against a modern one, K=2 modules still roughly matches backprop on CIFAR-10, while K=4 loses on every configuration.
- **The shadow is what makes 2BW stale.** Dropping it (1BW) recovers all of 2BW's accuracy loss on ResNet-50 and 50–78% of it in BERT pre-training, and all of it with 4 microbatches per pipeline stage. The early phase of training decides the gap.
- **A one-step delay narrows the usable learning rate.** Post-LN BERT with 2BW does not leave the pre-training plateau at rates where vanilla trains.
- **PolicyFRZ's gradient statistics did not single out converged layers during pre-training** against strong baselines. Freezing saved only ~4% of compute on ResNet-50, and the signal did no better than random layer choice. It does work when **fine-tuning** converged checkpoints (−16% stage-1 memory on BERT-Large at equal accuracy).
