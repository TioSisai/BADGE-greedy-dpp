# Greedy Volume Maximization of Gradient Embeddings for Long-Tailed Frame-Level Bioacoustic Active Learning

Official implementation of [*Greedy Volume Maximization of Gradient Embeddings for Long-Tailed Frame-Level Bioacoustic Active Learning*](https://arxiv.org/abs/2607.13555) by Shiqi Zhang, Marius Faiß, Ariana Strandburg-Peshkin, and Tuomas Virtanen.

The experiments compare eight active learning query strategies and a full-supervised reference on HyenaSET. A small MLP head is trained on frozen animal2vec frame embeddings, and each query selects 10-second segments for frame-level annotation.

| CLI strategy | Paper name |
|---|---|
| `random` | Random |
| `entropy` | Entropy |
| `farthest-traversal` | Farthest Traversal |
| `disagreement` | Disagreement |
| `mfft` | MFFT |
| `badge-kmeans++` | Vanilla BADGE (KMeans++) |
| `badge-mcmc-dpp` | Vanilla BADGE (MCMC DPP) |
| `badge-greedy-dpp` | BADGE Greedy DPP (proposal) |
| `full-supervised` | Full supervised |

The default protocol selects 300 segments per round for 10 rounds and repeats every strategy over 10 runs with seeds 0 to 9. Round 0 is a random cold-start set shared by every strategy at the same seed.

## Repository layout

```
scripts/           # experiment entry, multi-GPU launcher, summary tables, figures
src/               # active learning loop and the subpackages below
├── data/          # cache loading
├── models/        # MLP head, training round, frame-level mAP
├── analysis/      # metric definitions, significance tests, figure styling
└── strategies/    # query strategies
    ├── badge.py   # BADGE gradient embedding
    └── dpp.py     # greedy DPP selection
results/           # per-round results of the runs reported in the paper
tests/             # synthetic-data tests, CPU only
```

The proposal consists of `badge_gradient` in [src/strategies/badge.py](src/strategies/badge.py), which builds the gradient embedding of every segment, and `greedy_dpp` in [src/strategies/dpp.py](src/strategies/dpp.py), which runs the greedy selection on it.

## Setup

Linux is required, and the versions pinned in [requirements.txt](requirements.txt) target Python 3.12 with CUDA 13.2. The multi-GPU launcher also needs `nvidia-smi` and `taskset`.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
# Optional GPU selection backend. Package names must match the CUDA major version.
pip install -r requirements-gpu.txt --extra-index-url https://pypi.nvidia.com
```

Without the optional GPU packages the selection falls back to sklearn and numpy on CPU, and such results are written to a separate configuration directory. Tests use synthetic data and run on CPU with `python -m pytest`.

## Data and paths

Copy [.env.example](.env.example) to `.env` and set `CACHE_ROOT` and `OUTPUT_ROOT`. Scripts load `.env` automatically, and the `--cache-root` and `--output-root` arguments take precedence. The cache is published on Zenodo with DOI [10.5281/zenodo.22849166](https://doi.org/10.5281/zenodo.22849166). Unpack it into `HyenaSET/` under `CACHE_ROOT` (about 4.5 GiB, no raw audio) to get the files below.

| File | Shape and dtype | Content |
|---|---|---|
| `{train,val,test}_embedding.npy` | `[N, 20, 768]` float32 | animal2vec frame embeddings, read by the pipeline |
| `{train,val,test}_label.npy` | `[N, 20, 10]` float32 | Frame-level multi-hot call type labels, read by the pipeline |
| `train_seg_embedding.npy` | `[51760, 768]` float32 | Mean embedding over the 20 frames of each train segment, read by the pipeline |
| `meta.json` | JSON | Class names and array shapes, read by the pipeline, `summarize.py`, and `plot_enrichment.py` |
| `train_seg_labels.npy` | `[51760, 10]` int64 | Max-pooled labels over the 20 frames of each train segment, read by `summarize.py` and `plot_enrichment.py` for the rare call type enrichment |
| `{train,val,test}_valid_lengths.npy` | `[N]` int64 | Valid frames per segment, always 20, shipped for reference |
| `segments.csv` | CSV | `split`, `index`, `audio_filename`, `seg_start`, `seg_end` of every array row, shipped for reference |

The train, validation, and test splits hold 51760, 11091, and 11092 segments. The last label axis follows the class order `oth`, `whp`, `rum`, `grn`, `gig`, `str`, `fed`, `sql`, `snr`, `gwl`, and the paper reports `gwl`, `snr`, and `str` separately as the rare call types. See [CACHE_LICENSES.md](CACHE_LICENSES.md) for how the cache was built and its license.

## Run experiments

```bash
python scripts/pipeline.py
```

Run this from the repository root. It covers the eight query strategies and 10 seeds on one GPU. `--strategies` selects a subset, `--seed` and `--runs` select consecutive seeds starting at `--seed`, and `--n-tasks` sets the number of parallel workers sharing the GPU (default 4). Parallel workers contend for the card, so a run whose `query_time` is meant to be read needs `--n-tasks 1` and one process per card.

For multiple GPUs, [scripts/run_pipeline.sh](scripts/run_pipeline.sh) binds each process to the GPU given by `--cuda-id` and to its NUMA-local CPU cores, and forwards the other arguments to `pipeline.py`. It uses `python` from PATH, or the interpreter named by `BADGE_GREEDY_DPP_PYTHON`. The commands below split the default experiments by seed over four cards.

```bash
# Launch each command in a separate terminal or job step.
scripts/run_pipeline.sh --cuda-id 0 --seed 0 --runs 2
scripts/run_pipeline.sh --cuda-id 1 --seed 2 --runs 2
scripts/run_pipeline.sh --cuda-id 2 --seed 4 --runs 3
scripts/run_pipeline.sh --cuda-id 3 --seed 7 --runs 3
```

Strategy and seed combinations must be disjoint across processes, since processes that share a combination overwrite each other's outputs. The full-supervised reference labels the whole train pool in a single round and supplies the level that FS-Bud and FS-Rch in the summary table are measured against. It is outside the default strategy set and is launched separately.

```bash
scripts/run_pipeline.sh --cuda-id 0 --strategies full-supervised --n-tasks 1
```

### Outputs and resuming

```
{OUTPUT_ROOT}/{dataset}/{strategy}/{config_hash}/
├── config.json
└── seed_{s}/results.csv
```

`config_hash` is derived from the settings in `config.json`, so changed settings land in a new directory. Rerunning the same command skips seeds whose `results.csv` is complete and restarts incomplete seeds from round 0. `results.csv` holds one row per round with the columns below.

| Column | Content |
|---|---|
| `iteration` | Round index, starting at 0 |
| `num_labeled_samples` | Cumulative number of labeled segments |
| `query_time` | Wall-clock seconds of the query stage, excluding training and testing |
| `val_mAP`, `test_mAP` | Macro frame-level mAP on the validation and test splits |
| `test_AULC` | Normalized area under the `test_mAP` curve up to that round, blank at round 0 |
| `test_mAP_{class}` | Test average precision of each class |
| `queried_idxes_in_latest_iteration` | JSON list of the train rows selected in that round |

## Tables and figures

```bash
python scripts/summarize.py
python scripts/plot_curves.py
python scripts/plot_enrichment.py
```

`summarize.py` reads the complete seeds and writes `HyenaSET_main.{md,csv}`, `HyenaSET_permutation.{md,csv}`, and `HyenaSET_wilcoxon.{md,csv}` under `{OUTPUT_ROOT}/tables/`. The main table reports the mean and the sample standard deviation over seeds for the metrics below.

| Metric | Definition |
|---|---|
| N-AULC | Normalized area under the `test_mAP` curve over the labeled budget, in percent |
| Rare-N-AULC | The same area for the mean of `test_mAP_gwl`, `test_mAP_snr`, and `test_mAP_str`, in percent |
| F-mAP | `test_mAP` of the final round, in percent |
| F-rmAP | Mean of the three rare-class columns at the final round, in percent |
| R-Enr | Rare call type prevalence in the selected set divided by that in the train pool, averaged over the three types |
| FS-Bud | Smallest `num_labeled_samples` whose `test_mAP` reaches the mean final `test_mAP` of the full-supervised runs, averaged over the runs that reach it |
| FS-Rch | Share of runs that reach that level, in percent |
| QT | `query_time` summed over rounds 1 to 9, in seconds |

The permutation table holds a paired sign-flip permutation test with Holm-Bonferroni correction and the Wilcoxon table holds an exact paired Wilcoxon signed-rank test, both run on the first five metrics against a reference strategy that defaults to `badge-greedy-dpp` and is changed with `--reference`. If a strategy has several configuration directories, select one with `--config-hash PREFIX ...` or `--config-filter device=cuda backend=cuml+cupy` in any of the three scripts.

`plot_curves.py` draws the mean test mAP of every strategy against the labeled budget with a minimum to maximum band across seeds, and the full-supervised reference as a dashed line. `plot_enrichment.py` draws the enrichment of `gwl`, `snr`, and `str` in the selected sets relative to the train pool, with one standard deviation error bars. Figures are saved as SVG, PDF, and PNG under `{OUTPUT_ROOT}/figures/`.

## Results of the paper runs

[results/reference_result_on_mahti.csv](results/reference_result_on_mahti.csv) is the per-round record of the runs reported in the paper, covering the eight query strategies and the full-supervised reference over the same 10 seeds.

A fresh run may not match these numbers exactly because the hardware and software environment differ. In this repo, IO has been optimized, the time costs for all strategies has seen an overall decrease; And round 0 is now shared across strategies per seed, and the significance tests are now paired, while the ranking of the strategies is preserved.

## Citation and license

```bibtex
@misc{zhang2026greedy,
  title         = {Greedy Volume Maximization of Gradient Embeddings for Long-Tailed Frame-Level Bioacoustic Active Learning},
  author        = {Zhang, Shiqi and Fai{\ss}, Marius and Strandburg-Peshkin, Ariana and Virtanen, Tuomas},
  year          = {2026},
  eprint        = {2607.13555},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  doi           = {10.48550/arXiv.2607.13555},
  url           = {https://arxiv.org/abs/2607.13555}
}
```

Code is released under the [MIT license](LICENSE). The [published HyenaSET dataset](https://doi.org/10.17617/3.8ZSP3J) uses [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [cache licenses and sources](CACHE_LICENSES.md) for cache provenance, attribution, and version scope.
