# AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2603.05868-b31b1b.svg)](https://arxiv.org/abs/2603.05868)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://heo0224.github.io/AnyCamVLA/)


**🎉 Our paper has been accepted to IROS 2026!**

**Authors:** Hyeongjun Heo, Seungyeon Woo, Sang Min Kim, Junho Kim, Junho Lee, Yonghyeon Lee, Young Min Kim

**Affiliations:** Seoul National University, Massachusetts Institute of Technology

> **📌 Scope:** This repository covers the **LIBERO simulation experiments with the π₀.₅ (pi05) policy** from our paper — evaluating pi05 under camera viewpoint changes, with and without LVSM-based zero-shot camera adaptation. Other parts of the paper (e.g., real-world robot experiments) are not included here.

## 🗂️ Repository Structure

```
AnyCamVLA/
├── openpi/                     # openpi (Physical-Intelligence/openpi @ 95aadc6, vendored)
│   ├── examples/libero/        # LIBERO evaluation client with camera perturbation + LVSM NVS
│   │   ├── main.py             # Evaluation entry point
│   │   ├── nvs.py              # LVSMWrapper (novel view synthesis)
│   │   ├── spherical_camera_utils.py
│   │   ├── libero_utils.py
│   │   ├── vis_utils.py
│   │   └── config/             # Spherical variation levels + per-suite scene info
│   └── third_party/libero      # LIBERO benchmark (git submodule)
├── LVSM/                       # LVSM (Haian-Jin/LVSM @ ef1dff3, vendored)
│   ├── configs/inference_LVSM_LIBERO-Plus_custom.yaml
│   └── ckpt/LIBERO-Plus_custom/  # Finetuned LVSM checkpoint (not in git, see below)
├── metric_checkpoint/          # VGG weights for LVSM perceptual loss init (not in git)
└── output/                     # Evaluation outputs (not in git)
```

Files **not included in git**:
- `LVSM/ckpt/LIBERO-Plus_custom/ckpt_0000000000020000.pt` — LVSM full-finetune checkpoint (~1.3 GB, trained on our custom LIBERO-Plus dataset for 20k steps). Download from [heo0224/AnyCamVLA-LVSM](https://huggingface.co/heo0224/AnyCamVLA-LVSM):
  ```bash
  hf download heo0224/AnyCamVLA-LVSM ckpt_0000000000020000.pt --local-dir LVSM/ckpt/LIBERO-Plus_custom
  ```
- `metric_checkpoint/imagenet-vgg-verydeep-19.mat` — VGG weights (~535 MB); auto-downloaded on first LVSM load if missing

The LVSM training dataset is also available at [heo0224/LIBERO-Plus_custom](https://huggingface.co/datasets/heo0224/LIBERO-Plus_custom) (only needed to re-finetune LVSM, not for evaluation).

## 🛠️ Setup

Requires [uv](https://docs.astral.sh/uv/). All commands run from the repository root.

```bash
git clone --recursive --single-branch https://github.com/heo0224/AnyCamVLA.git
cd AnyCamVLA   # if you already cloned without --recursive: git submodule update --init
uv venv --python 3.8 .venv
source .venv/bin/activate
uv pip sync openpi/examples/libero/requirements.txt --index-strategy=unsafe-best-match
uv pip install -e openpi/packages/openpi-client
```

Notes:
- LIBERO's own `requirements.txt` is **not** synced directly (it pins `einops==0.4.1`, which conflicts with LVSM); its dependencies are already included in `openpi/examples/libero/requirements.txt`.
- If building `evdev` or `egl-probe` fails (e.g. a conda toolchain — `cc`, `cmake` — shadows the system one on your `PATH`), prefer the system toolchain for the sync: `PATH="/usr/bin:$PATH" uv pip sync ...`.
- LIBERO is used via `PYTHONPATH` (not pip-installed). LIBERO stores its dataset paths in `~/.libero/config.yaml`; if the file already exists it takes precedence, so make sure it points to the LIBERO tree (bddl files / init states) you intend to use.
- If off-screen rendering fails with an EGL error, prefix commands with `MUJOCO_GL=glx`.

## 🕹️ Running LIBERO Evaluation

**1. Start a policy server** (on any machine with the pi05 LIBERO checkpoint, using stock openpi):

```bash
uv run scripts/serve_policy.py --env LIBERO
```

**2. Run the evaluation client** from the AnyCamVLA root:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/openpi/third_party/libero:$PWD/LVSM:$PYTHONPATH"
```

Base policy under spherical camera perturbation (no view synthesis):

```bash
python openpi/examples/libero/main.py \
  --args.host <SERVER_HOST> --args.port <SERVER_PORT> \
  --args.task_suite_name <suite> \
  --args.spherical_variations combined_<level> \
  --args.experiment-name pi05/<suite>/base/spherical_combined_<level>
```

With LVSM novel view synthesis (re-renders the perturbed view back to the training viewpoint before feeding the policy):

```bash
python openpi/examples/libero/main.py \
  --args.host <SERVER_HOST> --args.port <SERVER_PORT> \
  --args.task_suite_name <suite> \
  --args.spherical_variations combined_<level> \
  --args.LVSM --args.LVSM_config_path "./LVSM/configs/inference_LVSM_LIBERO-Plus_custom.yaml" \
  --args.experiment-name pi05/<suite>/LVSM_LIBERO-Plus_custom/spherical_combined_<level>
```

- `<suite>`: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- `<level>`: `small` (Δθ=10°), `medium` (Δr=0.1m, Δθ=30°, Δφ=5°), `large` (Δr=0.15m, Δθ=60°, Δφ=10°) — see `openpi/examples/libero/config/spherical_camera_variations.json`

**Wrist camera perturbation** — perturbs the wrist camera's mounting pose instead of the agentview orbit. In either command above, replace the `--args.spherical_variations combined_<level>` flag with:

```bash
  --args.camera-pose-variations wrist_<level> \
  --args.pose_noise_config "config/wrist_camera_variations.json"
```

- `wrist_<level>`: `wrist_small` (+3cm along the mount x-axis), `wrist_medium` (+4cm, −4° roll), `wrist_large` (+5cm/+2cm, −8° roll) — see `openpi/examples/libero/config/wrist_camera_variations.json`
- Use experiment names like `pi05/<suite>/base/wrist_<level>` and `pi05/<suite>/LVSM_LIBERO-Plus_custom/wrist_<level>`.

Outputs are written to `output/<experiment-name>/`: `results.json` (per-task and total success rates), `args.yaml`, rollout videos under `videos/` (ground-truth restored view, and `*_before_nvs` / `*_after_nvs` streams when LVSM is enabled), and one-time camera geometry visualizations (`camera_vis_*.png`).

**Smoke test without a policy server** (uses dummy actions; verifies env, camera perturbation, and LVSM inference end-to-end):

```bash
python openpi/examples/libero/main.py --args.dummy_policy \
  --args.task_suite_name libero_spatial --args.spherical_variations combined_small \
  --args.num_trials_per_task 1 --args.experiment-name smoke/base
```

## 📖 Citation

If you find our work useful, please consider citing:

```bibtex
@article{heo2026anycamvla,
  title={AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models},
  author={Heo, Hyeongjun and Woo, Seungyeon and Kim, Sang Min and Kim, Junho and Lee, Junho and Lee, Yonghyeon and Kim, Young Min},
  journal={arXiv preprint arXiv:2603.05868},
  year={2026}
}
```

## 📧 Contact

For questions or collaborations, please contact:
- Hyeongjun Heo: heo0224@snu.ac.kr ([personal page](https://heo0224.github.io/))

## 🔗 Links

- **Project Page:** [https://heo0224.github.io/AnyCamVLA/](https://heo0224.github.io/AnyCamVLA/)
- **Paper:** [https://arxiv.org/abs/2603.05868](https://arxiv.org/abs/2603.05868)

## 🙏 Acknowledgements

This repository is built on top of the following open-source projects. The `openpi/` and `LVSM/` directories are vendored snapshots of the corresponding upstream repositories (at the commits noted in the repository structure above) with modifications for our evaluation pipeline:

- [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence — the VLA policy framework and LIBERO evaluation client that our pipeline extends
- [LVSM](https://github.com/Haian-Jin/LVSM) by Haian Jin et al. — the feed-forward novel view synthesis model we finetune and use for zero-shot camera adaptation
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — the simulation benchmark used in our experiments (git submodule at `openpi/third_party/libero`)

## License

Each vendored component keeps its original upstream license:

- `openpi/` — [Apache License 2.0](openpi/LICENSE), following [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) (Gemma model weights are additionally subject to the [Gemma terms of use](openpi/LICENSE_GEMMA.txt))
- `LVSM/` (including the finetuned checkpoint) — [CC BY-NC-SA 4.0](LVSM/LICENSE.md), following [Haian-Jin/LVSM](https://github.com/Haian-Jin/LVSM); note that this is a **non-commercial** license
- `openpi/third_party/libero/` — [MIT License](openpi/third_party/libero/LICENSE), following [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

Our modifications and additions within each directory are released under the same license as the component they belong to.
