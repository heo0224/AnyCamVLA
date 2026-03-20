# AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2603.05868-b31b1b.svg)](https://arxiv.org/abs/2603.05868)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://heo0224.github.io/AnyCamVLA/)

**Authors:** Hyeongjun Heo, Seungyeon Woo, Sang Min Kim, Junho Kim, Junho Lee, Yonghyeon Lee, Young Min Kim

**Affiliations:** Seoul National University, Massachusetts Institute of Technology

## Abstract

Despite remarkable progress in Vision-Language-Action models (VLAs) for robot manipulation, these large pre-trained models require fine-tuning to be deployed in specific environments. These fine-tuned models are highly sensitive to camera viewpoint changes that frequently occur in unstructured environments. In this paper, we propose a zero-shot camera adaptation framework without additional demonstration data, policy fine-tuning, or architectural modification. Our key idea is to virtually adjust test-time camera observations to match the training camera configuration in real-time. For that, we use a recent feed-forward novel view synthesis model which outputs high-quality target view images, handling both extrinsic and intrinsic parameters.

This plug-and-play approach preserves the pre-trained capabilities of VLAs and applies to any RGB-based policy. Through extensive experiments on the LIBERO benchmark, our method consistently outperforms baselines that use data augmentation for policy fine-tuning or additional 3D-aware features for visual input. We further validate that our approach constantly enhances viewpoint robustness in real-world robotic manipulation scenarios, including settings with varying camera extrinsics, intrinsics, and freely moving handheld cameras.

## 🚀 Code Release (Coming Soon)

The code for AnyCamVLA will be released soon. Stay tuned!

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
- Hyeongjun Heo: [https://heo0224.github.io/](https://heo0224.github.io/)

## 🔗 Links

- **Project Page:** [https://heo0224.github.io/AnyCamVLA/](https://heo0224.github.io/AnyCamVLA/)
- **Paper:** [https://arxiv.org/abs/2603.05868](https://arxiv.org/abs/2603.05868)

## License

This project is under review. License information will be updated upon publication.
