<div align="center">

# LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias 

### ICLR 2025 (Oral)

<p align="center">  
    <a href="https://haian-jin.github.io/">Haian Jin</a>,
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang</a>,
    <a href="https://www.cs.unc.edu/~airsplay/">Hao Tan</a>,
    <a href="https://kai-46.github.io/website/">Kai Zhang</a>,
    <a href="https://sai-bi.github.io/">Sai Bi</a>,
    <a href="https://tianyuanzhang.com/">Tianyuan Zhang</a>,
    <a href="https://luanfujun.com/">Fujun Luan</a>,
    <a href="https://www.cs.cornell.edu/~snavely/">Noah Snavely</a>,
    <a href="https://zexiangxu.github.io/">Zexiang Xu</a>

</p>


</div>


<div align="center">
    <a href="https://haian-jin.github.io/projects/LVSM/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2410.17242"><strong>Paper</strong></a> 
</div>

<br>


## 0. Clarification

This is the **official repository** for the paper _"LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias"_.

The code here is a **re-implementation** and **differs** from the original version developed at Adobe. However, the provided checkpoints are from the original Adobe implementation and were trained inside Adobe.

We have verified that the re-implemented version matches the performance of the original. For any questions or issues, please contact Haian Jin at [haianjin0415@gmail.com](mailto:haianjin0415@gmail.com).

---



## 1. Preparation

### Environment
```
conda create -n LVSM python=3.11
conda activate LVSM
pip install -r requirements.txt
```
As we used [xformers](https://github.com/facebookresearch/xformers) `memory_efficient_attention`, the GPU device compute capability needs > 8.0. Otherwise, it would pop up an error. Check your GPU compute capability in [CUDA GPUs Page](https://developer.nvidia.com/cuda-gpus#compute).

### Data
Download the RealEstate10K dataset from [this link](http://schadenfreude.csail.mit.edu:8000/), which is provided by [pixelSplat](https://github.com/dcharatan/pixelsplat), and `unzip` the zip file and put the data in `YOUR_RAW_DATAPATH`.
Run the following command to preprocess the data into our format.
```bash
python process_data.py --base_path YOUR_RAW_DATAPATH --output_dir YOUR_PROCESSED_DATAPATH --mode ['train' or 'test']
```

### Checkpoints
The scene-level evaluation is conducted on the [RealEstate10K](http://schadenfreude.csail.mit.edu:8000/) dataset prepocessed by [pixelSplat](https://github.com/dcharatan/pixelsplat). The model checkpoints are host on [HuggingFace](https://huggingface.co/coast01/LVSM/tree/main). 

| Model | PSNR  | SSIM  | LPIPS |
| ----- | ----- | ----- | ----- |
| [LVSM Decoder-Only Scene-Level res256×256 (full)](https://huggingface.co/coast01/LVSM/resolve/main/scene_decoder_only_256.pt?download=true) | 29.67 | 0.906 | 0.098 |
| [LVSM Encoder-Decoder Scene-Level res256×256 (full)](https://huggingface.co/coast01/LVSM/resolve/main/scene_encoder_decoder_256.pt?download=true) | 28.60 | 0.893 | 0.114 |
| [LVSM Decoder-Only Scene-Level res512×512](https://huggingface.co/coast01/LVSM/blob/main/lvsm_scene_decoder_only_res512.pt) | N/A | N/A | N/A |
| [LVSM Encoder-Decoder Scene-Level res512×512](https://huggingface.co/coast01/LVSM/blob/main/lvsm_scene_encoder_decoder_res512.pt) | N/A | N/A | N/A |

As we discussed in the limitation sections of this paper:

>Our model’s performance degrades when provided with images with aspect ratios and resolutions different from those seen during training.

Therefore, if you plan to use the model for inference at resolutions or aspect ratios different from those used to train our provided checkpoints (256×256 or 512×512), we recommend fine-tuning the model for the specific resolution and aspect ratio.



## 2. Training

Before training, you need to follow the instructions [here](https://docs.wandb.ai/guides/track/public-api-guide/#:~:text=You%20can%20generate%20an%20API,in%20the%20upper%20right%20corner.) to generate the Wandb key file for logging and save it in the `configs` folder as `api_keys.yaml`. You can use the `configs/api_keys_example.yaml` as a template.

The original training command:
```bash
torchrun --nproc_per_node 8 --nnodes 8 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/LVSM_scene_decoder_only.yaml
```
The training will be distributed across 8 GPUs and 8 nodes with a total batch size of 512.
`LVSM_scene_decoder_only.yaml` is the config file for the scene-level Decoder-Only LVSM model. You can also use `LVSM_scene_encoder_decoder.yaml` for the training of the scene-level Encoder-Decoder LVSM model.

If you have limited resources, you can use the following command to train a smaller model with a smaller batch size:
```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/LVSM_scene_decoder_only.yaml \
    model.transformer.n_layer = 12 \
    training.batch_size_per_gpu = 16

```
Here, we decrease the total batch size from 512 to 128, and the transformer layers from 24 to 12. You can also increase the patch-size from 8 to 16 for faster training with lower performance. 
We have also discussed the efficient settings (single/two GPU training) in the paper.


## 3. Inference

```bash
torchrun --nproc_per_node 8 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
inference.py --config "configs/LVSM_scene_decoder_only.yaml" \
training.dataset_path = "./preprocessed_data/test/full_list.txt" \
training.batch_size_per_gpu = 4 \
training.target_has_input =  false \
training.num_views = 5 \
training.square_crop = true \
training.num_input_views = 2 \
training.num_target_views = 3 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = true \
inference_out_dir = ./experiments/evaluation/test
```
We use `./data/evaluation_index_re10k.json` to specify the input and target view indice. This json file is originally from [pixelSplat](https://github.com/dcharatan/pixelsplat). 

After the inference, the code will generate a html file in the `inference_out_dir` folder. You can open the html file to view the results.

## 4. Citation 

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{
jin2025lvsm,
title={LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias},
author={Haian Jin and Hanwen Jiang and Hao Tan and Kai Zhang and Sai Bi and Tianyuan Zhang and Fujun Luan and Noah Snavely and Zexiang Xu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=QQBPWtvtcn}
}
```

## 5. Acknowledgement
We thank Kalyan Sunkavalli for helpful discussions and support. This work was done when Haian Jin, Hanwen Jiang, and Tianyuan Zhang were research interns at Adobe Research.  This work was also partly funded by the National Science Foundation (IIS-2211259, IIS-2212084).

