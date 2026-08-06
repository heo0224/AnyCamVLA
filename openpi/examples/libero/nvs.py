import importlib

from easydict import EasyDict as edict
import numpy as np
from omegaconf import OmegaConf
import PIL
import torch
import torch.nn.functional as F
from vis_utils import visualize_cameras


class LVSMWrapper:
    def __init__(self, config_path, device="cuda"):
        # Load configuration
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        self.config = edict(config)
        self.amp_dtype_mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "tf32": torch.float32,
        }

        self.device = device

        module, class_name = self.config.model.class_name.rsplit(".", 1)
        LVSM = importlib.import_module(module).__dict__[class_name]
        self.model = LVSM(self.config)

        self.model.load_ckpt(self.config.training.checkpoint_dir)

        # Move model to device after loading checkpoint
        self.model = self.model.to(self.device)

    def preprocess_frames(self, images_list, intrinsics_list, extrinsics_list, opengl=True, flip=False):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for i, img in enumerate(images_list):
            # img to PIL image
            if isinstance(img, np.ndarray):
                image = PIL.Image.fromarray((img).astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                image = PIL.Image.fromarray((img.cpu().numpy()).astype(np.uint8))
            else:
                image = img  # assume PIL image
            original_image_w, original_image_h = image.size

            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            images.append(image)
        for i in range(len(intrinsics_list)):
            fxfycxcy = intrinsics_list[i]  # fx, fy, cx, cy
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            intrinsics.append(fxfycxcy)
        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = np.array(extrinsics_list)
        flip_matrix = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        opengl_to_opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if flip:
            c2ws = c2ws @ flip_matrix
        if opengl:
            c2ws = c2ws @ opengl_to_opencv
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)  # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0)  # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)  # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)  # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device)  # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose)  # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        if scene_scale < 1e-6:
            scene_scale = 1.0

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def infer(self, input_images, input_cam_params, target_cam_params, visualize=False, vis_save_path=None):
        self.model.eval()
        intrinsics = np.concatenate([input_cam_params["intrinsics"], target_cam_params["intrinsics"]], axis=0)
        extrinsics = np.concatenate(
            [input_cam_params["extrinsic_matrix"], target_cam_params["extrinsic_matrix"]], axis=0
        )
        input_images, input_intrinsics, input_extrinsics = self.preprocess_frames(input_images, intrinsics, extrinsics)

        if visualize:
            pre_process_extrinsics = input_extrinsics.cpu().numpy()

        input_extrinsics = self.preprocess_poses(input_extrinsics)

        if visualize:
            post_process_extrinsics = input_extrinsics.cpu().numpy()
            visualize_cameras(
                pre_process_extrinsics,
                post_process_extrinsics,
                save_path=vis_save_path,
                opengl=False
            )

        input_data = {
            "image": input_images.unsqueeze(0).to(self.device),  # [1, v, c, h, w]
            "c2w": input_extrinsics.unsqueeze(0).to(self.device),  # [1, v, 4, 4]
            "fxfycxcy": input_intrinsics.unsqueeze(0).to(self.device),  # [1, v, 4]
        }
        with torch.autocast(device_type=self.device, dtype=self.amp_dtype_mapping[self.config.training.amp_dtype], enabled=self.config.training.use_amp), torch.no_grad():
            outputs = self.model(input_data, has_target_image=False)
        pred_image = outputs["render"]
        return pred_image.squeeze().cpu().float().numpy()  # [c, h, w]
