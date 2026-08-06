import collections
import dataclasses
import json
import logging
import math
import pathlib
from time import time
from typing import List, Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero_utils import extract_camera_parameters, render_gt_with_camera_restoration
import numpy as np
from nvs import LVSMWrapper
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R
from spherical_camera_utils import apply_spherical_perturbation, get_camera_look_at_point_on_table
import torch
import tqdm
import tyro
from vis_utils import visualize_cameras
import yaml

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 format for video writing."""
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    dummy_policy: bool = False  # Whether to use dummy policy

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    experiment_name: str = "libero_cam_var"  # Name of the experiment

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: Optional[str] = None
    seed: int = 7  # Random Seed (for reproducibility)
    device: str = "cuda"  # Device to run the model on

    #################################################################################################################
    # Camera variation parameters
    #################################################################################################################
    spherical_variations: Optional[str] = None  # Spherical coordinate-based variations for agentview
    spherical_config: str = "config/spherical_camera_variations.json"  # Path to spherical variation config file
    camera_names: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
    )  # Names of cameras to extract parameters for

    #################################################################################################################
    # LVSM novel view synthesis parameters
    #################################################################################################################
    LVSM: bool = False  # Whether to use LVSM
    LVSM_config_path: str = "./LVSM/configs/inference_LVSM_LIBERO-Plus_custom.yaml"

    def __post_init__(self):
        if self.video_out_path is None:
            self.video_out_path = f"output/{self.experiment_name}/videos"

        if self.LVSM:
            self.NVS = LVSMWrapper(self.LVSM_config_path, device=self.device)
        else:
            self.NVS = None


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Load scene information for spherical variations
    scene_info_dict = {}
    if args.spherical_variations:
        scene_info_path = pathlib.Path(__file__).parent / "config" / f"{args.task_suite_name}_scene_info.json"
        if scene_info_path.exists():
            with open(scene_info_path, "r") as f:
                scene_info_dict = json.load(f)
            logging.info(f"Loaded scene information from: {scene_info_path}")
        else:
            logging.warning(f"Scene info file not found: {scene_info_path}. Using default scene center")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Create subdirectories for video streams
    video_subdirs = [
        'agentview',
        'robot0_eye_in_hand'
    ]
    if args.LVSM:
        video_subdirs.extend([
            'agentview_before_nvs',
            'agentview_after_nvs',
            'robot0_eye_in_hand_before_nvs',
            'robot0_eye_in_hand_after_nvs'
        ])
    for subdir in video_subdirs:
        (pathlib.Path(args.video_out_path) / subdir).mkdir(parents=True, exist_ok=True)

    try:
        args_save_path = pathlib.Path(args.video_out_path).parent / "args.yaml"
        with open(args_save_path, "w") as f:
            yaml.dump(dataclasses.asdict(args), f, default_flow_style=False, sort_keys=False)
        logging.info(f"Arguments saved to: {args_save_path}")
    except Exception as e:
        logging.error(f"Failed to save arguments: {e}")

    results_save_path = pathlib.Path(args.video_out_path).parent / "results.json"
    results_log = {
        "experiment_name": args.experiment_name,
        "total_summary": {},
        "tasks": {},
    }

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    if not args.dummy_policy:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    else:
        client = None

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Capture canonical camera parameters (before any perturbation)
        canonical_cam_params = extract_camera_parameters(
            env.sim,
            args.camera_names,
            image_height=LIBERO_ENV_RESOLUTION,
            image_width=LIBERO_ENV_RESOLUTION,
        )

        # camera pose variations
        new_cam_pos = {}
        new_cam_quat = {}
        camera_names = args.camera_names

        # Spherical coordinate-based camera variations (for agentview only)
        if args.spherical_variations:
            # Load spherical variations config
            spherical_json_path = pathlib.Path(args.spherical_config)
            if not spherical_json_path.is_absolute():
                spherical_json_path = pathlib.Path(__file__).parent / spherical_json_path

            with open(spherical_json_path, "r") as f:
                SPHERICAL_CONFIG = json.load(f)

            if "spherical" in SPHERICAL_CONFIG and args.spherical_variations in SPHERICAL_CONFIG["spherical"]:
                variation_level = args.spherical_variations
                logging.info(f"Applying '{variation_level}' spherical camera variations.")

                # Get workspace height for this task (prefer robot_base_z for consistency)
                task_key = f"task_{task_id}"
                if scene_info_dict and task_key in scene_info_dict:
                    # Use robot_base_z as primary reference (more consistent across scenes)
                    # Fall back to table_surface_height for backward compatibility
                    workspace_height = scene_info_dict[task_key].get("robot_base_z")
                    if workspace_height is None:
                        workspace_height = scene_info_dict[task_key].get("table_surface_height",
                                                                          scene_info_dict[task_key].get("table_height", 0.0))
                        logging.info(f"Using table_surface_height as workspace reference (backward compatibility)")
                else:
                    # Fallback: use workspace height 0
                    logging.warning(f"Scene info not found for {task_key}, using default workspace height")
                    workspace_height = 0.0

                # Apply spherical perturbation to agentview camera
                if "agentview" in SPHERICAL_CONFIG["spherical"][variation_level]:
                    cam_id = env.sim.model.camera_name2id("agentview")
                    original_cam_pos = env.sim.model.cam_pos[cam_id].copy()
                    original_cam_quat = env.sim.model.cam_quat[cam_id].copy()  # w, x, y, z

                    # Get reference point: intersection of camera center ray with workspace plane
                    reference_point = get_camera_look_at_point_on_table(
                        original_cam_pos, original_cam_quat, workspace_height
                    )

                    # Get spherical deltas
                    spherical_params = SPHERICAL_CONFIG["spherical"][variation_level]["agentview"]
                    delta_radius = spherical_params.get("delta_radius", 0.0)
                    delta_theta = spherical_params.get("delta_theta", 0.0)
                    delta_phi = spherical_params.get("delta_phi", 0.0)

                    # Apply spherical perturbation with reference point as origin
                    new_pos, new_quat_mujoco = apply_spherical_perturbation(
                        original_cam_pos, original_cam_quat, reference_point,
                        delta_radius, delta_theta, delta_phi
                    )

                    # Store new camera pose
                    new_cam_pos["agentview"] = new_pos
                    new_cam_quat["agentview"] = new_quat_mujoco

                    logging.info(f"  Agentview: Δr={delta_radius:.3f}m, Δθ={delta_theta:.1f}°, Δφ={delta_phi:.1f}°")
                    logging.info(f"  Original pos: {original_cam_pos}")
                    logging.info(f"  New pos: {new_pos}")
                    logging.info(f"  Reference point: {reference_point}")

        # Store original camera mounting (unperturbed)
        original_cam_pos_dict = {}
        original_cam_quat_dict = {}
        for camera_name in args.camera_names:
            cam_id = env.sim.model.camera_name2id(camera_name)
            original_cam_pos_dict[camera_name] = env.sim.model.cam_pos[cam_id].copy()
            original_cam_quat_dict[camera_name] = env.sim.model.cam_quat[cam_id].copy()

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_pbar = tqdm.tqdm(range(args.num_trials_per_task))
        for episode_idx in episode_pbar:
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            # Apply spherical camera variations
            if args.spherical_variations:
                for camera_name in new_cam_pos.keys():
                    cam_id = env.sim.model.camera_name2id(camera_name)
                    env.sim.model.cam_pos[cam_id] = new_cam_pos[camera_name]
                    env.sim.model.cam_quat[cam_id] = new_cam_quat[camera_name]
            # Step simulation to update camera extrinsics in sim.data
            env.sim.forward()
            # Compute relative transformation in LOCAL FRAME (Mounting)
            # T_corr = inv(T_local_pert) @ T_local_ref
            relative_camera_transforms = {}
            target_cameras = []

            for camera_name in camera_names:
                if args.spherical_variations and camera_name in new_cam_pos:
                    # Get Local Reference (Original)
                    pos_ref = original_cam_pos_dict[camera_name]
                    quat_ref = original_cam_quat_dict[camera_name]  # w, x, y, z
                    r_ref = R.from_quat(quat_ref[[1, 2, 3, 0]])
                    T_local_ref = np.eye(4)
                    T_local_ref[:3, :3] = r_ref.as_matrix()
                    T_local_ref[:3, 3] = pos_ref

                    # Get Local Perturbed (New)
                    pos_pert = new_cam_pos[camera_name]
                    quat_pert = new_cam_quat[camera_name]  # w, x, y, z
                    r_pert = R.from_quat(quat_pert[[1, 2, 3, 0]])
                    T_local_pert = np.eye(4)
                    T_local_pert[:3, :3] = r_pert.as_matrix()
                    T_local_pert[:3, 3] = pos_pert

                    T_local_pert_inv = np.linalg.inv(T_local_pert)
                    T_relative = T_local_pert_inv @ T_local_ref
                else:
                    T_relative = np.eye(4)

                relative_camera_transforms[camera_name] = T_relative
                if not np.isclose(T_relative, np.eye(4)).all():
                    target_cameras.append(camera_name)
            logging.info(f"Target cameras: {target_cameras}")

            # Setup
            t = 0

            # Multi-stream video collectors
            video_frames = {
                'agentview': [],  # Base policy agent view
                'robot0_eye_in_hand': [],  # Base policy wrist view
            }

            # Add NVS streams if enabled
            if args.LVSM:
                video_frames.update({
                    'agentview_before_nvs': [],
                    'agentview_after_nvs': [],
                    'robot0_eye_in_hand_before_nvs': [],
                    'robot0_eye_in_hand_after_nvs': [],
                })

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Save original images before NVS for video collection
                    img_before_nvs = img.copy()
                    wrist_img_before_nvs = wrist_img.copy()

                    # Render GT images with camera restoration
                    gt_img = None
                    gt_wrist_img = None
                    try:
                        gt_img = render_gt_with_camera_restoration(
                            env.sim, 'agentview',
                            original_cam_pos_dict, original_cam_quat_dict,
                            new_cam_pos, new_cam_quat,
                            LIBERO_ENV_RESOLUTION
                        )
                        gt_wrist_img = render_gt_with_camera_restoration(
                            env.sim, 'robot0_eye_in_hand',
                            original_cam_pos_dict, original_cam_quat_dict,
                            new_cam_pos, new_cam_quat,
                            LIBERO_ENV_RESOLUTION
                        )
                    except Exception as e:
                        logging.warning(f"GT rendering failed at frame {t}: {e}")
                        gt_img = None
                        gt_wrist_img = None

                    synthesized_images = {}

                    if args.LVSM:
                        # Get current camera parameters (perturbed)
                        current_cam_params = extract_camera_parameters(
                            env.sim,
                            args.camera_names,
                            image_height=LIBERO_ENV_RESOLUTION,
                            image_width=LIBERO_ENV_RESOLUTION,
                        )

                        # vertical flip
                        img_flipped = img[:, ::-1]
                        wrist_img_flipped = wrist_img[:, ::-1]
                        for target_cam_name in target_cameras:
                            # Joint inference: both cameras as context (matches the 2-input-view LVSM config)
                            input_images_list = [img_flipped, wrist_img_flipped]
                            input_cam_names = ["agentview", "robot0_eye_in_hand"]

                            # Construct input params for infer
                            input_cam_params_arg = {
                                "intrinsics": torch.stack(
                                    [torch.tensor(current_cam_params[name]["intrinsics"]) for name in input_cam_names]
                                ).numpy(),
                                "extrinsic_matrix": torch.stack(
                                    [
                                        torch.tensor(current_cam_params[name]["extrinsic_matrix"])
                                        for name in input_cam_names
                                    ]
                                ).numpy(),
                            }

                            current_extrinsic = np.array(current_cam_params[target_cam_name]["extrinsic_matrix"])
                            relative_extrinsic = relative_camera_transforms[target_cam_name]
                            target_extrinsic = current_extrinsic @ relative_extrinsic

                            target_cam_params_arg = {
                                "intrinsics": torch.stack(
                                    [torch.tensor(canonical_cam_params[target_cam_name]["intrinsics"])]
                                ).numpy(),
                                "extrinsic_matrix": torch.stack([torch.tensor(target_extrinsic)]).numpy(),
                            }

                            # Visualize cameras (only once)
                            if task_episodes == 0 and t == args.num_steps_wait:
                                vis_save_path = (
                                    pathlib.Path(args.video_out_path).parent / f"camera_vis_{target_cam_name}.png"
                                )
                                visualize_cameras(
                                    input_cam_params_arg["extrinsic_matrix"],
                                    target_cam_params_arg["extrinsic_matrix"],
                                    str(vis_save_path),
                                )
                                logging.info(f"Saved camera visualization to {vis_save_path}")

                            start_time = time()
                            visualize_lvsm = (task_episodes == 0 and t == args.num_steps_wait)
                            lvsm_vis_path = str(pathlib.Path(args.video_out_path).parent / f"lvsm_camera_vis_{target_cam_name}.png") if visualize_lvsm else None
                            synth_img = args.NVS.infer(input_images_list, input_cam_params_arg, target_cam_params_arg, visualize=visualize_lvsm, vis_save_path=lvsm_vis_path)
                            end_time = time()
                            episode_pbar.set_postfix(nvs_time=f"{end_time - start_time:.3f}s")
                            synth_img = synth_img[:, :, ::-1]
                            synth_img = np.transpose(synth_img, (1, 2, 0))  # (H, W, C)
                            synth_img = (synth_img * 255).clip(0, 255).astype(np.uint8)
                            synthesized_images[target_cam_name] = synth_img

                        if "agentview" in synthesized_images:
                            img = synthesized_images["agentview"]
                        if "robot0_eye_in_hand" in synthesized_images:
                            wrist_img = synthesized_images["robot0_eye_in_hand"]

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Always collect base policy frames
                    # Save GT images in agentview and robot0_eye_in_hand video folders
                    if gt_img is not None and gt_wrist_img is not None:
                        video_frames['agentview'].append(gt_img.copy())
                        video_frames['robot0_eye_in_hand'].append(gt_wrist_img.copy())
                    else:
                        # Fallback to original images if GT rendering failed
                        video_frames['agentview'].append(img_before_nvs.copy())
                        video_frames['robot0_eye_in_hand'].append(wrist_img_before_nvs.copy())

                    # If NVS is enabled, also collect before/after NVS streams
                    if args.LVSM:
                        video_frames['agentview_before_nvs'].append(img_before_nvs.copy())
                        video_frames['robot0_eye_in_hand_before_nvs'].append(wrist_img_before_nvs.copy())

                        # After NVS (or original if NVS not used)
                        if "agentview" in synthesized_images:
                            video_frames['agentview_after_nvs'].append(synthesized_images["agentview"].copy())
                        else:
                            video_frames['agentview_after_nvs'].append(img_before_nvs.copy())

                        if "robot0_eye_in_hand" in synthesized_images:
                            video_frames['robot0_eye_in_hand_after_nvs'].append(synthesized_images["robot0_eye_in_hand"].copy())
                        else:
                            video_frames['robot0_eye_in_hand_after_nvs'].append(wrist_img_before_nvs.copy())

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        if args.dummy_policy:
                            action_chunk = np.array([LIBERO_DUMMY_ACTION] * args.replan_steps)
                        else:
                            action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}", exc_info=True)
                    t += 1
                    continue

            task_episodes += 1
            total_episodes += 1

            # Save all video streams (2 base + 4 NVS if enabled)
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            episode_suffix = f"{task_segment}_ep{episode_idx}_{suffix}"

            for stream_name, frames in video_frames.items():
                if len(frames) > 0:
                    video_path = pathlib.Path(args.video_out_path) / stream_name / f"rollout_{episode_suffix}.mp4"
                    imageio.mimwrite(
                        video_path,
                        [np.asarray(ensure_uint8(x)) for x in frames],
                        fps=10,
                    )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            try:
                results_log["total_summary"] = {
                    "total_episodes": total_episodes,
                    "total_successes": total_successes,
                    "total_success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
                }
                results_log["tasks"][task_description] = {
                    "task_id": task_id,
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "success_rate": float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0,
                }
                with open(results_save_path, "w") as f:
                    json.dump(results_log, f, indent=4)

            except Exception as e:
                logging.error(f"Failed to save results log: {e}")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
