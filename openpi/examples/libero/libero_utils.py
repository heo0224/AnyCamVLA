import logging

import numpy as np
from scipy.spatial.transform import Rotation as R


def extract_camera_parameters(sim, camera_names, image_height, image_width, rotation_z_180=False):
    """
    Extracts extrinsic and intrinsic parameters for specified cameras.

    Args:
        sim: MuJoCo simulation object (env.sim)
        camera_names: List of camera names strings
        image_height: Height of the captured image (for intrinsics)
        image_width: Width of the captured image (for intrinsics)

    Returns:
        dict: A dictionary containing params for each camera.
    """
    camera_params = {}

    for cam_name in camera_names:
        cam_id = sim.model.camera_name2id(cam_name)

        # ---------------------------------------------------------
        # 1. Extrinsics (World -> Camera)
        # We fetch data from sim.data because it changes every step.
        # ---------------------------------------------------------

        # Position (x, y, z) in World Coordinates
        cam_pos = sim.data.cam_xpos[cam_id].copy()

        # Rotation Matrix (3x3) in World Coordinates
        cam_rot_mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()

        # Construct 4x4 Transformation Matrix (Camera-to-World)
        # T_c2w = [[R, t], [0, 1]]
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = cam_rot_mat
        T_c2w[:3, 3] = cam_pos
        if rotation_z_180:
            rotation_matrix = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T_c2w = T_c2w @ rotation_matrix

        # ---------------------------------------------------------
        # 2. Intrinsics (Camera Matrix K)
        # MuJoCo defines cameras by vertical FOV (fovy) in degrees.
        # We calculate focal length based on the image resolution.
        # ---------------------------------------------------------
        fovy_deg = sim.model.cam_fovy[cam_id]
        fovy_rad = np.deg2rad(fovy_deg)

        # Focal length calculation: f = H / (2 * tan(fovy / 2))
        # Assuming square pixels, fx = fy
        focal_length = (image_height / 2.0) / np.tan(fovy_rad / 2.0)

        # Principal point (cx, cy) - usually the image center
        cx = image_width / 2.0
        cy = image_height / 2.0

        # Intrinsic Matrix K (3x3)
        # K = [[fx,  0, cx],
        #      [ 0, fy, cy],
        #      [ 0,  0,  1]]
        K = np.array([[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]])

        camera_params[cam_name] = {
            "extrinsic_matrix": T_c2w.tolist(),  # Camera-to-World (Pose)
            "intrinsic_matrix": K.tolist(),  # Intrinsic K
            "intrinsics": [focal_length, focal_length, cx, cy],  # fx, fy, cx, cy
            "pos": cam_pos.tolist(),  # Raw Position
            "quat": R.from_matrix(cam_rot_mat).as_quat().tolist(),  # Raw Quaternion (x,y,z,w)
            "fov": fovy_deg,  # Field of View
        }

    return camera_params


def render_gt_with_camera_restoration(
    sim,
    camera_name: str,
    original_cam_pos_dict: dict,
    original_cam_quat_dict: dict,
    perturbed_cam_pos_dict: dict,
    perturbed_cam_quat_dict: dict,
    resolution: int = 256
) -> np.ndarray:
    """
    Render GT RGB image by temporarily restoring camera to unperturbed position.

    Critical: Uses try/finally to ensure perturbation is ALWAYS restored.

    Args:
        sim: MuJoCo sim object
        camera_name: Name of camera to render
        original_cam_pos_dict: Dict mapping camera names to original positions
        original_cam_quat_dict: Dict mapping camera names to original quaternions
        perturbed_cam_pos_dict: Dict mapping camera names to perturbed positions
        perturbed_cam_quat_dict: Dict mapping camera names to perturbed quaternions
        resolution: Image resolution

    Returns:
        RGB image (H, W, 3) numpy array, uint8, flipped to match observation format
    """
    cam_id = sim.model.camera_name2id(camera_name)

    try:
        # 1. Restore to original (unperturbed) position
        sim.model.cam_pos[cam_id] = original_cam_pos_dict[camera_name]
        sim.model.cam_quat[cam_id] = original_cam_quat_dict[camera_name]
        sim.forward()

        # 2. Render GT image
        rgb = sim.render(
            width=resolution,
            height=resolution,
            camera_name=camera_name,
            depth=False
        )

        # 3. Flip to match observation preprocessing (MuJoCo renders upside down)
        rgb_flipped = np.ascontiguousarray(rgb[::-1, ::-1])

    except Exception as e:
        logging.error(f"GT rendering failed for {camera_name}: {e}")
        raise

    finally:
        # CRITICAL: Always restore perturbation, even if rendering failed
        if camera_name in perturbed_cam_pos_dict:
            sim.model.cam_pos[cam_id] = perturbed_cam_pos_dict[camera_name]
        if camera_name in perturbed_cam_quat_dict:
            sim.model.cam_quat[cam_id] = perturbed_cam_quat_dict[camera_name]
        sim.forward()

    return rgb_flipped
