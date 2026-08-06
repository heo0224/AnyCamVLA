"""
Utility functions for spherical coordinate-based camera perturbations
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        radius: Distance from origin
        theta: Azimuth angle in degrees (rotation around z-axis, 0 = +x axis, increases counterclockwise)
        phi: Elevation angle in degrees (angle from xy-plane, positive = above)
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arctan2(y, x))  # Azimuth
    phi = np.degrees(np.arcsin(z / radius)) if radius > 0 else 0.0  # Elevation
    return radius, theta, phi


def spherical_to_cartesian(radius, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        radius: Distance from origin
        theta: Azimuth angle in degrees (rotation around z-axis)
        phi: Elevation angle in degrees (angle from xy-plane)

    Returns:
        x, y, z: Cartesian coordinates
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    x = radius * np.cos(phi_rad) * np.cos(theta_rad)
    y = radius * np.cos(phi_rad) * np.sin(theta_rad)
    z = radius * np.sin(phi_rad)

    return x, y, z


def compute_ray_plane_intersection(ray_origin, ray_direction, plane_normal, plane_point):
    """
    Compute intersection of a ray with a plane.

    Args:
        ray_origin: Origin of the ray (x, y, z)
        ray_direction: Direction of the ray (x, y, z), should be normalized
        plane_normal: Normal vector of the plane (x, y, z)
        plane_point: A point on the plane (x, y, z)

    Returns:
        intersection_point: (x, y, z) or None if ray is parallel to plane
    """
    # Compute denominator
    denom = np.dot(ray_direction, plane_normal)

    # Check if ray is parallel to plane
    if abs(denom) < 1e-6:
        return None

    # Compute distance along ray to intersection
    t = np.dot(plane_point - ray_origin, plane_normal) / denom

    # Check if intersection is in front of ray origin
    if t < 0:
        return None

    # Compute intersection point
    intersection = ray_origin + t * ray_direction
    return intersection


def get_camera_look_at_point_on_table(camera_pos, camera_quat_mujoco, table_height):
    """
    Get the point where the camera's center ray intersects the table plane.

    Args:
        camera_pos: Camera position (x, y, z)
        camera_quat_mujoco: Camera quaternion in MuJoCo format (w, x, y, z)
        table_height: Height of the table (z coordinate)

    Returns:
        intersection_point: (x, y, z) on the table plane
    """
    # Convert quaternion to rotation matrix
    quat_scipy = camera_quat_mujoco[[1, 2, 3, 0]]  # w,x,y,z -> x,y,z,w
    r = R.from_quat(quat_scipy)
    rot_mat = r.as_matrix()

    # Extract forward direction (camera looks along -Z in local frame)
    # Rotation matrix columns are [right, up, -forward]
    forward = -rot_mat[:, 2]

    # Define table plane
    plane_normal = np.array([0, 0, 1])  # Table is horizontal
    plane_point = np.array([0, 0, table_height])

    # Compute intersection
    intersection = compute_ray_plane_intersection(
        camera_pos, forward, plane_normal, plane_point
    )

    if intersection is None:
        # Camera is looking parallel to table or away from it
        # Fallback: project camera position onto table
        intersection = np.array([camera_pos[0], camera_pos[1], table_height])

    return intersection


def compute_lookat_rotation(camera_pos, target_pos, up_vector=np.array([0, 0, 1])):
    """
    Compute rotation matrix for a camera looking at a target point.

    Args:
        camera_pos: Camera position (x, y, z)
        target_pos: Target look-at position (x, y, z)
        up_vector: Up direction (default: z-axis)

    Returns:
        Rotation matrix (3x3) and quaternion (x, y, z, w) in scipy format
    """
    # Compute forward direction (from camera to target)
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    # Compute right direction (perpendicular to forward and up)
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)

    # Recompute up direction (perpendicular to forward and right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix
    # OpenCV/computer vision convention: camera looks along +Z axis
    # MuJoCo convention: camera looks along -Z axis
    # We use MuJoCo convention here
    # Rotation matrix columns are [right, up, -forward]
    rot_matrix = np.column_stack([right, up, -forward])

    # Convert to quaternion (scipy format: x, y, z, w)
    rotation = R.from_matrix(rot_matrix)
    quat_scipy = rotation.as_quat()

    return rot_matrix, quat_scipy


def apply_spherical_perturbation(original_cam_pos, original_cam_quat, reference_point, delta_radius, delta_theta, delta_phi):
    """
    Apply spherical coordinate perturbation to camera position while maintaining look-at toward reference point.

    Args:
        original_cam_pos: Original camera position (x, y, z)
        original_cam_quat: Original camera quaternion in MuJoCo format (w, x, y, z)
        reference_point: Reference point (x, y, z) - the origin of spherical coordinates
        delta_radius: Change in distance (meters)
        delta_theta: Change in azimuth angle (degrees)
        delta_phi: Change in elevation angle (degrees)

    Returns:
        new_cam_pos: New camera position (x, y, z)
        new_cam_quat_mujoco: New camera orientation as quaternion in MuJoCo format (w, x, y, z)
    """
    # Convert to numpy arrays
    original_cam_pos = np.array(original_cam_pos)
    original_cam_quat = np.array(original_cam_quat)
    reference_point = np.array(reference_point)

    # Compute relative position (camera relative to reference point)
    relative_pos = original_cam_pos - reference_point

    # Convert to spherical coordinates
    radius, theta, phi = cartesian_to_spherical(relative_pos[0], relative_pos[1], relative_pos[2])

    # Apply perturbations
    new_radius = radius + delta_radius
    new_theta = theta + delta_theta
    new_phi = phi + delta_phi

    # Clamp radius to positive values
    new_radius = max(new_radius, 0.1)

    # Clamp phi to avoid singularities at poles
    new_phi = np.clip(new_phi, -89.0, 89.0)

    # Convert back to Cartesian
    new_relative_x, new_relative_y, new_relative_z = spherical_to_cartesian(new_radius, new_theta, new_phi)

    # Compute new camera position in world frame
    new_cam_pos = reference_point + np.array([new_relative_x, new_relative_y, new_relative_z])

    # Compute new camera orientation (looking at reference point)
    rot_matrix, quat_scipy = compute_lookat_rotation(new_cam_pos, reference_point)

    # Convert quaternion from scipy format (x, y, z, w) to MuJoCo format (w, x, y, z)
    quat_mujoco = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

    return new_cam_pos, quat_mujoco


def test_spherical_perturbation():
    """Test function to verify spherical perturbation logic"""
    print("Testing Spherical Camera Perturbation (Reference Point-based)")
    print("=" * 60)

    # Test case: Camera looking down at workspace
    original_pos = np.array([0.5, 0.3, 1.2])

    # Reference point: end effector (x, y) with table height z
    eef_x, eef_y = -0.037, -0.005
    table_height = 0.0
    reference_point = np.array([eef_x, eef_y, table_height])

    # Create a camera quaternion looking at reference point
    _, quat_scipy = compute_lookat_rotation(original_pos, reference_point)
    original_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])  # MuJoCo format

    print(f"Original camera position: {original_pos}")
    print(f"Original camera quaternion (w,x,y,z): {original_quat}")
    print(f"Reference point (eef_x, eef_y, table_z): {reference_point}")
    print(f"Original relative position: {original_pos - reference_point}")

    # Test different perturbations
    test_cases = [
        (0.0, 0.0, 0.0, "No change"),
        (0.1, 0.0, 0.0, "Move 0.1m further from reference"),
        (-0.1, 0.0, 0.0, "Move 0.1m closer to reference"),
        (0.0, 30.0, 0.0, "Rotate 30° left (azimuth)"),
        (0.0, -30.0, 0.0, "Rotate 30° right (azimuth)"),
        (0.0, 0.0, 15.0, "Rotate 15° higher (elevation)"),
        (0.0, 0.0, -15.0, "Rotate 15° lower (elevation)"),
    ]

    for delta_r, delta_theta, delta_phi, description in test_cases:
        new_pos, new_quat = apply_spherical_perturbation(
            original_pos, original_quat, reference_point, delta_r, delta_theta, delta_phi
        )

        print(f"\n{description}:")
        print(f"  Δradius={delta_r:.2f}m, Δθ={delta_theta:.1f}°, Δφ={delta_phi:.1f}°")
        print(f"  New position: [{new_pos[0]:.4f}, {new_pos[1]:.4f}, {new_pos[2]:.4f}]")
        print(f"  New quaternion (w,x,y,z): [{new_quat[0]:.4f}, {new_quat[1]:.4f}, {new_quat[2]:.4f}, {new_quat[3]:.4f}]")

        # Compute where new camera is looking
        forward = -R.from_quat(new_quat[[1, 2, 3, 0]]).as_matrix()[:, 2]
        look_at_dir = reference_point - new_pos
        look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
        dot_product = np.dot(forward, look_at_dir)
        print(f"  Camera forward dot reference direction: {dot_product:.6f} (should be ~1.0)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_spherical_perturbation()
