"""
Mathematical Utilities Module.

Provides common mathematical functions used throughout the quadruped
control stack including Bezier curves, rotations, and transforms.
"""

import numpy as np
from typing import Tuple, Optional


def cubic_bezier(t: float) -> float:
    """
    Cubic Bezier interpolation function.

    Returns a smooth interpolation using cubic Bezier curve
    with zero velocity at endpoints.

    Args:
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated value [0, 1]
    """
    t = np.clip(t, 0.0, 1.0)
    return 3.0 * t * t * (1.0 - t)


def smooth_step(t: float) -> float:
    """
    Smooth step function with continuous first derivative.

    Uses the smoothstep polynomial: 3t² - 2t³

    Args:
        t: Input parameter [0, 1]

    Returns:
        Smoothed value [0, 1]
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smc(
    x: np.ndarray,
    s: float = 0.5,
    phi: float = 0.01,
) -> np.ndarray:
    """
    Super-twisting sliding mode control function.

    Provides robust control with finite-time convergence
    and chattering reduction.

    Args:
        x: Sliding surface value
        s: Sliding surface gain
        phi: Boundary layer thickness

    Returns:
        Control output
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0

    x = np.atleast_1d(x)
    result = np.zeros_like(x)

    for i in range(len(x)):
        xi = x[i]
        if abs(xi) <= phi:
            result[i] = s * xi * xi * np.sign(xi) / (2.0 * phi)
        else:
            result[i] = s * np.sign(xi)

    return result.item() if scalar else result


def rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """
    Create rotation matrix from roll-pitch-yaw angles.

    Args:
        rpy: Roll, pitch, yaw angles [3]

    Returns:
        Rotation matrix [3, 3]
    """
    roll, pitch, yaw = rpy

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [    -sp,             cp * sr,             cp * cr]
    ])

    return R


def quaternion_to_rpy(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to roll-pitch-yaw angles.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        RPY angles [roll, pitch, yaw]
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """
    Convert roll-pitch-yaw angles to quaternion.

    Args:
        rpy: RPY angles [roll, pitch, yaw]

    Returns:
        Quaternion [w, x, y, z]
    """
    roll, pitch, yaw = rpy

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def hat(v: np.ndarray) -> np.ndarray:
    """
    Hat operator: maps 3-vector to skew-symmetric matrix.

    For a vector v = [v1, v2, v3]:
    hat(v) = [[0, -v3, v2],
              [v3, 0, -v1],
              [-v2, v1, 0]]

    Args:
        v: Input vector [3]

    Returns:
        Skew-symmetric matrix [3, 3]
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def vee(S: np.ndarray) -> np.ndarray:
    """
    Vee operator: inverse of hat operator.

    Maps skew-symmetric matrix to 3-vector.

    Args:
        S: Skew-symmetric matrix [3, 3]

    Returns:
        Vector [3]
    """
    return np.array([-S[1, 2], S[0, 2], -S[0, 1]])


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3).

    Computes rotation matrix from axis-angle using Rodrigues' formula.

    Args:
        omega: Rotation vector (axis * angle) [3]

    Returns:
        Rotation matrix [3, 3]
    """
    theta = np.linalg.norm(omega)

    if theta < 1e-8:
        return np.eye(3)

    axis = omega / theta
    K = hat(axis)

    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def log_so3(R: np.ndarray) -> np.ndarray:
    """
    Logarithmic map from SO(3) to so(3).

    Computes rotation vector from rotation matrix.

    Args:
        R: Rotation matrix [3, 3]

    Returns:
        Rotation vector [3]
    """
    trace = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)

    if theta < 1e-8:
        return np.zeros(3)

    # Axis from skew-symmetric part
    axis = vee(R - R.T) / (2.0 * np.sin(theta))

    return theta * axis


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.

    Args:
        q1, q2: Quaternions [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Conjugate [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion inverse.

    Args:
        q: Unit quaternion [w, x, y, z]

    Returns:
        Inverse quaternion [w, -x, -y, -z]
    """
    return quaternion_conjugate(q)


def transform_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Transform vector by quaternion.

    Args:
        v: Vector to transform [3]
        q: Quaternion [w, x, y, z]

    Returns:
        Transformed vector [3]
    """
    qv = np.array([0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    result = quaternion_multiply(quaternion_multiply(q, qv), q_conj)
    return result[1:]


def low_pass_filter(
    current: np.ndarray,
    previous: np.ndarray,
    cutoff_freq: float,
    dt: float,
) -> np.ndarray:
    """
    Apply first-order low-pass filter.

    Args:
        current: Current value
        previous: Previous filtered value
        cutoff_freq: Cutoff frequency [Hz]
        dt: Time step [s]

    Returns:
        Filtered value
    """
    rc = 1.0 / (2.0 * np.pi * cutoff_freq)
    alpha = dt / (rc + dt)
    return alpha * current + (1.0 - alpha) * previous


def finite_difference(
    current: np.ndarray,
    previous: np.ndarray,
    dt: float,
    order: int = 1,
) -> np.ndarray:
    """
    Compute finite difference derivative.

    Args:
        current: Current value
        previous: Previous value
        dt: Time step [s]
        order: Derivative order (1 or 2)

    Returns:
        Derivative estimate
    """
    if order == 1:
        return (current - previous) / dt
    elif order == 2:
        # Need previous and pre-previous for second order
        raise NotImplementedError("Second-order finite difference needs two previous values")
    else:
        raise ValueError(f"Order {order} not supported")


def saturate(x: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    """
    Apply saturation limits to array.

    Args:
        x: Input array
        limits: (min, max) limits

    Returns:
        Saturated array
    """
    return np.clip(x, limits[0], limits[1])


def dead_zone(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply dead zone to input.

    Args:
        x: Input array
        threshold: Dead zone threshold

    Returns:
        Dead-zoned array
    """
    result = np.where(np.abs(x) < threshold, 0.0, x)
    result = np.where(x >= threshold, x - threshold, result)
    result = np.where(x <= -threshold, x + threshold, result)
    return result


def polyfit_matrix(x: np.ndarray, order: int) -> np.ndarray:
    """
    Build polynomial fitting matrix.

    Args:
        x: Input points [n]
        order: Polynomial order

    Returns:
        Design matrix [n, order+1]
    """
    return np.vander(x, order + 1)


def weighted_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    W: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve weighted least squares problem.

    Args:
        A: Design matrix [m, n]
        b: Target vector [m]
        W: Weight matrix [m, m] (identity if None)

    Returns:
        Solution vector [n]
    """
    if W is None:
        return np.linalg.lstsq(A, b, rcond=None)[0]

    # Convert to standard LS: sqrt(W) A x = sqrt(W) b
    sqrt_W = np.sqrt(W)
    A_weighted = sqrt_W @ A
    b_weighted = sqrt_W @ b

    return np.linalg.lstsq(A_weighted, b_weighted, rcond=None)[0]


def pinv_svd(A: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """
    Compute Moore-Penrose pseudoinverse via SVD.

    Args:
        A: Input matrix [m, n]
        rcond: Relative condition number cutoff

    Returns:
        Pseudoinverse [n, m]
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    cutoff = rcond * np.max(s)
    s_inv = np.where(s > cutoff, 1.0 / s, 0.0)
    return Vh.T @ np.diag(s_inv) @ U.T


def weighted_pinv(
    A: np.ndarray,
    W: np.ndarray,
    rcond: float = 1e-6,
) -> np.ndarray:
    """
    Compute weighted pseudoinverse.

    Solves: min ||W(Ax - b)||² for weighted least squares

    Args:
        A: Input matrix [m, n]
        W: Weight matrix [m, m] (symmetric positive definite)
        rcond: Condition number cutoff

    Returns:
        Weighted pseudoinverse [n, m]
    """
    sqrt_W = np.sqrt(W)
    A_weighted = sqrt_W @ A
    A_pinv_weighted = pinv_svd(A_weighted, rcond)
    return A_pinv_weighted @ sqrt_W


def block_diag(*matrices: np.ndarray) -> np.ndarray:
    """
    Build block diagonal matrix.

    Args:
        matrices: Arrays to arrange diagonally

    Returns:
        Block diagonal matrix
    """
    result = np.zeros((sum(A.shape[0] for A in matrices),
                       sum(A.shape[1] for A in matrices)))
    row, col = 0, 0
    for A in matrices:
        r, c = A.shape
        result[row:row + r, col:col + c] = A
        row += r
        col += c
    return result


def regularize(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Add diagonal regularization to matrix.

    Args:
        matrix: Square matrix to regularize
        epsilon: Regularization amount

    Returns:
        Regularized matrix
    """
    return matrix + epsilon * np.eye(matrix.shape[0])
