import numpy as np
import scipy.linalg


# 0.95 quantile chi-square thresholds for gating
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592
}

class KalmanFilterWorldPoint:
    """
    A simple Kalman filter for tracking 2D world points (x, y) on the ground plane.

    The 4-dimensional state space:
        x, y, vx, vy
    uses a constant velocity motion model and direct position observations.
    """
    def __init__(self, dt: float = 1.0,
                 std_weight_position: float = 1.0/20,
                 std_weight_velocity: float = 1.0/160):
        # dimensions
        self.ndim = 2
        self.dt = dt
        # state transition matrix (4x4)
        self._motion_mat = np.eye(2*self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt
        # observation matrix (2x4)
        self._update_mat = np.eye(self.ndim, 2*self.ndim)
        # process/measurement noise weights
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement: np.ndarray):
        """
        Create track from unassociated 2D measurement.
        Parameters
        ----------
        measurement : array_like, shape (2,)
            [x, y] position on the ground plane.
        Returns
        -------
        mean : ndarray, shape (4,)
            Initial state mean [x, y, vx, vy].
        covariance : ndarray, shape (4, 4)
            Initial state covariance.
        """
        mean_pos = measurement.copy()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_velocity
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Run Kalman filter prediction step.
        """
        std_pos = [
            self._std_weight_position,
            self._std_weight_position
        ]
        std_vel = [
            self._std_weight_velocity,
            self._std_weight_velocity
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # state prediction
        mean = np.dot(self._motion_mat, mean)
        # covariance prediction
        covariance = (self._motion_mat @ covariance @ self._motion_mat.T) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Project state distribution to measurement space.
        Returns projected mean and covariance for position.
        """
        std = [
            self._std_weight_position,
            self._std_weight_position
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T
        return projected_mean, projected_cov + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """
        Run Kalman filter correction step with a new 2D measurement.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain
        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        # 2. Chuyển vị kết quả cuối cùng để được K
        b = (covariance @ self._update_mat.T).T
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), b, check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray, only_position: bool = True,
                        metric: str = 'maha') -> np.ndarray:
        """
        Compute gating (Mahalanobis) distance between state and measurements.
        Parameters
        ----------
        measurements : array_like, shape (N, 2)
        Returns
        -------
        distances : ndarray, shape (N,)
        """
        # project to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)
        d = measurements - projected_mean

        if metric == 'gaussian':
            return np.sum(d**2, axis=1)
        elif metric == 'maha':
            # cholesky of cov
            ch = np.linalg.cholesky(projected_cov) 
            z = scipy.linalg.solve_triangular(ch, d.T, lower=True)
            return np.sum(z*z, axis=0)
        else:
            raise ValueError(f"Unknown metric {metric}")

    def gating_threshold(self, only_position: bool = True) -> float:
        """
        Return chi-square gating threshold at 0.95 quantile.
        """
        dof = 2 if only_position else 4
        return chi2inv95[dof]
