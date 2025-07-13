# file: python/core/stereo_projector.py (Tối ưu hóa)

import numpy as np
import cv2
import os
import logging
import config

# Thiết lập logging
try:
    from utils.logging_python_orangepi import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

calib_file_path = config.OPI_CONFIG.get("calib_file_path", "depth_processing/cali_result")

class StereoProjector:
    """
    Quản lý việc chiếu điểm và bounding box giữa camera RGB và ToF.
    Phiên bản tối ưu: cache map undistort, vector hóa, kiểm tra FOV ngay trong projection.
    """
    def __init__(self,
                 min_depth_mm: int = 450,
                 max_depth_mm: int = 8000,
                 min_valid_pixels: int = 100,
                 max_std_dev: float = 5.0,
                 assumed_depth_mm: float = 2500.0):
        self.params = self._load_calibration(calib_file_path)
        self.mtx_rgb = self.params['mtx_rgb']
        self.dist_rgb = self.params['dist_rgb']
        self.mtx_tof = self.params['mtx_tof']
        self.dist_tof = self.params['dist_tof']
        self.R = self.params['R']
        self.T = self.params['T']

        # Cache các thông số
        self.fx, self.fy = self.mtx_rgb[0,0], self.mtx_rgb[1,1]
        self.cx, self.cy = self.mtx_rgb[0,2], self.mtx_rgb[1,2]
        self.MIN_D, self.MAX_D = min_depth_mm, max_depth_mm
        self.MIN_PIX, self.MAX_STD = min_valid_pixels, max_std_dev
        self.ASSUMED = assumed_depth_mm

        # Tạo bản đồ remap nếu cần remap toàn bộ ảnh
        # self.map1, self.map2 = cv2.initUndistortRectifyMap(
        #     self.mtx_rgb, self.dist_rgb, None, self.mtx_rgb,
        #     (int(self.params['img_width']), int(self.params['img_height'])), cv2.CV_32FC1)

        logger.info("StereoProjector tối ưu đã khởi tạo.")

    def _load_calibration(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file hiệu chỉnh: {path}")
        with np.load(path) as data:
            return {k: data[k] for k in data}

    def _back_project_to_3d(self, pts_undist: np.ndarray, depth: float) -> np.ndarray:
        u,v = pts_undist[:,0], pts_undist[:,1]
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = np.full_like(x, depth)
        return np.stack((x,y,z), axis=-1)

    def _project(self, rgb_pts: np.ndarray, depth: float) -> np.ndarray:
        # Undistort điểm vectorized
        pts_undist = cv2.undistortPoints(rgb_pts, self.mtx_rgb, self.dist_rgb,
                                         P=self.mtx_rgb).reshape(-1,2)
        # 3D và project
        pts3d = self._back_project(pts_undist, depth)
        proj,_ = cv2.projectPoints(pts3d, self.R, self.T, self.mtx_tof, self.dist_tof)
        return proj.reshape(-1,2)

    def get_robust_distance(self, rgb_box: tuple, depth_map: np.ndarray, refine: int =1) -> tuple[float|None,str]:
        x1,y1,x2,y2 = map(int,rgb_box)
        if x2<=x1 or y2<=y1:
            return None, "Invalid RGB box"
        d = self.ASSUMED
        h,w = depth_map.shape
        for i in range(refine+1):
            corners = np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]],np.float32).reshape(-1,1,2)
            try:
                proj = self._project(corners, d)
            except Exception:
                return (d, "OK") if i>0 else (None, "Project fail")
            xs, ys = proj[:,0], proj[:,1]
            xi1, yi1 = max(0,int(xs.min())), max(0,int(ys.min()))
            xi2, yi2 = min(w,int(xs.max())), min(h,int(ys.max()))
            if xi1>=xi2 or yi1>=yi2:
                return None, "ROI empty"
            roi = depth_map[yi1:yi2, xi1:xi2]
            vals = roi[roi>0]
            if vals.size < self.MIN_PIX:
                return None, f"Few points({vals.size})"
            std = vals.std()
            if std < self.MAX_STD:
                return None, f"Flat noise({std:.1f})"
            d = float(np.median(vals))
        if not (self.MIN_D < d < self.MAX_D):
            return None, f"Out range({d:.0f})"
        return d, "OK"

    def project_rgb_box_to_tof(self,
                                 rgb_box: tuple[int,int,int,int],
                                 depth_map: np.ndarray,
                                 precomputed_distance: float | None = None
                               ) -> tuple[int,int,int,int]|None:
        """
        Chiếu và cắt bounding box từ RGB sang ToF;
        nếu đã có precomputed_distance thì không tính lại.
        """
        if precomputed_distance is None:
            d, status = self.get_robust_distance(rgb_box, depth_map)
            if status != "OK":
                return None
        else:
            d = precomputed_distance
        corners = np.array([[rgb_box[0],rgb_box[1]],[rgb_box[0],rgb_box[3]],
                            [rgb_box[2],rgb_box[1]],[rgb_box[2],rgb_box[3]]],
                           np.float32).reshape(-1,1,2)
        proj = self._project(corners, d)
        h,w = depth_map.shape
        xmn, ymn = max(0,int(proj[:,0].min())), max(0,int(proj[:,1].min()))
        xmx, ymx = min(w,int(proj[:,0].max())), min(h,int(proj[:,1].max()))
        if xmn>=xmx or ymn>=ymx:
            return None
        return (xmn, ymn, xmx, ymx)
    
