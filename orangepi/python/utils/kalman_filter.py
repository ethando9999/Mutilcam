# file: utils/kalman_filter.py
import numpy as np

class SimpleKalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_error=1.0, initial_value=None):
        """
        Khởi tạo bộ lọc Kalman 1D đơn giản.
        Args:
            process_variance (float): Độ nhiễu của quá trình (trạng thái thay đổi bao nhiêu giữa các bước).
            measurement_variance (float): Độ nhiễu của phép đo (tin tưởng phép đo mới đến mức nào).
            estimated_error (float): Sai số ước tính ban đầu.
            initial_value (float, optional): Giá trị khởi tạo.
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_error = estimated_error
        self.current_estimate = initial_value
        
    def update(self, measurement):
        """
        Cập nhật bộ lọc với một phép đo mới.
        """
        if self.current_estimate is None:
            self.current_estimate = measurement
            return self.current_estimate

        # Dự đoán
        prediction_error = self.estimated_error + self.process_variance
        
        # Cập nhật
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.current_estimate = self.current_estimate + kalman_gain * (measurement - self.current_estimate)
        self.estimated_error = (1 - kalman_gain) * prediction_error
        
        return self.current_estimate