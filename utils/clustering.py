import numpy as np
import logging
from sklearn.cluster import KMeans, DBSCAN

logger = logging.getLogger(__name__)

class BaseCluster:
    """Lớp cơ sở cho các phương pháp phân cụm"""
    
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        
    def fit(self, coords):
        """Huấn luyện mô hình phân cụm"""
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")
        
    def predict(self, coords):
        """Dự đoán cụm cho tọa độ mới"""
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")

class GridCluster(BaseCluster):
    """Phân cụm dựa trên lưới tọa độ"""
    
    def __init__(self, n_clusters, random_state=42):
        super().__init__(n_clusters, random_state)
        self.grid_size = None
        self.x_step = None
        self.y_step = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        
    def fit(self, coords):
        """Huấn luyện mô hình phân cụm dựa trên lưới"""
        try:
            # Kiểm tra phương sai của tọa độ
            x_var = np.var(coords[:, 0])
            y_var = np.var(coords[:, 1])
            
            # Nếu tất cả tọa độ gần như giống nhau
            if x_var < 0.1 and y_var < 0.1:
                logger.warning("Tất cả tọa độ gần như giống nhau. Tạo cụm ngẫu nhiên.")
                return self._create_random_clusters(coords)
            
            # Tính toán kích thước lưới
            self.x_min, self.x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
            self.y_min, self.y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
            
            self.grid_size = int(np.ceil(np.sqrt(self.n_clusters)))
            self.x_step = (self.x_max - self.x_min) / self.grid_size if self.x_max > self.x_min else 1
            self.y_step = (self.y_max - self.y_min) / self.grid_size if self.y_max > self.y_min else 1
            
            # Phân cụm theo lưới
            n_samples = len(coords)
            labels = np.zeros(n_samples, dtype=int)
            
            for i in range(n_samples):
                x, y = coords[i]
                x_grid = min(self.grid_size-1, int((x - self.x_min) / self.x_step)) if self.x_step > 0 else 0
                y_grid = min(self.grid_size-1, int((y - self.y_min) / self.y_step)) if self.y_step > 0 else 0
                label = x_grid + y_grid * self.grid_size
                labels[i] = min(label, self.n_clusters-1)
            
            # Đảm bảo nhãn là các số nguyên liên tiếp
            unique_labels = np.unique(labels)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            self.labels_ = np.array([label_map[label] for label in labels])
            
            # Tính tâm cụm
            self.cluster_centers_ = np.zeros((len(unique_labels), 2))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                self.cluster_centers_[i] = np.mean(coords[mask], axis=0)
            
            return self
            
        except Exception as e:
            logger.error(f"Lỗi khi phân cụm theo lưới: {str(e)}")
            return self
            
    def predict(self, coords):
        """Dự đoán cụm cho tọa độ mới"""
        if self.grid_size is None:
            raise ValueError("Mô hình chưa được huấn luyện")
            
        n_samples = len(coords)
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            x, y = coords[i]
            x_grid = min(self.grid_size-1, int((x - self.x_min) / self.x_step)) if self.x_step > 0 else 0
            y_grid = min(self.grid_size-1, int((y - self.y_min) / self.y_step)) if self.y_step > 0 else 0
            label = x_grid + y_grid * self.grid_size
            labels[i] = min(label, self.n_clusters-1)
            
        return labels
        
    def _create_random_clusters(self, coords):
        """Tạo cụm ngẫu nhiên khi tọa độ gần như giống nhau"""
        np.random.seed(self.random_state)
        n_samples = len(coords)
        self.labels_ = np.random.randint(0, self.n_clusters, size=n_samples)
        
        # Tạo tâm cụm phân bố đều quanh vòng tròn
        self.cluster_centers_ = np.zeros((self.n_clusters, 2))
        radius = 100
        for i in range(self.n_clusters):
            angle = 2 * np.pi * i / self.n_clusters
            self.cluster_centers_[i, 0] = 540 + radius * np.cos(angle)
            self.cluster_centers_[i, 1] = 60 + radius * np.sin(angle)
            
        return self

class KMeansCluster(BaseCluster):
    """Phân cụm sử dụng K-means"""
    
    def __init__(self, n_clusters, random_state=42):
        super().__init__(n_clusters, random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        
    def fit(self, coords):
        """Huấn luyện mô hình K-means"""
        try:
            self.kmeans.fit(coords)
            self.labels_ = self.kmeans.labels_
            self.cluster_centers_ = self.kmeans.cluster_centers_
            return self
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện K-means: {str(e)}")
            return self
            
    def predict(self, coords):
        """Dự đoán cụm cho tọa độ mới"""
        return self.kmeans.predict(coords) 

class DBSCANCluster(BaseCluster):
    """Phân cụm sử dụng DBSCAN"""
    
    def __init__(self, eps=0.5, min_samples=5, random_state=42):
        """
        Khởi tạo DBSCAN
        Args:
            eps (float): Khoảng cách tối đa giữa hai mẫu để được coi là lân cận
            min_samples (int): Số lượng mẫu tối thiểu trong một lân cận để tạo thành một cụm
            random_state (int): Seed cho random state
        """
        super().__init__(n_clusters=None, random_state=random_state)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
    def fit(self, coords):
        """Huấn luyện mô hình DBSCAN"""
        try:
            self.dbscan.fit(coords)
            self.labels_ = self.dbscan.labels_
            
            # Tính tâm cụm cho các cụm hợp lệ (không phải nhiễu)
            unique_labels = np.unique(self.labels_)
            valid_labels = unique_labels[unique_labels != -1]  # Loại bỏ nhãn nhiễu (-1)
            
            if len(valid_labels) > 0:
                self.cluster_centers_ = np.zeros((len(valid_labels), coords.shape[1]))
                for i, label in enumerate(valid_labels):
                    mask = self.labels_ == label
                    self.cluster_centers_[i] = np.mean(coords[mask], axis=0)
            else:
                self.cluster_centers_ = np.array([])
                
            logger.info(f"DBSCAN tìm thấy {len(valid_labels)} cụm và {np.sum(self.labels_ == -1)} điểm nhiễu")
            return self
            
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện DBSCAN: {str(e)}")
            return self
            
    def predict(self, coords):
        """Dự đoán cụm cho tọa độ mới"""
        return self.dbscan.fit_predict(coords)
        
    def get_cluster_info(self):
        """Lấy thông tin về các cụm"""
        if self.labels_ is None:
            return None
            
        unique_labels = np.unique(self.labels_)
        info = {
            'n_clusters': len(unique_labels[unique_labels != -1]),
            'n_noise': np.sum(self.labels_ == -1),
            'cluster_sizes': {}
        }
        
        for label in unique_labels:
            if label != -1:
                info['cluster_sizes'][label] = np.sum(self.labels_ == label)
                
        return info 