import os
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.decomposition import KernelPCA
from .models.model_factory import create_classifier, create_predictor
from .cluster_utils import find_optimal_clusters
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import N_COMPONENTS

logger = logging.getLogger(__name__)

class ClusterRegression:
    """Mô hình hồi quy dựa trên phân cụm tọa độ để dự đoán vị trí"""
    
    def __init__(self, max_clusters=5, classifier_type='knn', global_predictor_type='knn',
                 cluster_predictor_type='knn', n_neighbors=5, auto_tune=False, 
                 random_state=42, visualize=False, use_kernel_pca=True):
        """Khởi tạo mô hình"""
        # Tham số mô hình
        self.max_clusters = max_clusters
        self.classifier_type = classifier_type
        self.global_predictor_type = global_predictor_type
        self.cluster_predictor_type = cluster_predictor_type
        self.n_neighbors = n_neighbors
        self.auto_tune = auto_tune
        self.random_state = random_state
        self.visualize = visualize
        self.use_kernel_pca = use_kernel_pca
        self.kernel = 'cosine'  # Thay đổi kernel thành cosine
        self.n_components = N_COMPONENTS
        
        # Các mô hình thành phần
        self.trained = False
        self.coords_kmeans = None
        self.classifier = None
        self.global_predictor = None
        self.cluster_predictors = {}
        self.cluster_coords = {}
        self.kernel_pca = None
        self.valid_clusters = []
        
    def reshape_csi_data(self, X):
        """Reshape dữ liệu CSI 3D thành 2D"""
        if X is None: return None
        if len(X.shape) == 3:
            samples, channels, features = X.shape
            return X.reshape(samples, channels * features)
        return X
    
    def fit(self, X, y=None, coords=None):
        """Huấn luyện mô hình"""
        try:
            # Kiểm tra dữ liệu
            if X is None:
                logger.error("Không có dữ liệu đầu vào")
                return self
            if coords is None:
                coords = y if y is not None else None
            if coords is None:
                logger.error("Không có tọa độ cho huấn luyện")
                return self
            
            # Chuyển đổi DataFrame thành numpy array nếu cần
            if isinstance(X, pd.DataFrame):
                X = X.values
            elif isinstance(X, list):
                X = np.array(X)
                
            if isinstance(coords, pd.DataFrame):
                coords = coords.values
            elif isinstance(coords, list):
                coords = np.array(coords)
            
            # Đảm bảo X và coords là numpy arrays
            X = np.array(X)
            coords = np.array(coords)
            
            # Đảm bảo chỉ lấy tọa độ 2D (x, y)
            if coords.shape[1] > 2:
                logger.warning(f"Tọa độ có {coords.shape[1]} chiều. Chỉ sử dụng 2 chiều đầu tiên (x, y).")
                coords = coords[:, :2]
            
            # Xử lý đặc biệt cho coords có định dạng chuỗi 'AxB'
            coords_numeric = np.zeros((coords.shape[0], 2), dtype=float)
            for i in range(coords.shape[0]):
                for j in range(min(coords.shape[1], 2)):  # Đảm bảo chỉ xử lý tối đa 2 chiều
                    val = coords[i, j]
                    if isinstance(val, (int, float)):
                        coords_numeric[i, j] = val
                    elif isinstance(val, str) and 'x' in val:
                        # Xử lý chuỗi định dạng "AxB"
                        try:
                            # Lấy số đầu tiên trước dấu 'x'
                            coords_numeric[i, j] = float(val.split('x')[0])
                            logger.debug(f"Đã chuyển đổi tọa độ '{val}' thành {coords_numeric[i, j]}")
                        except (ValueError, IndexError):
                            logger.warning(f"Không thể chuyển đổi tọa độ '{val}', gán giá trị 0")
                            coords_numeric[i, j] = 0.0
                    else:
                        try:
                            coords_numeric[i, j] = float(val)
                        except (ValueError, TypeError):
                            logger.warning(f"Không thể chuyển đổi tọa độ '{val}', gán giá trị 0")
                            coords_numeric[i, j] = 0.0
            
            # Log thông tin
            logger.info(f"Dữ liệu huấn luyện: X shape={X.shape}, coords shape={coords_numeric.shape}")
            logger.info(f"Đã chuyển đổi tọa độ thành dạng số 2D (x, y)")
            
            # Xử lý dữ liệu và giảm chiều với kernel cosine
            X_reshaped = self.reshape_csi_data(X)
            
            # Khởi tạo và áp dụng KernelPCA với số components cố định từ config
            logger.info(f"Sử dụng KernelPCA với kernel='cosine', n_components={self.n_components}")
            self.kernel_pca = KernelPCA(n_components=self.n_components, kernel='cosine')
            X_pca = self.kernel_pca.fit_transform(X_reshaped)
            logger.info(f"Đã giảm chiều dữ liệu CSI từ {X_reshaped.shape[1]} xuống {X_pca.shape[1]} chiều")
            
            # Huấn luyện bộ dự đoán toàn cục trước
            logger.info("Huấn luyện bộ dự đoán toàn cục")
            self.global_predictor = create_predictor(self.global_predictor_type, n_neighbors=5, random_state=self.random_state)
            self.global_predictor.train(X_pca, np.zeros(len(X_pca)), coords_numeric)
            
            # Phân cụm tọa độ với số cụm cố định thay vì tìm tối ưu
            # Số cụm cần đủ nhỏ để đảm bảo mỗi cụm có đủ dữ liệu
            n_samples = len(coords_numeric)
            # Tính số cụm dựa vào số lượng dữ liệu, đảm bảo mỗi cụm có ít nhất 5 mẫu
            n_clusters = min(self.max_clusters, max(2, n_samples // 10))
            logger.info(f"Sử dụng {n_clusters} cụm cho {n_samples} mẫu dữ liệu")
            
            # THAY ĐỔI: Thay thế KMeans bằng phân vùng Grid để tránh vấn đề tất cả mẫu vào một cụm
            # Phân chia tọa độ theo lưới hình chữ nhật thay vì dùng KMeans
            
            # Kiểm tra phương sai của tọa độ thay vì so sánh chính xác
            x_var = np.var(coords_numeric[:, 0])
            y_var = np.var(coords_numeric[:, 1])
            x_min, x_max = np.min(coords_numeric[:, 0]), np.max(coords_numeric[:, 0])
            y_min, y_max = np.min(coords_numeric[:, 1]), np.max(coords_numeric[:, 1])
            
            logger.info(f"Phạm vi tọa độ x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")
            logger.info(f"Phương sai tọa độ: x_var={x_var:.2f}, y_var={y_var:.2f}")
            
            # Kiểm tra tọa độ đầu và cuối
            logger.info(f"Mẫu tọa độ đầu: {coords_numeric[:5]}")
            logger.info(f"Mẫu tọa độ cuối: {coords_numeric[-5:]}")
            
            # Sử dụng ngưỡng phương sai nhỏ để xác định tọa độ giống nhau
            if x_var < 0.1 and y_var < 0.1:
                logger.warning("Tất cả tọa độ gần như giống nhau (phương sai < 0.1). Tạo cụm ngẫu nhiên.")
                # Nếu tất cả tọa độ giống nhau, chia ngẫu nhiên
                np.random.seed(self.random_state)
                coord_labels = np.random.randint(0, n_clusters, size=n_samples)
                
                # Tạo tọa độ ngẫu nhiên cho từng cụm
                for i in range(n_clusters):
                    # Tìm các mẫu thuộc cụm i
                    cluster_indices = np.where(coord_labels == i)[0]
                    if len(cluster_indices) > 0:
                        # Tạo tọa độ ngẫu nhiên cho cụm này
                        angle = 2 * np.pi * i / n_clusters  # Phân bố đều quanh vòng tròn
                        radius = 100  # Bán kính
                        # Tạo tọa độ cụm khác nhau dựa trên vị trí trên vòng tròn
                        center_x = 540 + radius * np.cos(angle)
                        center_y = 60 + radius * np.sin(angle)
                        
                        # Biến đổi tọa độ của cụm để phân biệt với các cụm khác
                        for idx in cluster_indices:
                            # Thêm nhiễu nhỏ để mỗi điểm khác nhau một chút
                            noise_x = np.random.uniform(-10, 10)
                            noise_y = np.random.uniform(-10, 10)
                            coords_numeric[idx, 0] = center_x + noise_x
                            coords_numeric[idx, 1] = center_y + noise_y
                
                logger.info("Đã tạo tọa độ ngẫu nhiên cho mỗi cụm để phân biệt chúng")
            else:
                # Tính toán số lưới theo mỗi chiều (căn bậc hai của số cụm)
                grid_size = int(np.ceil(np.sqrt(n_clusters)))
                
                # Tính kích thước mỗi ô lưới
                x_step = (x_max - x_min) / grid_size if x_max > x_min else 1
                y_step = (y_max - y_min) / grid_size if y_max > y_min else 1
                
                # Khởi tạo nhãn
                coord_labels = np.zeros(n_samples, dtype=int)
                
                # Phân cụm theo lưới
                for i in range(n_samples):
                    x, y = coords_numeric[i]
                    # Tính vị trí lưới (bị chặn ở grid_size-1 để tránh vượt quá)
                    x_grid = min(grid_size-1, int((x - x_min) / x_step)) if x_step > 0 else 0
                    y_grid = min(grid_size-1, int((y - y_min) / y_step)) if y_step > 0 else 0
                    # Chuyển vị trí lưới 2D thành nhãn 1D
                    label = x_grid + y_grid * grid_size
                    # Đảm bảo nhãn không vượt quá n_clusters
                    coord_labels[i] = min(label, n_clusters-1)
                
                # Đảm bảo nhãn là các số nguyên liên tiếp từ 0 đến n_clusters-1
                unique_labels = np.unique(coord_labels)
                label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                
                # Áp dụng ánh xạ nhãn
                for i in range(n_samples):
                    coord_labels[i] = label_map[coord_labels[i]]
                    
                logger.info("Đã phân cụm dựa trên lưới tọa độ")
            
            # Cập nhật số cụm thực tế
            self.n_clusters = len(np.unique(coord_labels))
            logger.info(f"Số cụm thực tế sau khi phân chia: {self.n_clusters}")
            
            # Hiển thị thông tin về sự phân bố của các cụm
            for i in range(self.n_clusters):
                cluster_count = np.sum(coord_labels == i)
                cluster_percentage = cluster_count / n_samples * 100
                logger.info(f"Cụm {i}: {cluster_count} mẫu ({cluster_percentage:.1f}%)")
            
            # Huấn luyện bộ phân loại cho cụm tọa độ
            logger.info("Huấn luyện bộ phân loại cụm tọa độ")
            self.classifier = create_classifier(self.classifier_type, n_neighbors=5, random_state=self.random_state)
            self.classifier.fit(X_pca, coord_labels)  # Phân loại dữ liệu PCA vào các cụm tọa độ
            
            # Huấn luyện bộ dự đoán cho từng cụm tọa độ
            logger.info(f"Huấn luyện bộ dự đoán cho {n_clusters} cụm tọa độ")
            for i in range(n_clusters):
                idx = np.where(coord_labels == i)[0]
                if len(idx) < 5: 
                    logger.warning(f"Cụm {i} chỉ có {len(idx)} mẫu, không đủ để huấn luyện")
                    continue
                
                X_cluster = X_pca[idx]
                coords_cluster = coords_numeric[idx]
                
                # Lưu thông tin về tọa độ cụm
                self.cluster_coords[i] = {
                    'mean': np.mean(coords_cluster, axis=0),
                    'std': np.std(coords_cluster, axis=0),
                    'count': len(coords_cluster)
                }
                logger.info(f"Cụm {i}: {len(idx)} mẫu, tọa độ trung bình: {self.cluster_coords[i]['mean']}")
                
                # Tạo bộ dự đoán cho cụm
                predictor = create_predictor(self.cluster_predictor_type, n_neighbors=5, random_state=self.random_state)
                predictor.train(X_cluster, np.zeros(len(X_cluster)), coords_cluster)
                self.cluster_predictors[i] = predictor
            
            # Cập nhật trạng thái mô hình
            self.valid_clusters = list(self.cluster_predictors.keys())
            self.trained = True
            logger.info(f"Hoàn tất huấn luyện với {len(self.valid_clusters)}/{n_clusters} cụm hợp lệ")
            
            return self
            
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện: {str(e)}")
            return self
            
    def predict(self, X, return_details=False):
        """Dự đoán tọa độ theo quy trình giống hình test phase"""
        try:
            if not self.trained:
                logger.error("Mô hình chưa được huấn luyện")
                return None
            
            # Xử lý dữ liệu
            if isinstance(X, pd.DataFrame):
                X = X.values
            elif isinstance(X, list):
                X = np.array(X)
            elif isinstance(X, dict):
                X = np.array(list(X.values()))
            
            # Đảm bảo X là numpy array
            X = np.array(X)
            
            X_reshaped = self.reshape_csi_data(X)
            X_pca = self.kernel_pca.transform(X_reshaped)
            
            # BƯỚC 1: Sử dụng Classifier để lấy nhãn cục bộ (C_local)
            local_labels = self.classifier.predict(X_pca)  # C_local
            
            # BƯỚC 2: Sử dụng Global Estimator để dự đoán tọa độ toàn cục
            global_coords = self.global_predictor.predict(X_pca)  # (x_global, y_global)
            
            # Đảm bảo global_coords là mảng numpy
            if isinstance(global_coords, list):
                global_coords = np.array(global_coords)
            
            # Xử lý trường hợp kết quả là mảng 1D
            if len(global_coords.shape) == 1 and len(X_pca) == 1:
                global_coords = global_coords.reshape(1, -1)
            
            # Xác định kích thước tọa độ (2D hoặc 3D)
            coord_dim = global_coords.shape[1] if len(global_coords.shape) > 1 else global_coords.size
            logger.info(f"Kích thước tọa độ dự đoán: {coord_dim}D")
            
            # BƯỚC 3: Thay thế phân cụm KMeans bằng phương pháp phân vùng grid tương tự fit()
            global_labels = np.zeros(len(global_coords), dtype=int)
            
            # Kiểm tra xem chúng ta có cluster_coords đã được tạo từ fit() không
            if len(self.cluster_coords) > 0:
                # Với từng tọa độ dự đoán, tìm cụm gần nhất dựa trên tọa độ trung bình của cụm
                for i in range(len(global_coords)):
                    min_dist = float('inf')
                    best_cluster = 0
                    
                    for cluster_id, cluster_info in self.cluster_coords.items():
                        cluster_mean = cluster_info['mean']
                        # Đảm bảo kích thước phù hợp cho việc tính toán khoảng cách
                        if len(global_coords[i]) != len(cluster_mean):
                            # Trường hợp kích thước khác nhau, chỉ so sánh kích thước chung nhỏ nhất
                            min_dim = min(len(global_coords[i]), len(cluster_mean))
                            dist = np.sqrt(np.sum((global_coords[i][:min_dim] - cluster_mean[:min_dim]) ** 2))
                        else:
                            # Tính khoảng cách Euclidean
                            dist = np.sqrt(np.sum((global_coords[i] - cluster_mean) ** 2))
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = cluster_id
                    
                    global_labels[i] = best_cluster
            else:
                # Nếu không có thông tin cluster_coords, sử dụng nhãn cục bộ
                logger.warning("Không có thông tin về cụm tọa độ, sử dụng nhãn cục bộ làm nhãn toàn cục")
                global_labels = local_labels.copy()
            
            # Thống kê nhãn phân loại
            match_count = np.sum(local_labels == global_labels)
            match_percentage = match_count / len(local_labels) * 100
            logger.info(f"Tỷ lệ nhãn cục bộ và toàn cục khớp nhau: {match_percentage:.2f}% ({match_count}/{len(local_labels)})")
            
            # Chi tiết hơn về phân bố các nhãn
            local_dist = np.bincount(local_labels)
            global_dist = np.bincount(global_labels)
            logger.info(f"Phân bố nhãn cục bộ: {local_dist}")
            logger.info(f"Phân bố nhãn toàn cục: {global_dist}")
            
            # Kết hợp dự đoán
            n_samples = len(X_pca)
            predictions = np.zeros((n_samples, coord_dim))
            details = {'local_labels': [], 'global_labels': [], 'match': [], 'selected': []} if return_details else None
            
            for i in range(n_samples):
                local_label = local_labels[i]
                global_label = global_labels[i]
                
                if return_details:
                    details['local_labels'].append(local_label)
                    details['global_labels'].append(global_label)
                
                # Dự đoán tọa độ cục bộ sử dụng bộ dự đoán của cụm
                if local_label in self.valid_clusters:
                    try:
                        # Dự đoán và xử lý kết quả
                        local_pred = self.cluster_predictors[local_label].predict([X_pca[i]])
                        
                        # Xử lý trường hợp kết quả là list
                        if isinstance(local_pred, list):
                            if len(local_pred) > 0:
                                local_coords = np.array(local_pred[0])
                            else:
                                logger.warning(f"Danh sách dự đoán rỗng, sử dụng tọa độ toàn cục")
                                local_coords = global_coords[i]
                        else:
                            # Trường hợp là mảng numpy
                            local_coords = local_pred[0] if len(local_pred.shape) > 1 else local_pred
                        
                        # Đảm bảo là mảng numpy
                        local_coords = np.array(local_coords)
                        
                        # Đảm bảo local_coords có cùng kích thước với global_coords
                        if len(local_coords) != coord_dim:
                            logger.warning(f"Kích thước tọa độ cục bộ không khớp: {len(local_coords)} vs {coord_dim}")
                            # Điều chỉnh kích thước
                            if len(local_coords) < coord_dim:
                                # Mở rộng mảng nếu cần
                                local_coords = np.pad(local_coords, (0, coord_dim - len(local_coords)), 'constant')
                            else:
                                # Cắt bớt mảng nếu cần
                                local_coords = local_coords[:coord_dim]
                    except Exception as e:
                        logger.warning(f"Lỗi khi dự đoán cục bộ cho nhãn {local_label}: {str(e)}")
                        local_coords = global_coords[i]
                else:
                    # Nếu không có bộ dự đoán cho cụm, sử dụng tọa độ toàn cục
                    local_coords = global_coords[i]
                
                # BƯỚC 4: So sánh C_local và C_global
                if local_label == global_label and local_label in self.valid_clusters:
                    # Nếu hai nhãn khớp nhau, sử dụng tọa độ cục bộ
                    predictions[i] = local_coords
                    if return_details:
                        details['match'].append(True)
                        details['selected'].append('local')
                else:
                    # Nếu không khớp, lấy trung bình của hai tọa độ
                    predictions[i] = (local_coords + global_coords[i]) / 2
                    if return_details:
                        details['match'].append(False)
                        details['selected'].append('average')
            
            return (predictions, details) if return_details else predictions
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def evaluate(self, X_test, coords_test):
        """Đánh giá mô hình"""
        try:
            if not self.trained:
                logger.error("Mô hình chưa được huấn luyện")
                return None
            
            # Đảm bảo dữ liệu là numpy arrays
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values
            elif isinstance(X_test, list):
                X_test = np.array(X_test)
                
            if isinstance(coords_test, pd.DataFrame):
                coords_test = coords_test.values
            elif isinstance(coords_test, list):
                coords_test = np.array(coords_test)
                
            # Chuyển đổi thành numpy array
            X_test = np.array(X_test)
            coords_test = np.array(coords_test)
            
            # Đảm bảo chỉ lấy tọa độ 2D (x, y)
            if coords_test.shape[1] > 2:
                logger.warning(f"Tọa độ kiểm tra có {coords_test.shape[1]} chiều. Chỉ sử dụng 2 chiều đầu tiên (x, y).")
                coords_test = coords_test[:, :2]
            
            # Xử lý đặc biệt cho coords_test có định dạng chuỗi 'AxB'
            coords_test_numeric = np.zeros((coords_test.shape[0], 2), dtype=float)
            for i in range(coords_test.shape[0]):
                for j in range(min(coords_test.shape[1], 2)):  # Đảm bảo chỉ xử lý tối đa 2 chiều
                    val = coords_test[i, j]
                    if isinstance(val, (int, float)):
                        coords_test_numeric[i, j] = val
                    elif isinstance(val, str) and 'x' in val:
                        # Xử lý chuỗi định dạng "AxB"
                        try:
                            # Lấy số đầu tiên trước dấu 'x'
                            coords_test_numeric[i, j] = float(val.split('x')[0])
                            logger.debug(f"Đã chuyển đổi tọa độ kiểm tra '{val}' thành {coords_test_numeric[i, j]}")
                        except (ValueError, IndexError):
                            logger.warning(f"Không thể chuyển đổi tọa độ kiểm tra '{val}', gán giá trị 0")
                            coords_test_numeric[i, j] = 0.0
                    else:
                        try:
                            coords_test_numeric[i, j] = float(val)
                        except (ValueError, TypeError):
                            logger.warning(f"Không thể chuyển đổi tọa độ kiểm tra '{val}', gán giá trị 0")
                            coords_test_numeric[i, j] = 0.0
            
            logger.info(f"Đã chuyển đổi tọa độ kiểm tra thành dạng 2D (x, y)")
            
            # Dự đoán và tính khoảng cách lỗi
            predictions, details = self.predict(X_test, return_details=True)
            
            if predictions is None:
                logger.error("Không thể thực hiện dự đoán")
                return None
                
            # Đảm bảo chỉ so sánh trong không gian 2D
            if predictions.shape[1] > 2:
                logger.warning(f"Tọa độ dự đoán có {predictions.shape[1]} chiều. Chỉ so sánh 2 chiều đầu tiên (x, y).")
                predictions = predictions[:, :2]
                
            # Tính khoảng cách Euclidean
            distances = np.sqrt(np.sum((predictions - coords_test_numeric) ** 2, axis=1))
            
            # Tính các metric cơ bản
            metrics = {
                'mean_distance': np.mean(distances),
                'median_distance': np.median(distances),
                'max_distance': np.max(distances),
                'min_distance': np.min(distances),
                'std_distance': np.std(distances)
            }
            
            # Log thông tin chi tiết về sai số
            logger.info(f"Sai số trung bình: {metrics['mean_distance']:.2f} cm")
            logger.info(f"Sai số trung vị: {metrics['median_distance']:.2f} cm")
            logger.info(f"Sai số lớn nhất: {metrics['max_distance']:.2f} cm")
            logger.info(f"Sai số nhỏ nhất: {metrics['min_distance']:.2f} cm")
            
            # Phân tích theo nhóm dự đoán
            cluster_idx = np.where(np.array(details['selected']) == 'local')[0]
            global_idx = np.where(np.array(details['selected']) == 'average')[0]
            
            if len(cluster_idx) > 0:
                metrics['cluster'] = {
                    'count': len(cluster_idx),
                    'percentage': len(cluster_idx) / len(distances) * 100,
                    'mean_distance': np.mean(distances[cluster_idx])
                }
                logger.info(f"Sai số trung bình cho mẫu khớp nhãn ({len(cluster_idx)} mẫu, {metrics['cluster']['percentage']:.1f}%): {metrics['cluster']['mean_distance']:.2f} cm")
            
            if len(global_idx) > 0:
                metrics['global'] = {
                    'count': len(global_idx),
                    'percentage': len(global_idx) / len(distances) * 100,
                    'mean_distance': np.mean(distances[global_idx])
                }
                logger.info(f"Sai số trung bình cho mẫu không khớp nhãn ({len(global_idx)} mẫu, {metrics['global']['percentage']:.1f}%): {metrics['global']['mean_distance']:.2f} cm")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá: {str(e)}")
            return None
    
    def save_models(self, base_path):
        """Lưu mô hình"""
        try:
            os.makedirs(base_path, exist_ok=True)
            
            # Lưu cấu hình
            config = {
                'max_clusters': self.max_clusters,
                'classifier_type': self.classifier_type,
                'global_predictor_type': self.global_predictor_type,
                'cluster_predictor_type': self.cluster_predictor_type,
                'n_neighbors': self.n_neighbors,
                'random_state': self.random_state,
                'use_kernel_pca': self.use_kernel_pca,
                'kernel': self.kernel,
                'n_components': self.n_components
            }
            joblib.dump(config, os.path.join(base_path, "config.pkl"))
            
            # Lưu các mô hình
            joblib.dump(self.coords_kmeans, os.path.join(base_path, "coords_kmeans.pkl"))
            joblib.dump(self.kernel_pca, os.path.join(base_path, "kernel_pca.pkl"))
            joblib.dump(self.cluster_coords, os.path.join(base_path, "cluster_coords.pkl"))
            
            # Lưu các bộ phân loại và dự đoán
            self.classifier.save(os.path.join(base_path, "classifier.pkl"))
            self.global_predictor.save(os.path.join(base_path, "global_predictor.pkl"))
            
            # Lưu các bộ dự đoán cụm
            os.makedirs(os.path.join(base_path, "predictors"), exist_ok=True)
            for cluster_id, predictor in self.cluster_predictors.items():
                predictor.save(os.path.join(base_path, f"predictors/cluster_{cluster_id}.pkl"))
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            return False
    
    @classmethod
    def load_models(cls, base_path):
        """Tải mô hình"""
        try:
            # Tải cấu hình
            config_path = os.path.join(base_path, "config.pkl")
            if not os.path.exists(config_path):
                logger.error(f"Không tìm thấy file cấu hình")
                return None
                
            config = joblib.load(config_path)
            instance = cls(**config)
            
            # Tải các mô hình
            instance.coords_kmeans = joblib.load(os.path.join(base_path, "coords_kmeans.pkl"))
            instance.kernel_pca = joblib.load(os.path.join(base_path, "kernel_pca.pkl"))
            instance.cluster_coords = joblib.load(os.path.join(base_path, "cluster_coords.pkl"))
            
            # Tải bộ phân loại và dự đoán
            instance.classifier = create_classifier(config['classifier_type'])
            instance.classifier = instance.classifier.__class__.load(os.path.join(base_path, "classifier.pkl"))
            
            instance.global_predictor = create_predictor(config['global_predictor_type'])
            instance.global_predictor = instance.global_predictor.__class__.load(os.path.join(base_path, "global_predictor.pkl"))
            
            # Tải các bộ dự đoán cụm
            predictors_dir = os.path.join(base_path, "predictors")
            import glob
            for pred_path in glob.glob(os.path.join(predictors_dir, "cluster_*.pkl")):
                cluster_id = int(os.path.basename(pred_path).split("_")[1].split(".")[0])
                predictor = create_predictor(config['cluster_predictor_type'])
                instance.cluster_predictors[cluster_id] = predictor.__class__.load(pred_path)
            
            instance.valid_clusters = list(instance.cluster_predictors.keys())
            instance.trained = True
            return instance
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return None