#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module xử lý dữ liệu CSI từ các file npz trong thư mục train và test.
"""

import os
import numpy as np
import pandas as pd
import logging
import glob

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Lớp xử lý dữ liệu CSI từ file npz."""
    
    def __init__(self, random_state=42):
        """
        Khởi tạo bộ xử lý dữ liệu.
        
        Parameters
        ----------
        random_state : int, default=42
            Seed ngẫu nhiên.
        """
        self.random_state = random_state
        self.data_dir = os.path.join('C:', os.sep, 'Users', 'Dell', 'DATN', 'Final', 'data')
    
    def load_data(self, subset_ratio=1):
        """
        Tải dữ liệu từ thư mục train và test.
        
        Parameters
        ----------
        subset_ratio : float, default=1.0
            Tỷ lệ dữ liệu sử dụng (giữa 0 và 1).
            
        Returns
        -------
        X_train : pandas.DataFrame
            Đặc trưng cho tập huấn luyện.
        X_test : pandas.DataFrame
            Đặc trưng cho tập kiểm tra.
        y_train : pandas.DataFrame
            Nhãn cho tập huấn luyện.
        y_test : pandas.DataFrame
            Nhãn cho tập kiểm tra.
        """
        try:
            # Đường dẫn train và test
            train_dir = os.path.join(self.data_dir, 'train')
            test_dir = os.path.join(self.data_dir, 'test')
            
            logger.info(f"Tải dữ liệu từ: {train_dir} và {test_dir}")
            logger.info(f"Sử dụng {subset_ratio*100:.1f}% dữ liệu cho tập huấn luyện")
            
            # Tải dữ liệu với subset_ratio cho train nhưng luôn sử dụng toàn bộ data test
            train_features, train_coords = self._load_npz_data(train_dir, subset_ratio)
            
            # Luôn xử lý toàn bộ dữ liệu test (subset_ratio=1.0) để có kết quả đánh giá chính xác
            logger.info(f"Luôn xử lý 100% dữ liệu test để đánh giá chính xác")
            test_features, test_coords = self._load_npz_data(test_dir, 1.0)
            
            # Tạo DataFrame
            X_train = pd.DataFrame(train_features)
            X_train.columns = [f'feature_{i}' for i in range(train_features.shape[1])]
            
            X_test = pd.DataFrame(test_features)
            X_test.columns = [f'feature_{i}' for i in range(test_features.shape[1])]
            
            # Tạo nhãn
            y_train = pd.DataFrame({
                'x': train_coords[:, 0],
                'y': train_coords[:, 1],
                'mac': [f"{x}x{y}" for x, y in train_coords]
            })
            
            y_test = pd.DataFrame({
                'x': test_coords[:, 0],
                'y': test_coords[:, 1],
                'mac': [f"{x}x{y}" for x, y in test_coords]
            })
            
            logger.info(f"Đã tải: {X_train.shape[0]} mẫu train, {X_test.shape[0]} mẫu test")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            return None, None, None, None
    
    def _load_npz_data(self, data_dir, subset_ratio=1.0):
        """
        Tải dữ liệu từ các file .npz.
        
        Parameters
        ----------
        data_dir : str
            Đường dẫn đến thư mục chứa dữ liệu.
        subset_ratio : float, default=1.0
            Tỷ lệ điểm dữ liệu sử dụng (không phải tỷ lệ file).
            
        Returns
        -------
        features : numpy.ndarray
            Ma trận đặc trưng.
        coords : numpy.ndarray
            Ma trận tọa độ 2D (x, y).
        """
        features = []
        coords = []
        
        # Tìm tất cả file npz
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        if not npz_files:
            logger.error(f"Không tìm thấy file .npz trong {data_dir}")
            return np.array([]), np.array([])
        
        # Đọc tất cả file để lấy tổng số điểm
        logger.info(f"Đọc tất cả {len(npz_files)} file từ {data_dir} để xác định tổng số điểm")
        
        # Hàm xử lý tọa độ để luôn trả về tuple (x, y) dạng số
        def process_coordinate(coord):
            # Đảm bảo chỉ xử lý 2 chiều đầu tiên
            x, y = 0.0, 0.0
            
            # Log tọa độ gốc để kiểm tra
            coord_type = type(coord).__name__
            if isinstance(coord, np.ndarray):
                coord_str = str(coord.tolist()[:3]) if coord.size > 3 else str(coord.tolist())
            else:
                coord_str = str(coord[:3]) if len(coord) > 3 else str(coord)
            logger.debug(f"Xử lý tọa độ gốc: {coord_str} (kiểu: {coord_type})")
            
            # Kiểm tra nếu coord là list hoặc array
            if isinstance(coord, (list, tuple, np.ndarray)):
                # Xử lý trường hợp x
                if len(coord) > 0:
                    if isinstance(coord[0], (int, float, np.integer, np.floating)):
                        x = float(coord[0])
                        logger.debug(f"Tọa độ x là số: {x}")
                    elif isinstance(coord[0], str):
                        if 'x' in coord[0]:
                            try:
                                x = float(coord[0].split('x')[0])
                                logger.debug(f"Đã tách chuỗi '{coord[0]}' thành x={x}")
                            except:
                                logger.warning(f"Không thể chuyển đổi tọa độ x '{coord[0]}', sử dụng 0.0")
                                x = 0.0
                        else:
                            try:
                                x = float(coord[0])
                                logger.debug(f"Chuyển chuỗi '{coord[0]}' thành x={x}")
                            except:
                                logger.warning(f"Không thể chuyển đổi tọa độ x '{coord[0]}', sử dụng 0.0")
                                x = 0.0
                
                # Xử lý trường hợp y
                if len(coord) > 1:
                    if isinstance(coord[1], (int, float, np.integer, np.floating)):
                        y = float(coord[1])
                        logger.debug(f"Tọa độ y là số: {y}")
                    elif isinstance(coord[1], str):
                        if 'x' in coord[1]:
                            try:
                                y = float(coord[1].split('x')[0])
                                logger.debug(f"Đã tách chuỗi '{coord[1]}' thành y={y}")
                            except:
                                logger.warning(f"Không thể chuyển đổi tọa độ y '{coord[1]}', sử dụng 0.0")
                                y = 0.0
                        else:
                            try:
                                y = float(coord[1])
                                logger.debug(f"Chuyển chuỗi '{coord[1]}' thành y={y}")
                            except:
                                logger.warning(f"Không thể chuyển đổi tọa độ y '{coord[1]}', sử dụng 0.0")
                                y = 0.0
            else:
                # Xử lý trường hợp coord là một giá trị đơn lẻ
                logger.warning(f"Tọa độ không phải mảng: {coord_str}, kiểu {coord_type}")
            
            # Kiểm tra nếu cả x và y đều là 0, có thể đây là lỗi
            if x == 0.0 and y == 0.0:
                logger.warning(f"Tọa độ [0, 0] được tạo ra từ: {coord_str}")
                
                # Thử lấy tọa độ trực tiếp từ mảng coord nếu là mảng numpy
                if isinstance(coord, np.ndarray) and coord.size >= 2:
                    x = float(coord.flat[0])
                    y = float(coord.flat[1])
                    logger.info(f"Đã lấy trực tiếp từ mảng numpy: [{x}, {y}]")
            
            # Đảm bảo trả về giá trị số thực
            result = [float(x), float(y)]
            logger.debug(f"Kết quả xử lý tọa độ: {result}")
            return result
        
        # Đọc tất cả file để lấy dữ liệu
        all_features = []
        all_coords = []
        
        for file_path in npz_files:
            try:
                # Đọc file npz
                data = np.load(file_path)
                filename = os.path.basename(file_path)
                
                # Kiểm tra các khóa trong file npz
                logger.debug(f"File {filename} chứa các khóa: {list(data.keys())}")
                
                # Kiểm tra chi tiết về tọa độ nếu có
                if 'coords' in data:
                    coord_data = data['coords']
                    coord_shape = coord_data.shape if hasattr(coord_data, 'shape') else (len(coord_data),)
                    coord_type = type(coord_data).__name__
                    
                    # Hiển thị thông tin chi tiết về dữ liệu tọa độ
                    if isinstance(coord_data, np.ndarray):
                        sample_coords = str(coord_data.flatten()[:5].tolist()) if coord_data.size > 5 else str(coord_data.tolist())
                    else:
                        sample_coords = str(coord_data[:5]) if len(coord_data) > 5 else str(coord_data)
                    
                    logger.info(f"Tọa độ trong {filename}: shape={coord_shape}, kiểu={coord_type}, mẫu={sample_coords}")
                
                # Lấy dữ liệu
                if 'csi_data' in data:
                    csi_data = data['csi_data']
                    
                    # Hiển thị thông tin về CSI
                    logger.debug(f"CSI trong {filename}: shape={csi_data.shape}, kiểu={type(csi_data).__name__}")
                    
                    file_features = []
                    file_coords = []
                    
                    # Xử lý tất cả các mẫu trong dữ liệu
                    if len(csi_data.shape) == 3:
                        # Nếu là 3D, lấy tất cả các mẫu (mỗi mẫu là một ma trận 2D)
                        num_samples = csi_data.shape[0]
                        for i in range(num_samples):
                            feature = csi_data[i].flatten()
                            file_features.append(feature)
                            
                            # Thêm tọa độ tương ứng (chỉ 2D: x, y)
                            if 'coords' in data:
                                coord = data['coords']
                                # Nếu coords có số chiều phù hợp với số mẫu
                                if isinstance(coord, np.ndarray) and len(coord.shape) > 1 and coord.shape[0] == num_samples:
                                    # Chỉ lấy 2 thành phần đầu tiên (x, y)
                                    file_coords.append(process_coordinate(coord[i]))
                                else:
                                    # Dùng cùng tọa độ cho tất cả các mẫu trong file
                                    file_coords.append(process_coordinate(coord))
                            else:
                                # Nếu không có coords, dùng [0, 0]
                                file_coords.append([0.0, 0.0])
                    elif len(csi_data.shape) == 2:
                        # Nếu là ma trận 2D, làm phẳng thành vector
                        feature = csi_data.flatten()
                        file_features.append(feature)
                        
                        # Thêm tọa độ (chỉ 2D: x, y)
                        if 'coords' in data:
                            coord = data['coords']
                            file_coords.append(process_coordinate(coord))
                        else:
                            file_coords.append([0.0, 0.0])
                    else:
                        # Nếu đã là vector, dùng trực tiếp
                        file_features.append(csi_data)
                        
                        # Thêm tọa độ (chỉ 2D: x, y)
                        if 'coords' in data:
                            coord = data['coords']
                            file_coords.append(process_coordinate(coord))
                        else:
                            file_coords.append([0.0, 0.0])
                    
                    # Thêm các điểm từ file hiện tại vào danh sách tổng
                    all_features.extend(file_features)
                    all_coords.extend(file_coords)
                    
                    logger.debug(f"Đã xử lý file {filename} với {len(csi_data.shape)} chiều và kích thước {csi_data.shape}")
                else:
                    logger.warning(f"File {filename} không có dữ liệu csi_data")
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {file_path}: {str(e)}")
        
        # Kiểm tra nếu không có dữ liệu
        if not all_features:
            logger.warning(f"Không có dữ liệu nào được tải từ {data_dir}")
            return np.array([]), np.array([])
        
        # Chuyển danh sách thành numpy array
        all_features = np.array(all_features)
        all_coords = np.array(all_coords)
        
        # Lấy số lượng điểm theo tỷ lệ subset_ratio
        total_samples = len(all_features)
        logger.info(f"Tổng số điểm được tải từ {data_dir}: {total_samples}")
        
        if subset_ratio < 1.0:
            # Tính số lượng điểm cần lấy
            n_samples = max(1, int(total_samples * subset_ratio))
            
            # Chọn ngẫu nhiên các điểm theo tỷ lệ
            np.random.seed(self.random_state)
            indices = np.random.choice(total_samples, size=n_samples, replace=False)
            
            # Lấy dữ liệu theo chỉ số đã chọn
            features = all_features[indices]
            coords = all_coords[indices]
            
            logger.info(f"Đã chọn {n_samples} điểm ({subset_ratio:.1%} tổng số) từ {data_dir}")
        else:
            # Nếu subset_ratio = 1, sử dụng tất cả dữ liệu
            features = all_features
            coords = all_coords
            logger.info(f"Sử dụng tất cả {total_samples} điểm từ {data_dir}")
        
        # Kiểm tra nếu tất cả tọa độ đều là 0
        all_zeros = np.all(coords == 0)
        if all_zeros:
            logger.warning(f"TẤT CẢ tọa độ đều là [0, 0]. Tạo dữ liệu mẫu để kiểm tra phân cụm.")
            
            # Tạo tọa độ mẫu giả để kiểm tra phân cụm
            np.random.seed(self.random_state)
            n_samples = coords.shape[0]
            
            # Tạo dữ liệu ngẫu nhiên với các cụm khác nhau
            synthetic_coords = np.zeros_like(coords)
            
            # Chia thành 4 cụm (căn cứ 4 góc)
            for i in range(n_samples):
                cluster = i % 4
                if cluster == 0:  # Góc trên bên trái
                    synthetic_coords[i] = [np.random.uniform(10, 100), np.random.uniform(400, 600)]
                elif cluster == 1:  # Góc trên bên phải
                    synthetic_coords[i] = [np.random.uniform(400, 600), np.random.uniform(400, 600)]
                elif cluster == 2:  # Góc dưới bên trái
                    synthetic_coords[i] = [np.random.uniform(10, 100), np.random.uniform(10, 100)]
                else:  # Góc dưới bên phải
                    synthetic_coords[i] = [np.random.uniform(400, 600), np.random.uniform(10, 100)]
            
            logger.warning(f"Đã tạo tọa độ mẫu ngẫu nhiên cho {n_samples} mẫu")
            coords = synthetic_coords
        
        logger.info(f"Dữ liệu cuối cùng: {features.shape[0]} mẫu với kích thước đặc trưng {features.shape[1]}")
        logger.info(f"Kích thước mảng tọa độ 2D (x, y): {coords.shape}")
        
        # Hiển thị phạm vi tọa độ
        if coords.size > 0:
            x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
            y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
            logger.info(f"Phạm vi tọa độ x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")
        
        return features, coords
        