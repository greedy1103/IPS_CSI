#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File cấu hình cho dự án dự đoán vị trí dựa trên dữ liệu CSI.
"""

import os
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Đường dẫn cơ sở
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_COMPARISON_DIR = os.path.join(RESULTS_DIR, 'model_comparison')

# Đảm bảo các thư mục tồn tại
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, MODEL_COMPARISON_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Cấu hình tỷ lệ dữ liệu sử dụng (1.0 = 100%)
DATA_SUBSET_RATIO = 1  # Mặc định sử dụng toàn bộ dữ liệu

# Cấu hình seed ngẫu nhiên
RANDOM_SEED = 42

# Cấu hình mặc định cho mô hình
DEFAULT_CLASSIFIER_TYPE = 'knn'
DEFAULT_GLOBAL_PREDICTOR_TYPE = 'knn'
DEFAULT_CLUSTER_PREDICTOR_TYPE = 'knn'
DEFAULT_MAX_CLUSTERS = 5
DEFAULT_N_NEIGHBORS = 5
DEFAULT_AUTO_TUNE = False

# Danh sách các loại mô hình được hỗ trợ
SUPPORTED_CLASSIFIER_TYPES = ['knn', 'rf', 'svm', 'gb', 'xgb']
SUPPORTED_PREDICTOR_TYPES = ['knn', 'rf', 'svr', 'gb']

# Cấu hình cho chạy nhanh
QUICK_TEST_CLASSIFIERS = ['knn', 'rf']
QUICK_TEST_PREDICTORS = ['knn', 'rf']

# Cấu hình đánh giá mô hình
EVALUATION_TEST_SIZE = 0.2
EVALUATION_METRICS = [
    'mean_distance',    # Khoảng cách Euclidean trung bình
    'median_distance',  # Khoảng cách Euclidean trung vị
    'std_distance',     # Độ lệch chuẩn của khoảng cách Euclidean
    'rmse',             # Căn bậc hai của sai số bình phương trung bình
    'mae_x',            # Sai số tuyệt đối trung bình theo trục x
    'mae_y'             # Sai số tuyệt đối trung bình theo trục y
]
EVALUATION_BEST_METRIC = 'mean_distance'  # Metric để xác định mô hình tốt nhất (giá trị càng thấp càng tốt)

# Cấu hình trực quan hóa
PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 300

# Cấu hình cho DataLoader
BATCH_SIZE = 32      # Số lượng mẫu trong mỗi batch
NUM_WORKERS = 8      # Số worker cho việc load dữ liệu

# Cấu hình cho KernelPCA (giảm chiều dữ liệu)
# Kernel Cosine - Phù hợp cho tín hiệu viễn thông
KERNEL = 'cosine'  # K(x,y) = cos(θ) = (x^T y) / (||x|| * ||y||)

# Kernel RBF 
#KERNEL = 'rbf' # K(x,y) = exp(-gamma * ||x-y||²)
#GAMMA = 0.1   

N_COMPONENTS = 150

# Cấu hình đường dẫn
SAVE_DIR = 'C:/Users/Dell/DATN/Final/results'  # Thư mục lưu kết quả 