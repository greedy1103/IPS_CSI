#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script dự đoán vị trí từ dữ liệu CSI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import json
import time
from datetime import datetime

# Thêm thư mục gốc vào đường dẫn để import các module từ dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataprocessor import DataPreprocessor
from utils.cluster_regression import ClusterRegression
from utils.cluster_utils import visualize_clusters, visualize_prediction_comparison
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED,
    DEFAULT_CLASSIFIER_TYPE, DEFAULT_GLOBAL_PREDICTOR_TYPE, DEFAULT_CLUSTER_PREDICTOR_TYPE,
    DEFAULT_MAX_CLUSTERS, DEFAULT_N_NEIGHBORS
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Phân tích tham số dòng lệnh.
    
    Returns
    -------
    argparse.Namespace
        Các tham số từ dòng lệnh
    """
    parser = argparse.ArgumentParser(description='Dự đoán vị trí từ dữ liệu CSI')
    
    # Tham số dữ liệu
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Đường dẫn đến thư mục dữ liệu')
    
    # Tham số mô hình
    parser.add_argument('--classifier_type', type=str, default=DEFAULT_CLASSIFIER_TYPE,
                        help='Loại bộ phân loại cho mô hình')
    parser.add_argument('--global_predictor_type', type=str, default=DEFAULT_GLOBAL_PREDICTOR_TYPE,
                        help='Loại bộ dự đoán toàn cục')
    parser.add_argument('--cluster_predictor_type', type=str, default=DEFAULT_CLUSTER_PREDICTOR_TYPE,
                        help='Loại bộ dự đoán cho mỗi cụm')
    parser.add_argument('--max_clusters', type=int, default=DEFAULT_MAX_CLUSTERS,
                        help='Số lượng cụm tối đa cho phân cụm')
    parser.add_argument('--n_neighbors', type=int, default=DEFAULT_N_NEIGHBORS,
                        help='Số lượng láng giềng cho KNN')
    
    # Tham số huấn luyện và đánh giá
    parser.add_argument('--subset_ratio', type=float, default=1.0, 
                        help='Tỷ lệ dữ liệu sử dụng cho huấn luyện')
    parser.add_argument('--random_state', type=int, default=RANDOM_SEED,
                        help='Seed ngẫu nhiên')
    
    # Tham số lưu và tải mô hình
    parser.add_argument('--save_model', action='store_true',
                        help='Lưu mô hình sau khi huấn luyện')
    parser.add_argument('--load_model', action='store_true',
                        help='Tải mô hình đã huấn luyện')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Đường dẫn đến file mô hình')
    
    # Tham số trực quan hóa
    parser.add_argument('--visualize', action='store_true',
                        help='Trực quan hóa kết quả')
    
    # Tham số đầu ra
    parser.add_argument('--save_results', action='store_true',
                        help='Lưu kết quả dự đoán')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                        help='Thư mục lưu kết quả')
    
    return parser.parse_args()

def main():
    """
    Hàm chính để dự đoán vị trí từ dữ liệu CSI.
    """
    # Phân tích tham số dòng lệnh
    args = parse_arguments()
    
    logger.info("=== BẮT ĐẦU CHƯƠNG TRÌNH ===")
    logger.info(f"Thư mục dữ liệu: {args.data_dir}")
    logger.info(f"Mô hình: Classifier={args.classifier_type}, Global={args.global_predictor_type}, Cluster={args.cluster_predictor_type}")
    
    try:
        # Xác định đường dẫn kết quả
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(args.results_dir, f"prediction_{timestamp}")
        
        if args.save_results:
            os.makedirs(results_dir, exist_ok=True)
            logger.info(f"Thư mục kết quả: {results_dir}")
        
        # Khởi tạo processor
        processor = DataPreprocessor(random_state=args.random_state)
        
        # Xử lý dữ liệu
        logger.info("Đang xử lý dữ liệu...")
        start_time = time.time()
        X_train, X_test, y_train, y_test, coords_train, coords_test = processor.load_data()
        data_processing_time = time.time() - start_time
        logger.info(f"Xử lý dữ liệu hoàn tất trong {data_processing_time:.2f} giây")
        
        if X_train is None or X_test is None:
            logger.error("Lỗi xử lý dữ liệu! Không thể tiếp tục.")
            return
        
        logger.info(f"Dữ liệu huấn luyện: {X_train.shape}, Dữ liệu kiểm tra: {X_test.shape}")
        logger.info(f"Tọa độ huấn luyện: {coords_train.shape}, Tọa độ kiểm tra: {coords_test.shape}")
        
        # Khởi tạo hoặc tải mô hình
        if args.load_model and args.model_path:
            logger.info(f"Đang tải mô hình từ {args.model_path}...")
            model = ClusterRegression.load_models(args.model_path)
            logger.info("Tải mô hình thành công!")
        else:
            logger.info("Khởi tạo mô hình mới...")
            model = ClusterRegression(
                max_clusters=args.max_clusters,
                classifier_type=args.classifier_type,
                global_predictor_type=args.global_predictor_type,
                cluster_predictor_type=args.cluster_predictor_type,
                random_state=args.random_state
            )
        
        # Huấn luyện mô hình nếu không tải mô hình
        if not args.load_model or not args.model_path:
            logger.info("Bắt đầu huấn luyện mô hình...")
            start_time = time.time()
            model.fit(X_train, coords=coords_train)
            train_time = time.time() - start_time
            logger.info(f"Huấn luyện mô hình hoàn tất trong {train_time:.2f} giây")
            
            # Lưu mô hình nếu cần
            if args.save_model:
                if not args.model_path:
                    model_dir = os.path.join(MODELS_DIR, f"model_{timestamp}")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, "model.joblib")
                else:
                    model_path = args.model_path
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                logger.info(f"Đang lưu mô hình tại {model_path}...")
                model.save_models(model_path)
                logger.info("Lưu mô hình thành công!")
        
        # Đánh giá mô hình
        logger.info("Đang đánh giá mô hình...")
        start_time = time.time()
        eval_results = model.evaluate(X_test, coords_test)
        eval_time = time.time() - start_time
        
        # Hiển thị kết quả đánh giá
        logger.info("Kết quả đánh giá:")
        logger.info(f"  Khoảng cách trung bình: {eval_results['mean_distance']:.2f} cm")
        logger.info(f"  Khoảng cách trung vị: {eval_results['median_distance']:.2f} cm")
        logger.info(f"  Khoảng cách lớn nhất: {eval_results['max_distance']:.2f} cm")
        logger.info(f"  Khoảng cách nhỏ nhất: {eval_results['min_distance']:.2f} cm")
        logger.info(f"  Độ lệch chuẩn: {eval_results['std_distance']:.2f} cm")
        logger.info(f"  Thời gian đánh giá: {eval_time:.2f} giây")
        
        # Trực quan hóa kết quả
        if args.visualize:
            predictions, details = model.predict(X_test, return_details=True)
            
            # Phân cụm
            if hasattr(model, 'clusters_') and model.clusters_ is not None:
                plot_path = os.path.join(results_dir, "clusters.png") if args.save_results else None
                visualize_clusters(coords_train, model.clusters_, save_path=plot_path)
            
            # So sánh dự đoán
            plot_path = os.path.join(results_dir, "predictions.png") if args.save_results else None
            visualize_prediction_comparison(coords_test, predictions, save_path=plot_path)
        
        # Lưu kết quả
        if args.save_results:
            # Lưu kết quả đánh giá
            eval_results_path = os.path.join(results_dir, "evaluation_results.json")
            with open(eval_results_path, 'w') as f:
                json.dump(eval_results, f, indent=4)
            logger.info(f"Đã lưu kết quả đánh giá tại {eval_results_path}")
            
            # Lưu dự đoán
            predictions, _ = model.predict(X_test)
            predictions_df = pd.DataFrame({
                'true_x': coords_test[:, 0],
                'true_y': coords_test[:, 1],
                'pred_x': predictions[:, 0],
                'pred_y': predictions[:, 1]
            })
            
            # Tính khoảng cách lỗi
            distances = np.sqrt(np.sum((coords_test - predictions) ** 2, axis=1))
            predictions_df['error_distance'] = distances
            
            # Lưu file CSV
            predictions_path = os.path.join(results_dir, "predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Đã lưu kết quả dự đoán tại {predictions_path}")
        
        logger.info("=== CHƯƠNG TRÌNH KẾT THÚC ===")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()