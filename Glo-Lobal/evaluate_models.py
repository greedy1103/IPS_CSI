#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script đánh giá và so sánh các mô hình dự đoán vị trí khác nhau.
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
import warnings
from datetime import datetime
import itertools
import glob

# Thêm thư mục gốc vào đường dẫn để import các module từ dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataprocessor import DataPreprocessor
from utils.cluster_regression import ClusterRegression
from utils.cluster_utils import visualize_clusters, visualize_prediction_comparison
from utils.models.model_factory import create_classifier, create_predictor
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, MODEL_COMPARISON_DIR, RANDOM_SEED,
    DEFAULT_CLASSIFIER_TYPE, DEFAULT_GLOBAL_PREDICTOR_TYPE, DEFAULT_CLUSTER_PREDICTOR_TYPE,
    DEFAULT_MAX_CLUSTERS, DEFAULT_N_NEIGHBORS, DEFAULT_AUTO_TUNE,
    SUPPORTED_CLASSIFIER_TYPES, SUPPORTED_PREDICTOR_TYPES,
    QUICK_TEST_CLASSIFIERS, QUICK_TEST_PREDICTORS,
    EVALUATION_METRICS, EVALUATION_BEST_METRIC, DATA_SUBSET_RATIO
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Loại bỏ cảnh báo không cần thiết
warnings.filterwarnings("ignore", category=UserWarning)

def save_intermediate_results(results, file_path):
    """
    Lưu kết quả trung gian để tiếp tục sau nếu quá trình bị gián đoạn.
    
    Parameters
    ----------
    results : list
        Danh sách các kết quả đánh giá.
    file_path : str
        Đường dẫn tới file lưu kết quả.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Đã lưu kết quả trung gian tại {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu kết quả trung gian: {str(e)}")

def load_intermediate_results(file_path):
    """
    Tải kết quả trung gian từ file.
    
    Parameters
    ----------
    file_path : str
        Đường dẫn tới file kết quả.
        
    Returns
    -------
    list
        Danh sách các kết quả đánh giá.
    """
    if not os.path.exists(file_path):
        logger.info(f"Không tìm thấy file kết quả trung gian {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Đã tải {len(results)} kết quả từ {file_path}")
        return results
    except Exception as e:
        logger.error(f"Lỗi khi tải kết quả trung gian: {str(e)}")
        return []

def run_all_combinations(X_train, coords_train, X_test, coords_test, y_test=None, classifier_types=None, predictor_types=None, clustering_method=None, dbscan_eps=None, dbscan_min_samples=None):
    """
    Thử nghiệm mọi kết hợp có thể của các loại bộ phân loại và bộ dự đoán.
    
    Parameters
    ----------
    X_train : array-like
        Dữ liệu huấn luyện
    coords_train : array-like
        Tọa độ tương ứng với dữ liệu huấn luyện
    X_test : array-like
        Dữ liệu kiểm tra
    coords_test : array-like
        Tọa độ tương ứng với dữ liệu kiểm tra
    y_test : array-like, optional
        Nhãn thực tế cho dữ liệu kiểm tra
    classifier_types : list, optional
        Danh sách các loại bộ phân loại để thử nghiệm
    predictor_types : list, optional
        Danh sách các loại bộ dự đoán để thử nghiệm
    clustering_method : str, optional
        Phương pháp phân cụm
    dbscan_eps : float, optional
        Khoảng cách tối đa cho DBSCAN
    dbscan_min_samples : int, optional
        Số mẫu tối thiểu cho DBSCAN
        
    Returns
    -------
    results : list
        Danh sách các kết quả đánh giá
    """
    # In ra thông tin về dữ liệu đầu vào
    logger.info("Thông tin dữ liệu đầu vào:")
    logger.info(f"X_train type: {type(X_train)}")
    logger.info(f"coords_train type: {type(coords_train)}")
    if X_train is not None:
        logger.info(f"X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}")
    if coords_train is not None:
        logger.info(f"coords_train shape: {coords_train.shape if hasattr(coords_train, 'shape') else 'unknown'}")
    
    # Tiền xử lý dữ liệu đầu vào
    try:
        # Chuyển đổi dữ liệu thành numpy arrays nếu cần
        if X_train is not None and not isinstance(X_train, np.ndarray):
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            else:
                X_train = np.array(X_train)
        
        if coords_train is not None and not isinstance(coords_train, np.ndarray):
            if isinstance(coords_train, pd.DataFrame):
                coords_train = coords_train.values
            else:
                coords_train = np.array(coords_train)
                
        if X_test is not None and not isinstance(X_test, np.ndarray):
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values
            else:
                X_test = np.array(X_test)
                
        if coords_test is not None and not isinstance(coords_test, np.ndarray):
            if isinstance(coords_test, pd.DataFrame):
                coords_test = coords_test.values
            else:
                coords_test = np.array(coords_test)
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        return []
    
    # Đặt giá trị mặc định
    if classifier_types is None:
        classifier_types = ['knn', 'rf', 'svm']
    
    if predictor_types is None:
        predictor_types = ['knn', 'rf', 'svr', 'gb']
    
    # Lưu kết quả cho mọi kết hợp
    results = []
    
    for classifier_type in classifier_types:
        for global_predictor_type in predictor_types:
            for cluster_predictor_type in predictor_types:
                # Ghi log kết hợp hiện tại
                logger.info(f"Thử nghiệm kết hợp: Classifier={classifier_type}, " + 
                            f"Global Predictor={global_predictor_type}, " + 
                            f"Cluster Predictor={cluster_predictor_type}")
                
                try:
                    # Tạo mô hình với các tham số phân cụm
                    model_params = {
                        'max_clusters': 5,
                        'classifier_type': classifier_type,
                        'global_predictor_type': global_predictor_type,
                        'cluster_predictor_type': cluster_predictor_type,
                        'random_state': 42,
                        'clustering_method': clustering_method
                    }
                    
                    # Thêm tham số DBSCAN nếu được chỉ định
                    if clustering_method == 'dbscan':
                        model_params['dbscan_eps'] = dbscan_eps
                        model_params['dbscan_min_samples'] = dbscan_min_samples
                    
                    # Tạo mô hình với các tham số đã cấu hình
                    model = ClusterRegression(**model_params)
                    
                    # Huấn luyện mô hình
                    start_time = time.time()
                    model.fit(X_train, coords=coords_train)
                    train_time = time.time() - start_time
                    
                    # Đánh giá mô hình
                    start_time = time.time()
                    eval_results = model.evaluate(X_test, coords_test)
                    eval_time = time.time() - start_time
                    
                    if eval_results is None:
                        logger.error("Không nhận được kết quả đánh giá")
                        continue
                    
                    # Thêm thông tin về mô hình vào kết quả
                    eval_results['model'] = {
                        'classifier_type': classifier_type,
                        'global_predictor_type': global_predictor_type,
                        'cluster_predictor_type': cluster_predictor_type,
                        'clustering_method': clustering_method,
                        'train_time': train_time,
                        'eval_time': eval_time
                    }
                    
                    # Thêm vào danh sách kết quả
                    results.append(eval_results)
                    
                    # Ghi log kết quả chính
                    logger.info(f"Kết quả đánh giá:")
                    logger.info(f"  Khoảng cách trung bình: {eval_results['mean_distance']:.2f} cm")
                    logger.info(f"  Khoảng cách trung vị: {eval_results['median_distance']:.2f} cm")
                    logger.info(f"  Thời gian huấn luyện: {train_time:.2f} giây")
                    logger.info(f"  Thời gian đánh giá: {eval_time:.2f} giây")
                
                except Exception as e:
                    logger.error(f"Lỗi trong kết hợp {classifier_type}-{global_predictor_type}-{cluster_predictor_type}: {str(e)}")
                    continue
                
                # Lưu kết quả sau mỗi lần đánh giá
                if results:
                    with open('evaluation_results.json', 'w') as f:
                        json.dump(results, f, indent=2)
    
    return results

def create_comparison_table(results_df):
    """
    Tạo bảng so sánh từ DataFrame kết quả.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame chứa kết quả đánh giá
        
    Returns
    -------
    pandas.DataFrame
        Bảng so sánh đã được sắp xếp
    """
    # Kiểm tra cấu trúc dữ liệu
    logger.info("Cấu trúc dữ liệu kết quả:")
    logger.info(f"Columns: {results_df.columns.tolist()}")
    logger.info(f"Sample data:\n{results_df.head()}")
    
    # Xác định các cột cần thiết
    model_cols = ['classifier_type', 'global_predictor_type', 'cluster_predictor_type', 'clustering_method']
    metric_cols = ['mean_distance', 'median_distance', 'train_time', 'eval_time']
    
    # Kiểm tra xem các cột có tồn tại không
    missing_cols = [col for col in model_cols + metric_cols if col not in results_df.columns]
    if missing_cols:
        logger.warning(f"Các cột sau không tồn tại trong DataFrame: {missing_cols}")
        logger.warning("Thử trích xuất từ cột 'model'")
        
        # Nếu có cột 'model', trích xuất thông tin từ đó
        if 'model' in results_df.columns:
            # Tạo các cột mới từ thông tin trong cột 'model'
            for col in model_cols:
                if col not in results_df.columns:
                    results_df[col] = results_df['model'].apply(lambda x: x.get(col, 'unknown'))
    
    # Kiểm tra lại các cột
    available_cols = [col for col in model_cols + metric_cols if col in results_df.columns]
    logger.info(f"Các cột có sẵn: {available_cols}")
    
    # Tạo bảng so sánh
    comparison_table = results_df[available_cols].copy()
    
    # Sắp xếp theo khoảng cách trung bình
    if 'mean_distance' in comparison_table.columns:
        comparison_table = comparison_table.sort_values('mean_distance')
    
    return comparison_table

def visualize_comparison(comparison_table, save_path=None):
    """
    Trực quan hóa kết quả so sánh các mô hình.
    
    Parameters
    ----------
    comparison_table : pandas.DataFrame
        Bảng so sánh các mô hình
    save_path : str, optional
        Đường dẫn để lưu biểu đồ
    """
    try:
        # Tạo cột combination nếu chưa có
        if 'combination' not in comparison_table.columns:
            # Kiểm tra các cột cần thiết
            required_cols = ['classifier_type', 'global_predictor_type', 'cluster_predictor_type', 'clustering_method']
            missing_cols = [col for col in required_cols if col not in comparison_table.columns]
            
            if missing_cols:
                logger.warning(f"Thiếu các cột cần thiết: {missing_cols}")
                return
            
            # Tạo cột combination
            comparison_table['combination'] = comparison_table.apply(
                lambda row: f"{row['classifier_type']}-{row['global_predictor_type']}-{row['cluster_predictor_type']}-{row['clustering_method']}",
                axis=1
            )
        
        # Lấy 10 kết hợp tốt nhất
        top_combinations = comparison_table.head(10)['combination'].values
        top_metrics = comparison_table.head(10)['mean_distance'].values
        
        # Tạo biểu đồ
        plt.figure(figsize=(12, 6))
        bars = plt.bar(top_combinations, top_metrics)
        
        # Thêm giá trị trên mỗi cột
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('So sánh hiệu suất các kết hợp mô hình')
        plt.xlabel('Kết hợp mô hình')
        plt.ylabel('Khoảng cách trung bình (cm)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Lưu biểu đồ nếu có đường dẫn
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Đã lưu biểu đồ so sánh tại: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ so sánh: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def parse_arguments():
    """
    Phân tích tham số dòng lệnh.
    
    Returns
    -------
    argparse.Namespace
        Các tham số từ dòng lệnh
    """
    parser = argparse.ArgumentParser(description='Đánh giá và so sánh các mô hình dự đoán vị trí')
    
    # Tham số dữ liệu
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Đường dẫn đến thư mục dữ liệu')
    parser.add_argument('--subset_ratio', type=float, default=DATA_SUBSET_RATIO,
                        help='Tỷ lệ dữ liệu sử dụng cho đánh giá')
    
    # Tham số đánh giá
    parser.add_argument('--quick_test', action='store_true',
                        help='Chỉ đánh giá một số mô hình nhanh chóng')
    parser.add_argument('--best_metric', type=str, default=EVALUATION_BEST_METRIC,
                        choices=EVALUATION_METRICS, help='Metric chính để so sánh')
    
    # Tham số mô hình
    parser.add_argument('--classifier_types', type=str, nargs='+',
                        default=QUICK_TEST_CLASSIFIERS,
                        help='Danh sách các loại bộ phân loại để thử nghiệm')
    parser.add_argument('--predictor_types', type=str, nargs='+',
                        default=QUICK_TEST_PREDICTORS,
                        help='Danh sách các loại bộ dự đoán để thử nghiệm')
    parser.add_argument('--clustering_method', type=str, default='grid',
                        help='Phương pháp phân cụm (grid/kmeans/dbscan)')
    parser.add_argument('--dbscan_eps', type=float, default=0.5,
                        help='Khoảng cách tối đa cho DBSCAN')
    parser.add_argument('--dbscan_min_samples', type=int, default=5,
                        help='Số mẫu tối thiểu cho DBSCAN')
    
    # Thư mục kết quả
    parser.add_argument('--results_dir', type=str, default=MODEL_COMPARISON_DIR,
                        help='Thư mục lưu kết quả đánh giá')
    
    # Chi tiết đánh giá
    parser.add_argument('--visualize', action='store_true',
                        help='Hiển thị biểu đồ so sánh')
    
    # Tham số khác
    parser.add_argument('--random_state', type=int, default=RANDOM_SEED,
                        help='Seed ngẫu nhiên')
    parser.add_argument('--save_results', action='store_true',
                        help='Lưu kết quả đánh giá')
    
    return parser.parse_args()

def main():
    """Hàm chính để chạy đánh giá mô hình."""
    # Phân tích tham số dòng lệnh
    args = parse_arguments()
    
    # Thiết lập logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # In thông tin về tham số
    logger.info(f"Thư mục dữ liệu: {args.data_dir}")
    logger.info(f"Tỷ lệ dữ liệu: {args.subset_ratio}")
    logger.info(f"Loại bộ phân loại: {args.classifier_types}")
    logger.info(f"Loại bộ dự đoán: {args.predictor_types}")
    logger.info(f"Phương pháp phân cụm: {args.clustering_method}")
    if args.clustering_method == 'dbscan':
        logger.info(f"DBSCAN eps: {args.dbscan_eps}")
        logger.info(f"DBSCAN min_samples: {args.dbscan_min_samples}")
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Khởi tạo bộ xử lý dữ liệu
    processor = DataPreprocessor(random_state=args.random_state)
    
    # Tải dữ liệu
    X_train, X_test, y_train, y_test, coords_train, coords_test = processor.load_data(subset_ratio=args.subset_ratio)
    
    if X_train is None or X_test is None:
        logger.error("Không thể tải dữ liệu")
        return
    
    # Chạy đánh giá với các tham số đã chỉ định
    results = run_all_combinations(
        X_train=X_train,
        coords_train=coords_train,
        X_test=X_test,
        coords_test=coords_test,
        y_test=y_test,
        classifier_types=args.classifier_types,
        predictor_types=args.predictor_types,
        clustering_method=args.clustering_method,
        dbscan_eps=args.dbscan_eps if args.clustering_method == 'dbscan' else None,
        dbscan_min_samples=args.dbscan_min_samples if args.clustering_method == 'dbscan' else None
    )
    
    if not results:
        logger.error("Không có kết quả đánh giá nào được tạo ra")
        return
    
    # Tạo DataFrame từ kết quả
    results_df = pd.DataFrame(results)
    
    # Tạo bảng so sánh
    comparison_table = create_comparison_table(results_df)
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.results_dir, f"evaluation_results_{timestamp}.csv")
    comparison_file = os.path.join(args.results_dir, f"comparison_table_{timestamp}.csv")
    
    results_df.to_csv(results_file, index=False)
    comparison_table.to_csv(comparison_file, index=False)
    
    logger.info(f"Đã lưu kết quả đánh giá vào {results_file}")
    logger.info(f"Đã lưu bảng so sánh vào {comparison_file}")
    
    # Vẽ biểu đồ so sánh
    visualize_comparison(comparison_table, save_path=os.path.join(args.results_dir, f"comparison_plot_{timestamp}.png"))
    
    logger.info("Hoàn tất đánh giá mô hình")

if __name__ == "__main__":
    main()