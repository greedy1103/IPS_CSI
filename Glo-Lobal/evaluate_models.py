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

def run_all_combinations(X_train, coords_train, X_test, coords_test, y_test=None, classifier_types=None, predictor_types=None):
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
                    # Tạo mô hình
                    model = ClusterRegression(
                        max_clusters=3,
                        classifier_type=classifier_type,
                        global_predictor_type=global_predictor_type,
                        cluster_predictor_type=cluster_predictor_type,
                        random_state=42
                    )
                    
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
    Tạo bảng so sánh để đánh giá hiệu suất các kết hợp mô hình khác nhau.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame chứa kết quả đánh giá
    
    Returns
    -------
    pandas.DataFrame
        Bảng so sánh tổng hợp
    """
    comparison_cols = ['classifier_type', 'global_predictor_type', 'cluster_predictor_type']
    metric_cols = ['mean_distance', 'median_distance', 'max_distance', 'min_distance']
    
    # Kiểm tra xem có cột matching_percentage hay không
    if 'matching_percentage' in results_df.columns:
        metric_cols.extend(['matching_percentage', 'matching_mean_distance', 
                           'non_matching_percentage', 'non_matching_mean_distance'])
    
    # Thêm các cột thông tin tốc độ
    if 'train_time' in results_df.columns:
        metric_cols.append('train_time')
    if 'eval_time' in results_df.columns:
        metric_cols.append('eval_time')
    
    # Tạo bảng so sánh
    comparison_table = results_df[comparison_cols + metric_cols].copy()
    
    # Tìm mô hình tốt nhất theo mean_distance
    best_idx = comparison_table['mean_distance'].idxmin()
    comparison_table['is_best'] = False
    comparison_table.loc[best_idx, 'is_best'] = True
    
    # Xếp hạng các mô hình
    comparison_table['rank'] = comparison_table['mean_distance'].rank()
    
    # Tạo cột mô tả kết hợp
    comparison_table['combination'] = comparison_table.apply(
        lambda row: f"{row['classifier_type']}-{row['global_predictor_type']}-{row['cluster_predictor_type']}",
        axis=1
    )
    
    # Sắp xếp theo mean_distance
    comparison_table = comparison_table.sort_values('mean_distance')
    
    return comparison_table

def visualize_comparison(comparison_table, save_path=None):
    """
    Tạo biểu đồ so sánh hiệu suất các mô hình.
    
    Parameters
    ----------
    comparison_table : pandas.DataFrame
        Bảng dữ liệu so sánh
    save_path : str, optional
        Đường dẫn để lưu biểu đồ
    """
    if comparison_table.empty:
        logger.error("Bảng so sánh trống, không thể tạo biểu đồ.")
        return
    
    # Tạo biểu đồ so sánh mean_distance
    plt.figure(figsize=(12, 8))
    
    # Tìm các kết hợp tốt nhất
    top_combinations = comparison_table.head(10)['combination'].values
    
    # Lọc dữ liệu cho biểu đồ
    if len(comparison_table) > 10:
        plot_data = comparison_table[comparison_table['combination'].isin(top_combinations)]
        logger.info(f"Hiển thị 10 kết hợp tốt nhất trong tổng số {len(comparison_table)} kết hợp")
    else:
        plot_data = comparison_table
    
    # Tạo biểu đồ cột
    bar_colors = ['g' if is_best else 'b' for is_best in plot_data['is_best']]
    bars = plt.bar(plot_data['combination'], plot_data['mean_distance'], color=bar_colors)
    
    # Thêm giá trị trên đỉnh cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=0)
    
    # Tùy chỉnh biểu đồ
    plt.title('So sánh khoảng cách lỗi trung bình giữa các kết hợp mô hình')
    plt.xlabel('Kết hợp mô hình (Classifier-GlobalPredictor-ClusterPredictor)')
    plt.ylabel('Khoảng cách lỗi trung bình (cm)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Thêm đường tham chiếu
    plt.axhline(y=plot_data['mean_distance'].min(), color='r', linestyle='--', 
                label=f'Tốt nhất: {plot_data["mean_distance"].min():.2f} cm')
    plt.legend()
    
    # Lưu biểu đồ nếu cần
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Đã lưu biểu đồ so sánh tại {save_path}")
    
    plt.show()
    
    # Tạo biểu đồ thứ hai để so sánh tỷ lệ khớp và khoảng cách lỗi
    if 'matching_percentage' in plot_data.columns:
        plt.figure(figsize=(12, 8))
        
        # Thiết lập trục X cho các kết hợp
        x = np.arange(len(plot_data))
        width = 0.35
        
        # Vẽ biểu đồ cột cho khoảng cách lỗi
        ax1 = plt.subplot(111)
        bars1 = ax1.bar(x - width/2, plot_data['matching_mean_distance'], width, label='Khớp nhãn')
        bars2 = ax1.bar(x + width/2, plot_data['non_matching_mean_distance'], width, label='Không khớp nhãn')
        
        # Thêm đường cho tỷ lệ khớp
        ax2 = ax1.twinx()
        ax2.plot(x, plot_data['matching_percentage'], 'ro-', label='Tỷ lệ khớp (%)')
        
        # Tùy chỉnh biểu đồ
        ax1.set_xlabel('Kết hợp mô hình')
        ax1.set_ylabel('Khoảng cách lỗi (cm)')
        ax2.set_ylabel('Tỷ lệ khớp (%)')
        ax1.set_title('So sánh khoảng cách lỗi và tỷ lệ khớp nhãn')
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_data['combination'], rotation=45, ha='right')
        
        # Kết hợp legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            match_path = os.path.splitext(save_path)[0] + '_matching.png'
            plt.savefig(match_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ so sánh tỷ lệ khớp tại {match_path}")
        
        plt.show()

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
                        help='Tỷ lệ dữ liệu CSI sử dụng cho huấn luyện (0-1). Ví dụ: 0.5 sẽ chỉ sử dụng 50% dữ liệu CSI')
    
    # Tham số đánh giá
    parser.add_argument('--quick_test', action='store_true',
                        help='Chỉ đánh giá một số mô hình nhanh chóng')
    parser.add_argument('--best_metric', type=str, default=EVALUATION_BEST_METRIC,
                        choices=EVALUATION_METRICS, help='Metric chính để so sánh')
    
    # Tham số mô hình
    parser.add_argument('--classifier_types', type=str, nargs='+',
                        choices=SUPPORTED_CLASSIFIER_TYPES,
                        help='Chỉ định loại bộ phân loại cụ thể để đánh giá (vd: knn rf svm)')
    parser.add_argument('--predictor_types', type=str, nargs='+',
                        choices=SUPPORTED_PREDICTOR_TYPES,
                        help='Chỉ định loại bộ dự đoán cụ thể để đánh giá (vd: knn rf svr gb)')
    
    # Thư mục kết quả
    parser.add_argument('--results_dir', type=str, default=MODEL_COMPARISON_DIR,
                        help='Thư mục lưu kết quả đánh giá')
    
    # Chi tiết đánh giá
    parser.add_argument('--visualize', action='store_true',
                        help='Hiển thị biểu đồ so sánh')
    
    return parser.parse_args()

def main():
    """
    Hàm chính để đánh giá và so sánh các mô hình.
    """
    # Phân tích tham số dòng lệnh
    args = parse_arguments()
    
    logger.info("=== BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH ===")
    logger.info(f"Thư mục dữ liệu: {args.data_dir}")
    logger.info(f"Tỷ lệ dữ liệu: {args.subset_ratio}")
    logger.info(f"Chế độ kiểm tra nhanh: {args.quick_test}")
    
    try:
        # Xác định đường dẫn kết quả trung gian
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(args.results_dir, f"evaluation_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        intermediate_results_path = os.path.join(results_dir, "intermediate_results.json")
        
        # Khởi tạo processor
        processor = DataPreprocessor(random_state=RANDOM_SEED)
        
        # Xử lý dữ liệu - áp dụng subset_ratio khi load dữ liệu
        logger.info("Đang xử lý dữ liệu...")
        start_time = time.time()
        X_train, X_test, coords_train, coords_test = processor.load_data(subset_ratio=args.subset_ratio)
        y_train, y_test = None, None  # Khởi tạo biến y_train và y_test với giá trị None
        data_processing_time = time.time() - start_time
        logger.info(f"Xử lý dữ liệu hoàn tất trong {data_processing_time:.2f} giây")
        
        if X_train is None or X_test is None:
            logger.error("Lỗi xử lý dữ liệu! Không thể tiếp tục đánh giá.")
            return
        
        logger.info(f"Dữ liệu huấn luyện: {X_train.shape}, Dữ liệu kiểm tra: {X_test.shape}")
        logger.info(f"Tọa độ huấn luyện: {coords_train.shape}, Tọa độ kiểm tra: {coords_test.shape}")
        
        # Chọn loại mô hình cho đánh giá
        if args.classifier_types:
            # Sử dụng danh sách bộ phân loại từ tham số
            classifier_types = args.classifier_types
            logger.info(f"Sử dụng bộ phân loại theo tham số: {', '.join(classifier_types)}")
        elif args.quick_test:
            classifier_types = QUICK_TEST_CLASSIFIERS
            logger.info(f"Chế độ kiểm tra nhanh: đánh giá {len(classifier_types)} bộ phân loại")
        else:
            classifier_types = SUPPORTED_CLASSIFIER_TYPES
            logger.info(f"Chế độ đánh giá đầy đủ: đánh giá {len(classifier_types)} bộ phân loại")
        
        if args.predictor_types:
            # Sử dụng danh sách bộ dự đoán từ tham số
            predictor_types = args.predictor_types
            logger.info(f"Sử dụng bộ dự đoán theo tham số: {', '.join(predictor_types)}")
        elif args.quick_test:
            predictor_types = QUICK_TEST_PREDICTORS
            logger.info(f"Chế độ kiểm tra nhanh: đánh giá {len(predictor_types)} bộ dự đoán")
        else:
            predictor_types = SUPPORTED_PREDICTOR_TYPES
            logger.info(f"Chế độ đánh giá đầy đủ: đánh giá {len(predictor_types)} bộ dự đoán")
        
        # Tính số lượng kết hợp cần đánh giá
        n_combinations = len(classifier_types) * len(predictor_types) * len(predictor_types)
        logger.info(f"Tổng số kết hợp cần đánh giá: {n_combinations}")
        
        # Đánh giá tất cả các kết hợp - không áp dụng subset_ratio nữa vì đã được áp dụng khi tải dữ liệu
        start_time = time.time()
        results = run_all_combinations(
            X_train, coords_train, X_test, coords_test, y_test,
            classifier_types=classifier_types,
            predictor_types=predictor_types
        )
        evaluation_time = time.time() - start_time
        logger.info(f"Đánh giá hoàn tất trong {evaluation_time:.2f} giây")
        
        # Lưu kết quả cuối cùng
        final_results_path = os.path.join(results_dir, "final_results.json")
        with open(final_results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Đã lưu kết quả cuối cùng tại {final_results_path}")
        
        # Tạo DataFrame từ kết quả
        results_data = []
        for result in results:
            # Kiểm tra kết quả hợp lệ
            if result is None or 'model' not in result:
                logger.warning("Bỏ qua kết quả không hợp lệ")
                continue
                
            # Trích xuất thông tin mô hình
            model_info = result['model']
            
            # Tạo dữ liệu cơ bản
            result_entry = {
                'classifier_type': model_info['classifier_type'],
                'global_predictor_type': model_info['global_predictor_type'],
                'cluster_predictor_type': model_info['cluster_predictor_type'],
                'train_time': model_info['train_time'],
                'eval_time': model_info['eval_time']
            }
            
            # Thêm metrics hiệu suất
            result_entry['mean_distance'] = result['mean_distance']
            result_entry['median_distance'] = result['median_distance']
            
            if 'std_distance' in result:
                result_entry['std_distance'] = result['std_distance']
                
            if 'max_distance' in result:
                result_entry['max_distance'] = result['max_distance']
                
            if 'min_distance' in result:
                result_entry['min_distance'] = result['min_distance']
            
            # Thêm thông tin về matching/non-matching
            if 'matching' in result:
                result_entry['matching_count'] = result['matching']['count']
                result_entry['matching_percentage'] = result['matching']['percentage']
                result_entry['matching_mean_distance'] = result['matching']['mean_distance']
            
            if 'non_matching' in result:
                result_entry['non_matching_count'] = result['non_matching']['count']
                result_entry['non_matching_percentage'] = result['non_matching']['percentage']
                result_entry['non_matching_mean_distance'] = result['non_matching']['mean_distance']
            
            results_data.append(result_entry)
        
        # Kiểm tra xem có kết quả hợp lệ không
        if not results_data:
            logger.error("Không có kết quả hợp lệ để tạo bảng so sánh")
            return
            
        # Tạo DataFrame và lưu
        results_df = pd.DataFrame(results_data)
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Đã lưu kết quả dạng CSV tại {csv_path}")
        
        # Tạo bảng so sánh
        try:
            comparison_table = create_comparison_table(results_df)
            comparison_csv_path = os.path.join(results_dir, "comparison_table.csv")
            comparison_table.to_csv(comparison_csv_path, index=False)
            logger.info(f"Đã lưu bảng so sánh tại {comparison_csv_path}")
            
            # In ra các mô hình tốt nhất
            logger.info("\n=== MÔ HÌNH TỐT NHẤT ===")
            best_model = comparison_table.iloc[0]
            logger.info(f"Kết hợp: {best_model['combination']}")
            logger.info(f"Classifier: {best_model['classifier_type']}")
            logger.info(f"Global Predictor: {best_model['global_predictor_type']}")
            logger.info(f"Cluster Predictor: {best_model['cluster_predictor_type']}")
            logger.info(f"Khoảng cách trung bình: {best_model['mean_distance']:.2f} cm")
            logger.info(f"Khoảng cách trung vị: {best_model['median_distance']:.2f} cm")
            
            # Hiển thị biểu đồ
            if args.visualize:
                plot_path = os.path.join(results_dir, "model_comparison.png")
                visualize_comparison(comparison_table, save_path=plot_path)
            
            # Lưu thông tin tổng hợp
            summary = {
                'timestamp': timestamp,
                'data_dir': args.data_dir,
                'subset_ratio': args.subset_ratio,
                'quick_test': args.quick_test,
                'n_combinations': n_combinations,
                'data_processing_time': data_processing_time,
                'evaluation_time': evaluation_time,
                'best_model': {
                    'classifier_type': best_model['classifier_type'],
                    'global_predictor_type': best_model['global_predictor_type'],
                    'cluster_predictor_type': best_model['cluster_predictor_type'],
                    'mean_distance': float(best_model['mean_distance']),
                    'median_distance': float(best_model['median_distance'])
                }
            }
            
            summary_path = os.path.join(results_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            logger.info(f"Đã lưu thông tin tổng hợp tại {summary_path}")
        except Exception as e:
            logger.error(f"Lỗi khi tạo bảng so sánh: {str(e)}")
        
        logger.info("=== ĐÁNH GIÁ MÔ HÌNH HOÀN TẤT ===")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình đánh giá mô hình: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()