#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module chứa các hàm tiện ích cho phân cụm và xử lý dữ liệu.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

def find_optimal_clusters(X, max_clusters=10, random_state=42, min_clusters=2, visualize=False):
    """
    Tìm số lượng cụm tối ưu bằng cách sử dụng điểm silhouette.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào.
    max_clusters : int, default=10
        Số lượng cụm tối đa để kiểm tra.
    random_state : int, default=42
        Seed ngẫu nhiên.
    min_clusters : int, default=2
        Số lượng cụm tối thiểu để kiểm tra.
    visualize : bool, default=False
        Hiển thị biểu đồ điểm silhouette.
        
    Returns
    -------
    optimal_n_clusters : int
        Số lượng cụm tối ưu.
    silhouette_scores : list
        Danh sách điểm silhouette cho mỗi số lượng cụm.
    """
    # Đảm bảo X là numeric
    try:
        # Kiểm tra xem X có phải là đối tượng có thể chuyển đổi thành mảng không
        if hasattr(X, 'values'):
            X_numeric = X.values  # Nếu là DataFrame hoặc Series
        else:
            X_numeric = np.array(X)  # Nếu là danh sách hoặc mảng
            
        # Kiểm tra và chuyển đổi từng phần tử nếu cần
        if X_numeric.dtype == object or X_numeric.dtype == str:
            logger.info("Phát hiện dữ liệu không phải số, tiến hành chuyển đổi")
            X_numeric_converted = np.zeros_like(X_numeric, dtype=float)
            
            # Kiểm tra từng phần tử
            for i in range(X_numeric.shape[0]):
                for j in range(X_numeric.shape[1]):
                    val = X_numeric[i, j]
                    if isinstance(val, (int, float)):
                        X_numeric_converted[i, j] = val
                    elif isinstance(val, str) and 'x' in val:
                        # Xử lý chuỗi định dạng "AxB"
                        try:
                            # Lấy số đầu tiên trước dấu 'x'
                            X_numeric_converted[i, j] = float(val.split('x')[0])
                            logger.debug(f"Đã chuyển đổi '{val}' thành {X_numeric_converted[i, j]}")
                        except (ValueError, IndexError):
                            logger.warning(f"Không thể chuyển đổi '{val}', gán giá trị 0")
                            X_numeric_converted[i, j] = 0.0
                    else:
                        try:
                            X_numeric_converted[i, j] = float(val)
                        except (ValueError, TypeError):
                            logger.warning(f"Không thể chuyển đổi '{val}', gán giá trị 0")
                            X_numeric_converted[i, j] = 0.0
            
            X_numeric = X_numeric_converted
            logger.info(f"Đã chuyển đổi X thành mảng số có hình dạng {X_numeric.shape}")
    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi X thành mảng số: {str(e)}")
        logger.warning("Sử dụng mảng X ban đầu")
        X_numeric = X
    
    # Giới hạn số cụm dựa trên kích thước dữ liệu
    n_samples = len(X_numeric)
    max_clusters = min(max_clusters, n_samples // 10)
    max_clusters = max(max_clusters, min_clusters + 1)  # Đảm bảo ít nhất 2 cụm để kiểm tra
    
    silhouette_scores = []
    logger.info(f"Tìm số lượng cụm tối ưu trong khoảng từ {min_clusters} đến {max_clusters}")
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_numeric)
            
            if len(np.unique(cluster_labels)) < n_clusters:
                logger.warning(f"Chỉ tìm thấy {len(np.unique(cluster_labels))} cụm thay vì {n_clusters}")
                silhouette_avg = -1
            else:
                silhouette_avg = silhouette_score(X_numeric, cluster_labels)
            
            silhouette_scores.append(silhouette_avg)
            logger.info(f"Với {n_clusters} cụm, điểm silhouette là {silhouette_avg:.4f}")
        except Exception as e:
            logger.error(f"Lỗi khi tính điểm silhouette cho {n_clusters} cụm: {str(e)}")
            silhouette_scores.append(-1)
    
    if all(score == -1 for score in silhouette_scores):
        logger.warning("Không thể tính điểm silhouette cho bất kỳ số lượng cụm nào. Đặt mặc định là 2 cụm.")
        optimal_n_clusters = min_clusters
    else:
        # Chọn số lượng cụm tối ưu
        optimal_n_clusters = np.argmax(silhouette_scores) + min_clusters
    
    if visualize and not all(score == -1 for score in silhouette_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'o-')
        plt.axvline(x=optimal_n_clusters, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Số lượng cụm')
        plt.ylabel('Điểm silhouette')
        plt.title('Điểm silhouette cho các số lượng cụm khác nhau')
        plt.show()
    
    logger.info(f"Số lượng cụm tối ưu: {optimal_n_clusters}")
    return optimal_n_clusters, silhouette_scores

def visualize_clusters(X, y, cluster_labels, title="Phân cụm dữ liệu", save_path=None):
    """
    Hiển thị kết quả phân cụm.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào.
    y : array-like, shape (n_samples, 2)
        Tọa độ thực tế.
    cluster_labels : array-like, shape (n_samples,)
        Nhãn cụm.
    title : str, default="Phân cụm dữ liệu"
        Tiêu đề biểu đồ.
    save_path : str, default=None
        Đường dẫn để lưu biểu đồ. Nếu None, biểu đồ sẽ không được lưu.
        
    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 10))
    
    # Chuyển đổi y thành mảng numpy nếu cần
    if isinstance(y, pd.DataFrame):
        y = y.values
    
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        plt.scatter(
            y[mask, 0],
            y[mask, 1],
            alpha=0.7,
            label=f'Cụm {cluster_id}'
        )
    
    plt.title(title)
    plt.xlabel('Tọa độ X (cm)')
    plt.ylabel('Tọa độ Y (cm)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Đã lưu biểu đồ phân cụm tại: {save_path}")
    
    plt.show()

def visualize_prediction_comparison(true_coords, pred_coords, title="So sánh tọa độ thực tế và dự đoán", save_path=None):
    """
    Hiển thị so sánh giữa tọa độ thực tế và dự đoán.
    
    Parameters
    ----------
    true_coords : array-like, shape (n_samples, 2)
        Tọa độ thực tế.
    pred_coords : array-like, shape (n_samples, 2)
        Tọa độ dự đoán.
    title : str, default="So sánh tọa độ thực tế và dự đoán"
        Tiêu đề biểu đồ.
    save_path : str, default=None
        Đường dẫn để lưu biểu đồ. Nếu None, biểu đồ sẽ không được lưu.
        
    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 10))
    
    # Chuyển đổi sang mảng numpy nếu cần
    if isinstance(true_coords, pd.DataFrame):
        true_coords = true_coords.values
    if isinstance(pred_coords, pd.DataFrame):
        pred_coords = pred_coords.values
    
    plt.scatter(true_coords[:, 0], true_coords[:, 1], c='blue', alpha=0.7, label='Tọa độ thực tế')
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', alpha=0.7, label='Tọa độ dự đoán')
    
    # Vẽ đường nối giữa tọa độ thực tế và dự đoán
    for i in range(len(true_coords)):
        plt.plot([true_coords[i, 0], pred_coords[i, 0]],
                 [true_coords[i, 1], pred_coords[i, 1]],
                 'k-', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Tọa độ X (cm)')
    plt.ylabel('Tọa độ Y (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Đã lưu biểu đồ so sánh dự đoán tại: {save_path}")
    
    plt.show()

def calculate_metrics(true_coords, pred_coords):
    """
    Tính các chỉ số đánh giá giữa tọa độ thực tế và dự đoán.
    
    Parameters
    ----------
    true_coords : array-like, shape (n_samples, 2 or 3)
        Tọa độ thực tế (x, y) hoặc (x, y, z).
    pred_coords : array-like, shape (n_samples, 2 or 3)
        Tọa độ dự đoán (x, y) hoặc (x, y, z).
        
    Returns
    -------
    metrics : dict
        Từ điển chứa các chỉ số đánh giá:
        - mae_x, mae_y, mae_z: Sai số tuyệt đối trung bình theo từng trục
        - rmse: Căn bậc hai của sai số bình phương trung bình
        - mean_distance: Khoảng cách Euclidean trung bình
        - median_distance: Khoảng cách Euclidean trung vị
        - std_distance: Độ lệch chuẩn của khoảng cách Euclidean
        - distances: Mảng khoảng cách Euclidean
    """
    # Chuyển đổi sang numpy array nếu cần
    if isinstance(true_coords, pd.DataFrame):
        true_coords = true_coords.values
    if isinstance(pred_coords, pd.DataFrame):
        pred_coords = pred_coords.values
    
    # Tính sai số tuyệt đối trung bình cho mỗi trục
    mae_x = np.mean(np.abs(true_coords[:, 0] - pred_coords[:, 0]))
    mae_y = np.mean(np.abs(true_coords[:, 1] - pred_coords[:, 1]))
    
    # Tính khoảng cách Euclidean
    if true_coords.shape[1] == 3 and pred_coords.shape[1] == 3:
        mae_z = np.mean(np.abs(true_coords[:, 2] - pred_coords[:, 2]))
        distances = np.sqrt(np.sum((true_coords - pred_coords) ** 2, axis=1))
    else:
        mae_z = None
        distances = np.sqrt(np.sum((true_coords[:, :2] - pred_coords[:, :2]) ** 2, axis=1))
    
    # Tính RMSE
    mse = np.mean(np.sum((true_coords - pred_coords) ** 2, axis=1))
    rmse = np.sqrt(mse)
    
    # Khoảng cách trung bình và trung vị
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    std_distance = np.std(distances)
    
    # Trả về từ điển chỉ số
    metrics = {
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_z': mae_z,
        'rmse': rmse,
        'mean_distance': mean_distance,
        'median_distance': median_distance,
        'std_distance': std_distance,
        'distances': distances
    }
    
    return metrics 