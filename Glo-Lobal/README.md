# Mô hình dự đoán vị trí sử dụng các phương pháp khác nhau

Dự án này thực hiện việc dự đoán tọa độ dựa trên dữ liệu CSI, với khả năng thử nghiệm nhiều phương pháp khác nhau cho bộ phân loại cluster và bộ dự đoán tọa độ.

## Cấu trúc dự án

```
Final/
│
├── Glo-Lobal/                          # Thư mục chứa mã nguồn chính
│   ├── regressor.py                    # Phiên bản ban đầu của mô hình
│   ├── evaluate_models.py              # Script đánh giá nhiều mô hình
│   └── README.md                       # Tài liệu hướng dẫn
│
├── utils/                              # Thư mục chứa các tiện ích
│   ├── models/                         # Thư mục chứa các lớp mô hình
│   │   ├── __init__.py
│   │   ├── base_models.py              # Lớp cơ sở cho các mô hình
│   │   ├── classifiers.py              # Các bộ phân loại cluster
│   │   ├── predictors.py               # Các bộ dự đoán tọa độ
│   │   └── model_factory.py            # Factory để tạo mô hình
│   │
│   ├── cluster_utils.py                # Tiện ích phân cụm
│   ├── dataprocessor.py                # Bộ xử lý dữ liệu
│   └── cluster_regression.py           # Mô hình hồi quy theo cluster
│
└── config.py                           # Cấu hình chung
```

## Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn
```

Để sử dụng XGBoost (tùy chọn):
```bash
pip install xgboost
```

## Sử dụng

### Chạy mô hình với một phương pháp cụ thể

```bash
python regressor.py --classifier_type knn --global_predictor_type knn --cluster_predictor_type knn --max_clusters 5
```

Các tham số có thể sử dụng:
- `--classifier_type`: Loại bộ phân loại cluster (`knn`, `rf`, `svm`, `gb`, `mlp`)
- `--global_predictor_type`: Loại bộ dự đoán tọa độ toàn cục (`knn`, `rf`, `svr`, `gb`, `mlp`)
- `--cluster_predictor_type`: Loại bộ dự đoán tọa độ cho từng cluster (`knn`, `rf`, `svr`, `gb`, `mlp`)
- `--n_neighbors`: Số lượng neighbors cho KNN
- `--auto_tune`: Tự động tìm giá trị n_neighbors tối ưu
- `--max_clusters`: Số lượng cụm tối đa để kiểm tra
- `--visualize`: Hiển thị biểu đồ phân cụm

### Đánh giá và so sánh nhiều phương pháp

```bash
python evaluate_models.py --max_clusters 5 --n_neighbors 5
```

Đánh giá nhanh một số mô hình phổ biến:
```bash
python evaluate_models.py --max_clusters 5 --quick_test
```

Đánh giá một tập hợp cụ thể các mô hình:
```bash
python evaluate_models.py --max_clusters 5 --classifier_types knn rf gb --global_predictor_types knn svr gb --cluster_predictor_types knn svr gb
```

Script này sẽ:
1. Chạy tất cả các kết hợp của bộ phân loại và bộ dự đoán
2. Tạo bảng so sánh kết quả
3. Vẽ biểu đồ so sánh
4. Xác định mô hình tốt nhất

## Các phương pháp được hỗ trợ

### Bộ phân loại cluster

- `KNNClusterClassifier`: Sử dụng KNN để phân loại cluster
- `RandomForestClusterClassifier`: Sử dụng Random Forest để phân loại cluster
- `SVMClusterClassifier`: Sử dụng SVM để phân loại cluster
- `GBClusterClassifier`: Sử dụng Gradient Boosting để phân loại cluster
- `XGBClusterClassifier`: Sử dụng XGBoost để phân loại cluster (yêu cầu cài đặt XGBoost)

### Bộ dự đoán tọa độ

- `KNNCoordinatePredictor`: Sử dụng KNN để dự đoán tọa độ
- `RFCoordinatePredictor`: Sử dụng Random Forest để dự đoán tọa độ
- `SVRCoordinatePredictor`: Sử dụng Support Vector Regression để dự đoán tọa độ
- `GBCoordinatePredictor`: Sử dụng Gradient Boosting để dự đoán tọa độ

## Mở rộng

Để thêm một phương pháp mới:

1. Tạo lớp mới trong `utils/models/classifiers.py` hoặc `utils/models/predictors.py`
2. Thêm phương pháp vào factory trong `utils/models/model_factory.py`
3. Cập nhật danh sách lựa chọn trong `regressor.py` và `evaluate_models.py`

## Xử lý lỗi phổ biến

- **Feature names warning**: Cảnh báo về tên đặc trưng đã được xử lý trong code và không ảnh hưởng đến kết quả.
- **Lỗi train thiếu tham số**: Đảm bảo rằng dữ liệu đầu vào có cả nhãn và tọa độ khi gọi `train()`.
- **Lỗi không tìm thấy bộ dự đoán**: Kiểm tra lại tham số `--classifier_type`, `--global_predictor_type`, và `--cluster_predictor_type`.

## Kết quả

Sau khi chạy đánh giá, kết quả sẽ được lưu trong thư mục `results/model_comparison`:
- `model_comparison.csv`: Bảng so sánh các mô hình
- `model_comparison.png`: Biểu đồ so sánh
- `model_comparison_summary.json`: Tổng hợp kết quả và thông tin về mô hình tốt nhất
- `intermediate_results.json`: Kết quả trung gian (có thể dùng để tiếp tục đánh giá)

hello