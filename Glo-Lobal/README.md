# Mô hình dự đoán vị trí sử dụng các phương pháp khác nhau

Dự án này thực hiện việc dự đoán tọa độ dựa trên dữ liệu CSI, với hai công cụ chính:
1. `regressor.py`: Huấn luyện và lưu một mô hình cụ thể
2. `evaluate_models.py`: So sánh hiệu suất của nhiều mô hình khác nhau

## Huấn luyện mô hình (regressor.py)

Script này dùng để huấn luyện và lưu một mô hình cụ thể với các tham số đã chọn.

```bash
# Huấn luyện và lưu mô hình
python regressor.py --classifier_type knn --global_predictor_type rf --cluster_predictor_type rf --save_model

# Tải và sử dụng mô hình đã lưu
python regressor.py --load_model --model_path path/to/model
```

Các tham số chính:
- `--classifier_type`: Loại bộ phân loại cluster (`knn`, `rf`, `svm`, `gb`)
- `--global_predictor_type`: Loại bộ dự đoán tọa độ toàn cục
- `--cluster_predictor_type`: Loại bộ dự đoán tọa độ cho từng cluster
- `--max_clusters`: Số lượng cụm (mặc định: 5)
- `--save_model`: Lưu mô hình sau khi huấn luyện
- `--load_model`: Tải mô hình đã lưu
- `--model_path`: Đường dẫn để lưu/tải mô hình
- `--visualize`: Hiển thị biểu đồ kết quả
- `--save_results`: Lưu kết quả dự đoán

## So sánh các mô hình (evaluate_models.py)

Script này dùng để thử nghiệm và so sánh hiệu suất của nhiều mô hình khác nhau.

```bash
# So sánh nhanh các mô hình phổ biến
python evaluate_models.py --quick_test

# So sánh các mô hình cụ thể
python evaluate_models.py --classifier_types knn rf --predictor_types knn rf svr
```

Các tham số chính:
- `--classifier_types`: Danh sách các bộ phân loại cần thử nghiệm
- `--predictor_types`: Danh sách các bộ dự đoán cần thử nghiệm
- `--clustering_method`: Phương pháp phân cụm (`grid`, `kmeans`, `dbscan`)
- `--quick_test`: Chỉ đánh giá một số mô hình phổ biến
- `--save_results`: Lưu kết quả so sánh
- `--visualize`: Hiển thị biểu đồ so sánh

## Quy trình sử dụng đề xuất

1. Dùng `evaluate_models.py` để tìm tổ hợp mô hình tốt nhất:
```bash
python evaluate_models.py --classifier_types knn rf svm --predictor_types knn rf svr --save_results
```

2. Sau khi xác định được tổ hợp tốt nhất, dùng `regressor.py` để huấn luyện và lưu mô hình đó:
```bash
python regressor.py --classifier_type knn --global_predictor_type rf --cluster_predictor_type rf --save_model
```

## Các phương pháp được hỗ trợ

### Bộ phân loại cluster
- `knn`: K-Nearest Neighbors
- `rf`: Random Forest
- `svm`: Support Vector Machine
- `gb`: Gradient Boosting

### Bộ dự đoán tọa độ
- `knn`: K-Nearest Neighbors Regressor
- `rf`: Random Forest Regressor
- `svr`: Support Vector Regression
- `gb`: Gradient Boosting Regressor

## Kết quả và đầu ra

### Từ regressor.py
- Mô hình đã huấn luyện (nếu dùng --save_model)
- Kết quả dự đoán và đánh giá
- Biểu đồ trực quan (nếu dùng --visualize)

### Từ evaluate_models.py
- Bảng so sánh các mô hình (`model_comparison.csv`)
- Biểu đồ so sánh (`model_comparison.png`)
- Tổng hợp kết quả (`evaluation_results.json`)
- Thông tin về mô hình tốt nhất

