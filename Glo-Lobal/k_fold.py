import os
import sys
import argparse
import logging
import time
import numpy as np
import pandas as pd # Thêm pandas để xử lý dữ liệu nếu cần trong DataPreprocessor
import glob # Thêm glob để tìm file
import re # Thêm import re để sử dụng regex
from sklearn.model_selection import GroupKFold
import joblib # Nếu bạn cần lưu gì đó, ví dụ kết quả

# Thêm thư mục gốc vào sys.path để import các module tùy chỉnh
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) # Đi lên một cấp từ Glo-Lobal
sys.path.append(project_root)

from utils.cluster_regression import ClusterRegression

from config import N_COMPONENTS # Giả sử bạn có file config

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Định nghĩa lớp DataPreprocessor tại đây ---
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
        # Lưu ý: data_dir sẽ được truyền từ args vào các phương thức
        # self.data_dir = os.path.join('C:', os.sep, 'Users', 'Dell', 'DATN', 'IPS_CSI', 'data')

    def _process_coordinate(self, coord):
        """Hàm xử lý tọa độ để luôn trả về tuple (x, y) dạng số."""
        x, y = 0.0, 0.0
        coord_type = type(coord).__name__
        if isinstance(coord, np.ndarray):
            coord_str = str(coord.tolist()[:3]) if coord.size > 3 else str(coord.tolist())
        else:
             coord_str = str(coord[:3]) if hasattr(coord, '__len__') and len(coord) > 3 else str(coord)
        # logger.debug(f"Xử lý tọa độ gốc: {coord_str} (kiểu: {coord_type})")

        if isinstance(coord, (list, tuple, np.ndarray)):
            if len(coord) > 0:
                val_x = coord[0]
                if isinstance(val_x, (int, float, np.integer, np.floating)): x = float(val_x)
                elif isinstance(val_x, str):
                    try: x = float(val_x.split('x')[0]) if 'x' in val_x else float(val_x)
                    except ValueError: logger.warning(f"Không thể chuyển đổi tọa độ x '{val_x}', sử dụng 0.0"); x = 0.0
            if len(coord) > 1:
                val_y = coord[1]
                if isinstance(val_y, (int, float, np.integer, np.floating)): y = float(val_y)
                elif isinstance(val_y, str):
                    try: y = float(val_y.split('x')[0]) if 'x' in val_y else float(val_y)
                    except ValueError: logger.warning(f"Không thể chuyển đổi tọa độ y '{val_y}', sử dụng 0.0"); y = 0.0
        else:
            logger.warning(f"Tọa độ không phải list/tuple/array: {coord_str}, kiểu {coord_type}")

        if x == 0.0 and y == 0.0 and isinstance(coord, np.ndarray) and coord.size >= 2:
             try: # Cố gắng lấy trực tiếp nếu là numpy array và kết quả là 0,0
                 x = float(coord.flat[0])
                 y = float(coord.flat[1])
                 # logger.info(f"Đã lấy trực tiếp từ mảng numpy: [{x}, {y}] từ gốc {coord_str}")
             except (ValueError, IndexError): pass # Bỏ qua nếu không lấy được

        result = [float(x), float(y)]
        # logger.debug(f"Kết quả xử lý tọa độ: {result}")
        return result

    def load_data_for_groupkfold(self, data_dir, group_map=None, group_id_counter=None):
        """
        Tải dữ liệu từ một thư mục và tạo/cập nhật mảng groups dựa trên phần 'MxN' của tên file, sử dụng regex.
        """
        if group_map is None:
            group_map = {}
        if group_id_counter is None:
            group_id_counter = 0

        all_features_list = []
        all_coords_list = []
        groups_list = []

        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if not npz_files:
            logger.warning(f"Không tìm thấy file .npz trong {data_dir}")
            return np.array([]), np.array([]), np.array([]), group_map, group_id_counter

        logger.info(f"Đang xử lý {len(npz_files)} file từ {data_dir}...")

        # Biểu thức chính quy để tìm mẫu <số>x<số>
        # \d+ : một hoặc nhiều chữ số
        # x : chữ 'x'
        # (\d+x\d+) : tạo thành một nhóm khớp hoàn chỉnh
        location_pattern = re.compile(r"(\d+x\d+)")

        for file_path in npz_files:
            filename = os.path.basename(file_path)
            filename_no_ext = os.path.splitext(filename)[0] # Lấy tên file không có đuôi

            # --- SỬA ĐỔI LOGIC LẤY group_name BẰNG REGEX ---
            match = location_pattern.search(filename_no_ext)
            if match:
                group_name = match.group(1) # Lấy phần khớp được (ví dụ: '11x1', '7x9')
                logger.debug(f"File: {filename} -> Regex matched group name: {group_name}")
            else:
                # Fallback nếu không tìm thấy mẫu MxN
                logger.warning(f"Không tìm thấy mẫu MxN trong tên file '{filename}'. Sử dụng toàn bộ tên (trừ đuôi) làm nhóm.")
                group_name = filename_no_ext # Fallback
            # --- KẾT THÚC SỬA ĐỔI ---

            if group_name not in group_map:
                group_map[group_name] = group_id_counter
                group_id_counter += 1
            current_group_id = group_map[group_name]

            try:
                data = np.load(file_path)
                if 'csi_data' not in data or 'coords' not in data:
                    logger.warning(f"Bỏ qua file {filename} vì thiếu 'csi_data' hoặc 'coords'.")
                    continue

                csi_data = data['csi_data']
                coord_data_original = data['coords']

                if not isinstance(csi_data, np.ndarray) or csi_data.ndim < 2:
                     logger.warning(f"Dữ liệu CSI trong {filename} không hợp lệ. Bỏ qua.")
                     continue

                if csi_data.ndim == 3:
                    num_samples_in_file = csi_data.shape[0]
                    file_features = csi_data.reshape(num_samples_in_file, -1)
                elif csi_data.ndim == 2:
                    num_samples_in_file = csi_data.shape[0]
                    file_features = csi_data
                else:
                     logger.warning(f"Định dạng CSI không mong đợi ({csi_data.ndim}D) trong {filename}. Bỏ qua.")
                     continue

                file_coords = []
                coord_shape = coord_data_original.shape if hasattr(coord_data_original, 'shape') else None
                is_single_coord = False

                if coord_shape and len(coord_shape) > 0 and coord_shape[0] == 1 and num_samples_in_file > 1:
                    is_single_coord = True
                elif coord_shape is None or len(coord_shape) == 0 or (len(coord_shape) == 1 and coord_shape[0]!= num_samples_in_file and coord_shape[0]==2): # Kích thước (2,) hoặc không có shape rõ ràng
                     is_single_coord = True
                elif coord_shape and coord_shape[0] != num_samples_in_file:
                     logger.warning(f"Số lượng tọa độ ({coord_shape}) không khớp số mẫu CSI ({num_samples_in_file}) trong {filename}. Bỏ qua file.")
                     continue


                if is_single_coord:
                    single_coord = coord_data_original[0] if coord_shape and coord_shape[0] == 1 else coord_data_original
                    processed_coord = self._process_coordinate(single_coord)
                    file_coords = [processed_coord] * num_samples_in_file
                    # logger.debug(f"Sử dụng tọa độ đơn {processed_coord} cho {num_samples_in_file} mẫu trong {filename}")
                else: # Mỗi mẫu có tọa độ riêng
                    for i in range(num_samples_in_file):
                        processed_coord = self._process_coordinate(coord_data_original[i])
                        file_coords.append(processed_coord)


                all_features_list.append(file_features)
                all_coords_list.extend(file_coords)
                groups_list.extend([current_group_id] * num_samples_in_file)

            except Exception as e:
                logger.error(f"Lỗi khi xử lý file {filename}: {e}", exc_info=True)

        if not all_features_list:
             return np.array([]), np.array([]), np.array([]), group_map, group_id_counter

        final_features = np.concatenate(all_features_list, axis=0)
        final_coords = np.array(all_coords_list)
        final_groups = np.array(groups_list)

        if final_features.shape[0] != final_coords.shape[0] or final_features.shape[0] != final_groups.shape[0]:
            logger.error("Lỗi không khớp số lượng mẫu cuối cùng!")
            logger.error(f"Shapes: Features={final_features.shape}, Coords={final_coords.shape}, Groups={final_groups.shape}")
            return np.array([]), np.array([]), np.array([]), group_map, group_id_counter

        return final_features, final_coords, final_groups, group_map, group_id_counter

# --- Hàm main được sửa đổi để gộp train và test ---
def main(args):
    """Hàm chính để huấn luyện và đánh giá bằng GroupKFold trên dữ liệu gộp."""
    start_time = time.time()
    logger.info("Bắt đầu quá trình huấn luyện và đánh giá bằng GroupKFold (gộp train/test).")
    logger.info(f"Các tham số đầu vào: {vars(args)}")

    # --- 1. Tải dữ liệu từ train và test, gộp lại ---
    processor = DataPreprocessor(random_state=args.random_state)
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')

    # Khởi tạo group map và counter chung
    global_group_map = {}
    global_group_id_counter = 0

    all_X_list = []
    all_coords_list = []
    all_groups_list = []

    logger.info(f"Đang tải dữ liệu từ thư mục train: {train_dir}")
    try:
        X_train, coords_train, groups_train, global_group_map, global_group_id_counter = processor.load_data_for_groupkfold(
            train_dir, global_group_map, global_group_id_counter
        )
        if X_train.size > 0:
            all_X_list.append(X_train)
            all_coords_list.append(coords_train)
            all_groups_list.append(groups_train)
            logger.info(f"  Đã tải {len(X_train)} mẫu từ train.")
        else:
             logger.warning("Không tải được dữ liệu từ thư mục train.")

    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu train: {e}", exc_info=True)
        # Có thể quyết định dừng hoặc tiếp tục chỉ với dữ liệu test

    logger.info(f"Đang tải dữ liệu từ thư mục test: {test_dir}")
    try:
        # Tiếp tục sử dụng và cập nhật global_group_map và counter
        X_test, coords_test, groups_test, global_group_map, global_group_id_counter = processor.load_data_for_groupkfold(
            test_dir, global_group_map, global_group_id_counter
        )
        if X_test.size > 0:
            all_X_list.append(X_test)
            all_coords_list.append(coords_test)
            all_groups_list.append(groups_test)
            logger.info(f"  Đã tải {len(X_test)} mẫu từ test.")
        else:
             logger.warning("Không tải được dữ liệu từ thư mục test.")

    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu test: {e}", exc_info=True)

    # Kiểm tra nếu không có dữ liệu nào được tải
    if not all_X_list:
        logger.error("Không thể tải được bất kỳ dữ liệu nào từ train hoặc test. Dừng.")
        return

    # Nối dữ liệu từ train và test
    X_all = np.concatenate(all_X_list, axis=0)
    coords_all = np.concatenate(all_coords_list, axis=0)
    groups_all = np.concatenate(all_groups_list, axis=0)

    logger.info("-" * 30)
    logger.info(f"Tổng cộng dữ liệu gộp: {len(X_all)} mẫu.")
    logger.info(f"Số lượng nhóm (vị trí) duy nhất tổng cộng: {len(np.unique(groups_all))}")
    logger.info(f"Ánh xạ Tên Nhóm -> ID: {global_group_map}")
    logger.info("-" * 30)

    # --- THÊM BƯỚC LẤY SUBSET ---
    if args.subset_ratio < 1.0 and len(X_all) > 0:
        num_subset_samples = int(len(X_all) * args.subset_ratio)
        if num_subset_samples == 0 and len(X_all) > 0: # Đảm bảo lấy ít nhất 1 mẫu nếu có thể
            num_subset_samples = 1
        logger.info(f"Áp dụng subset_ratio={args.subset_ratio}, lấy {num_subset_samples}/{len(X_all)} mẫu ngẫu nhiên.")
        # Lấy mẫu ngẫu nhiên
        subset_indices = np.random.choice(len(X_all), num_subset_samples, replace=False)
        X_all = X_all[subset_indices]
        coords_all = coords_all[subset_indices]
        groups_all = groups_all[subset_indices]
        logger.info(f"Dữ liệu sau khi lấy subset: {len(X_all)} mẫu.")
        # Cảnh báo nếu số nhóm giảm đáng kể
        n_groups_after_subset = len(np.unique(groups_all))
        logger.info(f"Số nhóm duy nhất sau khi lấy subset: {n_groups_after_subset}")
        if n_groups_after_subset < args.n_splits:
             logger.warning(f"CẢNH BÁO: Số nhóm duy nhất ({n_groups_after_subset}) sau khi lấy subset nhỏ hơn số fold ({args.n_splits}). Có thể gây lỗi hoặc kết quả không đáng tin cậy.")

    elif len(X_all) == 0:
         logger.error("Không có dữ liệu để xử lý sau khi gộp. Dừng.")
         return
    else:
        logger.info("Sử dụng toàn bộ dữ liệu gộp (subset_ratio=1.0).")
    # --- KẾT THÚC BƯỚC LẤY SUBSET ---

    # --- Các bước 2-6 sử dụng dữ liệu đã gộp và có thể đã subset (X_all, coords_all, groups_all) ---

    # Kiểm tra số lượng fold hợp lệ
    n_unique_groups = len(np.unique(groups_all))
    if n_unique_groups == 0:
         logger.error("Không có nhóm nào được tạo ra từ dữ liệu.")
         return
    if args.n_splits > n_unique_groups:
        logger.warning(f"Số lượng fold ({args.n_splits}) lớn hơn số nhóm duy nhất ({n_unique_groups}). Đặt n_splits = {n_unique_groups}")
        args.n_splits = n_unique_groups

    if args.n_splits < 2:
        logger.error("Số lượng fold phải ít nhất là 2.")
        return

    # --- 2. Khởi tạo GroupKFold ---
    gkf = GroupKFold(n_splits=args.n_splits)
    logger.info(f"Sử dụng GroupKFold với {args.n_splits} folds trên dữ liệu gộp.")

    # --- 3. Vòng lặp Cross-Validation ---
    all_fold_metrics = []
    # Sử dụng X_all, coords_all, groups_all để chia fold
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, coords_all, groups=groups_all)):
        fold_start_time = time.time()
        logger.info(f"--- Bắt đầu Fold {fold + 1}/{args.n_splits} ---")

        X_train_fold, X_val_fold = X_all[train_idx], X_all[val_idx]
        coords_train_fold, coords_val_fold = coords_all[train_idx], coords_all[val_idx]

        logger.info(f"  Số mẫu huấn luyện: {len(X_train_fold)}")
        logger.info(f"  Số mẫu kiểm tra: {len(X_val_fold)}")
        # In ra các nhóm trong tập train/val của fold này để kiểm tra
        train_groups_in_fold = np.unique(groups_all[train_idx])
        val_groups_in_fold = np.unique(groups_all[val_idx])
        logger.info(f"  Nhóm huấn luyện (IDs): {train_groups_in_fold}")
        logger.info(f"  Nhóm kiểm tra (IDs): {val_groups_in_fold}")
        assert len(np.intersect1d(train_groups_in_fold, val_groups_in_fold)) == 0, "Lỗi: Nhóm bị trùng giữa train và validation!"


        # --- 4. Khởi tạo và Huấn luyện Mô hình cho Fold này ---
        logger.info("  Đang khởi tạo mô hình ClusterRegression...")
        model = ClusterRegression(
            max_clusters=args.n_clusters,
            classifier_type=args.classifier,
            global_predictor_type=args.global_predictor,
            cluster_predictor_type=args.cluster_predictor,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state + fold, # Thêm fold vào random_state để có sự khác biệt nhỏ
            use_kernel_pca=True,
            clustering_method=args.clustering_method
        )

        logger.info("  Đang huấn luyện mô hình...")
        try:
            model.fit(X_train_fold, coords=coords_train_fold)
            logger.info("  Huấn luyện hoàn tất.")
        except Exception as e:
            logger.error(f"  Lỗi trong quá trình huấn luyện fold {fold + 1}: {e}", exc_info=True)
            continue

        # --- 5. Đánh giá Mô hình trên Fold này ---
        logger.info("  Đang đánh giá mô hình...")
        try:
            metrics = model.evaluate(X_val_fold, coords_val_fold)
            if metrics:
                logger.info(f"  Kết quả Fold {fold + 1}: {metrics}")
                all_fold_metrics.append(metrics)
            else:
                logger.warning(f"  Đánh giá Fold {fold + 1} không trả về kết quả.")
        except Exception as e:
            logger.error(f"  Lỗi trong quá trình đánh giá fold {fold + 1}: {e}", exc_info=True)

        fold_end_time = time.time()
        logger.info(f"--- Kết thúc Fold {fold + 1}/{args.n_splits} (Thời gian: {fold_end_time - fold_start_time:.2f} giây) ---")

    # --- 6. Tổng hợp và Hiển thị Kết quả ---
    logger.info("=" * 30)
    logger.info("Kết quả Tổng hợp Cross-Validation")
    if not all_fold_metrics:
        logger.warning("Không có kết quả nào được ghi lại từ các fold.")
    else:
        mean_distances = [m['mean_distance'] for m in all_fold_metrics if m and 'mean_distance' in m]
        median_distances = [m['median_distance'] for m in all_fold_metrics if m and 'median_distance' in m]

        if mean_distances:
            avg_mean_dist = np.mean(mean_distances)
            std_mean_dist = np.std(mean_distances)
            logger.info(f"Sai số khoảng cách trung bình (Mean Distance): {avg_mean_dist:.2f} +/- {std_mean_dist:.2f} cm")

        if median_distances:
            avg_median_dist = np.mean(median_distances)
            std_median_dist = np.std(median_distances)
            logger.info(f"Sai số khoảng cách trung vị (Median Distance): {avg_median_dist:.2f} +/- {std_median_dist:.2f} cm")

    end_time = time.time()
    logger.info(f"Tổng thời gian thực thi: {end_time - start_time:.2f} giây")
    logger.info("Hoàn tất.")


# --- Phần parser được cập nhật ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá mô hình Cluster Regression bằng GroupKFold CV trên dữ liệu gộp train/test.")
    # Sửa đổi help text cho data_dir
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn đến thư mục cha chứa hai thư mục con "train" và "test" chứa dữ liệu .npz.')
    parser.add_argument('--n_splits', type=int, default=5, help='Số lượng fold cho GroupKFold.')
    parser.add_argument('--n_clusters', type=int, default=10, help='Số lượng cụm tối đa.')
    parser.add_argument('--classifier', type=str, default='knn', help='Loại mô hình phân loại cụm.')
    parser.add_argument('--global_predictor', type=str, default='knn', help='Loại mô hình dự đoán toàn cục.')
    parser.add_argument('--cluster_predictor', type=str, default='knn', help='Loại mô hình dự đoán cho từng cụm.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Số neighbors cho KNN.')
    parser.add_argument('--n_components', type=int, default=None, help=f'Số thành phần cho Kernel PCA (mặc định lấy từ config: {N_COMPONENTS}).')
    parser.add_argument('--clustering_method', type=str, default='grid', choices=['grid', 'kmeans', 'dbscan'], help='Phương pháp phân cụm tọa độ.')
    parser.add_argument('--random_state', type=int, default=42, help='Seed cơ sở cho các quá trình ngẫu nhiên.')
    parser.add_argument('--subset_ratio', type=float, default=1.0, help='Tỷ lệ dữ liệu gộp sẽ sử dụng cho KFold (ví dụ: 0.01 cho 1%). Mặc định là 1.0 (toàn bộ).')
    args = parser.parse_args()

    # Gán lại n_components nếu người dùng không cung cấp để tránh lỗi None
    if args.n_components is None:
        args.n_components = N_COMPONENTS

    main(args)
