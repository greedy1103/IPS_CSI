import os, pickle, logging
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from .base_models import BaseCoordinatePredictor

logger = logging.getLogger(__name__)

class KNNCoordinatePredictor(BaseCoordinatePredictor):
    """Sử dụng KNN để dự đoán tọa độ"""
    
    def __init__(self, cluster_id=None, n_neighbors=5, auto_tune=False, cv=5):
        """
        Parameters
        ----------
        cluster_id : int or None, optional
            ID của cluster, None nếu là mô hình toàn cục
        n_neighbors : int, optional
            Số lượng neighbors, mặc định là 5
        auto_tune : bool, optional
            Tự động tìm n_neighbors tối ưu, mặc định là False
        cv : int, optional
            Số fold cho cross validation, mặc định là 5
        """
        self.cluster_id = cluster_id
        self.n_neighbors = n_neighbors
        self.auto_tune = auto_tune
        self.cv = cv
        self.model = None
        self.name = "KNN"
        
    def train(self, X, dummy_labels, coords):
        """Huấn luyện mô hình dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        dummy_labels : numpy.ndarray
            Nhãn giả (không dùng trong dự đoán)
        coords : numpy.ndarray
            Tọa độ thực tế
        """
        if self.auto_tune:
            # Tìm n_neighbors tối ưu
            param_grid = {'n_neighbors': range(1, 21)}
            grid_search = GridSearchCV(
                KNeighborsRegressor(),
                param_grid,
                cv=self.cv,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X, coords)
            best_n_neighbors = grid_search.best_params_['n_neighbors']
            self.n_neighbors = best_n_neighbors
            self.model = grid_search.best_estimator_
        else:
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            self.model.fit(X, coords)
        
    def predict(self, X):
        """Dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Tọa độ được dự đoán
        """
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Bỏ qua cảnh báo về feature names
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self.model.predict(X)
        
    def save(self, path):
        """Lưu mô hình
        
        Parameters
        ----------
        path : str
            Đường dẫn thư mục lưu mô hình
        """
        if self.cluster_id is None:
            model_path = os.path.join(path, "global_knn_model.pkl")
        else:
            model_path = os.path.join(path, f"cluster_{self.cluster_id}_knn_model.pkl")
        
        # Lưu model và thông tin
        model_info = {
            'model': self.model,
            'cluster_id': self.cluster_id,
            'n_neighbors': self.n_neighbors,
            'auto_tune': self.auto_tune,
            'cv': self.cv
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_info, model_path)
    
    @classmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        KNNCoordinatePredictor
            Instance của lớp với mô hình đã tải
        """
        model_info = joblib.load(path)
        instance = cls(
            cluster_id=model_info['cluster_id'],
            n_neighbors=model_info['n_neighbors'],
            auto_tune=model_info['auto_tune'],
            cv=model_info['cv']
        )
        instance.model = model_info['model']
        return instance

class RFCoordinatePredictor(BaseCoordinatePredictor):
    """Sử dụng Random Forest để dự đoán tọa độ"""
    
    def __init__(self, cluster_id=None, n_estimators=100, max_depth=None, random_state=42):
        """
        Parameters
        ----------
        cluster_id : int or None, optional
            ID của cluster, None nếu là mô hình toàn cục
        n_estimators : int, optional
            Số lượng cây, mặc định là 100
        max_depth : int, optional
            Độ sâu tối đa của cây, mặc định là None
        random_state : int, optional
            Seed ngẫu nhiên, mặc định là 42
        """
        self.cluster_id = cluster_id
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.name = "RandomForest"
        
    def train(self, X, dummy_labels, coords):
        """Huấn luyện mô hình dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        dummy_labels : numpy.ndarray
            Nhãn giả (không dùng trong dự đoán)
        coords : numpy.ndarray
            Tọa độ thực tế
        """
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X, coords)
        
    def predict(self, X):
        """Dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Tọa độ được dự đoán
        """
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Bỏ qua cảnh báo về feature names
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self.model.predict(X)
        
    def save(self, path):
        """Lưu mô hình
        
        Parameters
        ----------
        path : str
            Đường dẫn thư mục lưu mô hình
        """
        if self.cluster_id is None:
            model_path = os.path.join(path, "global_rf_model.pkl")
        else:
            model_path = os.path.join(path, f"cluster_{self.cluster_id}_rf_model.pkl")
        
        # Lưu model và thông tin
        model_info = {
            'model': self.model,
            'cluster_id': self.cluster_id,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_info, model_path)
    
    @classmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        RFCoordinatePredictor
            Instance của lớp với mô hình đã tải
        """
        model_info = joblib.load(path)
        instance = cls(
            cluster_id=model_info['cluster_id'],
            n_estimators=model_info['n_estimators'],
            max_depth=model_info['max_depth'],
            random_state=model_info['random_state']
        )
        instance.model = model_info['model']
        return instance

class SVRCoordinatePredictor(BaseCoordinatePredictor):
    """Sử dụng SVR để dự đoán tọa độ"""
    
    def __init__(self, cluster_id=None, C=1.0, epsilon=0.1, kernel='rbf'):
        """
        Parameters
        ----------
        cluster_id : int or None, optional
            ID của cluster, None nếu là mô hình toàn cục
        C : float, optional
            Tham số điều chỉnh, mặc định là 1.0
        epsilon : float, optional
            Tham số epsilon trong hàm mất mát, mặc định là 0.1
        kernel : str, optional
            Kernel function, mặc định là 'rbf'
        """
        self.cluster_id = cluster_id
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.model = None
        self.name = "SVR"
        
    def train(self, X, dummy_labels, coords):
        """Huấn luyện mô hình dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        dummy_labels : numpy.ndarray
            Nhãn giả (không dùng trong dự đoán)
        coords : numpy.ndarray
            Tọa độ thực tế
        """
        # Huấn luyện SVR cho mỗi tọa độ (x, y)
        n_coordinates = coords.shape[1]
        self.model = []
        
        for i in range(n_coordinates):
            regressor = SVR(
                C=self.C,
                epsilon=self.epsilon,
                kernel=self.kernel
            )
            regressor.fit(X, coords[:, i])
            self.model.append(regressor)
        
    def predict(self, X):
        """Dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Tọa độ được dự đoán
        """
        if self.model is None or len(self.model) == 0:
            raise ValueError("Mô hình chưa được huấn luyện")
            
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Dự đoán từng tọa độ
        n_samples = X.shape[0]
        n_coordinates = len(self.model)
        predictions = np.zeros((n_samples, n_coordinates))
        
        # Bỏ qua cảnh báo về feature names
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            for i in range(n_coordinates):
                predictions[:, i] = self.model[i].predict(X)
        
        return predictions
        
    def save(self, path):
        """Lưu mô hình
        
        Parameters
        ----------
        path : str
            Đường dẫn thư mục lưu mô hình
        """
        if self.cluster_id is None:
            model_path = os.path.join(path, "global_svr_model.pkl")
        else:
            model_path = os.path.join(path, f"cluster_{self.cluster_id}_svr_model.pkl")
        
        # Lưu model và thông tin
        model_info = {
            'model': self.model,
            'cluster_id': self.cluster_id,
            'C': self.C,
            'epsilon': self.epsilon,
            'kernel': self.kernel
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_info, model_path)
    
    @classmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        SVRCoordinatePredictor
            Instance của lớp với mô hình đã tải
        """
        model_info = joblib.load(path)
        instance = cls(
            cluster_id=model_info['cluster_id'],
            C=model_info['C'],
            epsilon=model_info['epsilon'],
            kernel=model_info['kernel']
        )
        instance.model = model_info['model']
        return instance

class GBCoordinatePredictor(BaseCoordinatePredictor):
    """Sử dụng Gradient Boosting để dự đoán tọa độ"""
    
    def __init__(self, cluster_id=None, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Parameters
        ----------
        cluster_id : int or None, optional
            ID của cluster, None nếu là mô hình toàn cục
        n_estimators : int, optional
            Số lượng cây, mặc định là 100
        learning_rate : float, optional
            Tốc độ học, mặc định là 0.1
        max_depth : int, optional
            Độ sâu tối đa của cây, mặc định là 3
        random_state : int, optional
            Seed ngẫu nhiên, mặc định là 42
        """
        self.cluster_id = cluster_id
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.name = "GradientBoosting"
        
    def train(self, X, dummy_labels, coords):
        """Huấn luyện mô hình dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        dummy_labels : numpy.ndarray
            Nhãn giả (không dùng trong dự đoán)
        coords : numpy.ndarray
            Tọa độ thực tế
        """
        # Huấn luyện GB cho mỗi tọa độ (x, y)
        n_coordinates = coords.shape[1]
        self.model = []
        
        for i in range(n_coordinates):
            regressor = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            regressor.fit(X, coords[:, i])
            self.model.append(regressor)
        
    def predict(self, X):
        """Dự đoán tọa độ
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Tọa độ được dự đoán
        """
        if self.model is None or len(self.model) == 0:
            raise ValueError("Mô hình chưa được huấn luyện")
            
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Dự đoán từng tọa độ
        n_samples = X.shape[0]
        n_coordinates = len(self.model)
        predictions = np.zeros((n_samples, n_coordinates))
        
        # Bỏ qua cảnh báo về feature names
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            for i in range(n_coordinates):
                predictions[:, i] = self.model[i].predict(X)
        
        return predictions
        
    def save(self, path):
        """Lưu mô hình
        
        Parameters
        ----------
        path : str
            Đường dẫn thư mục lưu mô hình
        """
        if self.cluster_id is None:
            model_path = os.path.join(path, "global_gb_model.pkl")
        else:
            model_path = os.path.join(path, f"cluster_{self.cluster_id}_gb_model.pkl")
        
        # Lưu model và thông tin
        model_info = {
            'model': self.model,
            'cluster_id': self.cluster_id,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_info, model_path)
    
    @classmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        GBCoordinatePredictor
            Instance của lớp với mô hình đã tải
        """
        model_info = joblib.load(path)
        instance = cls(
            cluster_id=model_info['cluster_id'],
            n_estimators=model_info['n_estimators'],
            learning_rate=model_info['learning_rate'],
            max_depth=model_info['max_depth'],
            random_state=model_info['random_state']
        )
        instance.model = model_info['model']
        return instance 