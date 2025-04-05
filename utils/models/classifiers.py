import os, joblib, logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from .base_models import BaseClusterClassifier

logger = logging.getLogger(__name__)

class KNNClusterClassifier(BaseClusterClassifier):
    """Sử dụng KNeighborsClassifier để phân loại cluster"""
    
    def __init__(self, n_neighbors=5):
        """
        Parameters
        ----------
        n_neighbors : int, optional
            Số lượng neighbors, mặc định là 5
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.name = "KNN"
        
    def train(self, X, y):
        """Huấn luyện bộ phân loại
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Nhãn cluster
        """
        self.model.fit(X, y)
    
    def fit(self, X, y):
        """Phương thức fit() cho tương thích với scikit-learn API
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Nhãn cluster
        """
        return self.train(X, y)
        
    def predict(self, X):
        """Dự đoán cluster
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Nhãn cluster được dự đoán
        """
        # Đảm bảo X có định dạng đúng
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
            Đường dẫn lưu mô hình
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        KNNClusterClassifier
            Instance của lớp với mô hình đã tải
        """
        instance = cls()
        instance.model = joblib.load(path)
        return instance

class RandomForestClusterClassifier(BaseClusterClassifier):
    """Sử dụng RandomForestClassifier để phân loại cluster"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Parameters
        ----------
        n_estimators : int, optional
            Số lượng cây, mặc định là 100
        max_depth : int, optional
            Độ sâu tối đa của cây, mặc định là None
        random_state : int, optional
            Seed ngẫu nhiên, mặc định là 42
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.name = "RandomForest"
        
    def train(self, X, y):
        """Huấn luyện bộ phân loại
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Nhãn cluster
        """
        self.model.fit(X, y)
    
    def fit(self, X, y):
        """Phương thức fit() cho tương thích với scikit-learn API
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Nhãn cluster
        """
        return self.train(X, y)
        
    def predict(self, X):
        """Dự đoán cluster
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Nhãn cluster được dự đoán
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")
 
        # Đảm bảo X có định dạng đúng
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
            Đường dẫn lưu mô hình
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        RandomForestClusterClassifier
            Instance của lớp với mô hình đã tải
        """
        instance = cls()
        instance.model = joblib.load(path)
        return instance

class SVMClusterClassifier(BaseClusterClassifier):
    """Sử dụng SVM để phân loại cluster"""
    
    def __init__(self, C=1.0, kernel='rbf', random_state=42):
        """
        Parameters
        ----------
        C : float, optional
            Tham số điều chỉnh, mặc định là 1.0
        kernel : str, optional
            Kernel function, mặc định là 'rbf'
        random_state : int, optional
            Seed ngẫu nhiên, mặc định là 42
        """
        self.model = SVC(C=C, kernel=kernel, random_state=random_state)
        self.name = "SVM"
        
    def train(self, X, y):
        """Huấn luyện bộ phân loại
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Nhãn cluster
        """
        self.model.fit(X, y)
    
    def fit(self, X, y):
        """Phương thức fit() cho tương thích với scikit-learn API
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Nhãn cluster
        """
        return self.train(X, y)
        
    def predict(self, X):
        """Dự đoán cluster
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Nhãn cluster được dự đoán
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")
 
        # Đảm bảo X có định dạng đúng
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
            Đường dẫn lưu mô hình
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        SVMClusterClassifier
            Instance của lớp với mô hình đã tải
        """
        instance = cls()
        instance.model = joblib.load(path)
        return instance

class GBClusterClassifier(BaseClusterClassifier):
    """Sử dụng Gradient Boosting để phân loại cluster"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Parameters
        ----------
        n_estimators : int, optional
            Số lượng cây, mặc định là 100
        learning_rate : float, optional
            Tốc độ học, mặc định là 0.1
        max_depth : int, optional
            Độ sâu tối đa của cây, mặc định là 3
        random_state : int, optional
            Seed ngẫu nhiên, mặc định là 42
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.name = "GradientBoosting"
        
    def train(self, X, cluster_assignments):
        """Huấn luyện bộ phân loại
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        cluster_assignments : numpy.ndarray
            Nhãn cluster
        """
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X, cluster_assignments)
    
    def fit(self, X, y):
        """Phương thức fit() cho tương thích với scikit-learn API
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        y : numpy.ndarray
            Nhãn cluster
        """
        return self.train(X, y)
        
    def predict(self, X):
        """Dự đoán cluster
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Nhãn cluster được dự đoán
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")
 
        # Đảm bảo X có định dạng đúng
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Bỏ qua cảnh báo về feature names
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self.model.predict(X)
        
    def predict_proba(self, X):
        """Dự đoán xác suất thuộc về các cluster
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
            
        Returns
        -------
        numpy.ndarray
            Ma trận xác suất thuộc về từng cluster
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")
            
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        return self.model.predict_proba(X)
        
    def save(self, path):
        """Lưu mô hình
        
        Parameters
        ----------
        path : str
            Đường dẫn thư mục lưu mô hình
        """
        model_path = os.path.join(path, "gb_classifier_model.pkl")
        
        # Lưu model và thông tin
        model_info = {
            'model': self.model,
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
        GBClusterClassifier
            Instance của lớp với mô hình đã tải
        """
        model_info = joblib.load(path)
        instance = cls(
            n_estimators=model_info['n_estimators'],
            learning_rate=model_info['learning_rate'],
            max_depth=model_info['max_depth'],
            random_state=model_info['random_state']
        )
        instance.model = model_info['model']
        return instance

try:
    import xgboost as xgb
    
    class XGBClusterClassifier(BaseClusterClassifier):
        """Sử dụng XGBoost để phân loại cluster"""
        
        def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
            """
            Parameters
            ----------
            n_estimators : int, optional
                Số lượng cây, mặc định là 100
            learning_rate : float, optional
                Tốc độ học, mặc định là 0.1
            max_depth : int, optional
                Độ sâu tối đa của cây, mặc định là 3
            random_state : int, optional
                Seed ngẫu nhiên, mặc định là 42
            """
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.random_state = random_state
            self.model = None
            self.name = "XGBoost"
            
        def train(self, X, cluster_assignments):
            """Huấn luyện bộ phân loại
            
            Parameters
            ----------
            X : numpy.ndarray
                Dữ liệu đầu vào
            cluster_assignments : numpy.ndarray
                Nhãn cluster
            """
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.model.fit(X, cluster_assignments)
            
        def fit(self, X, y):
            """Phương thức fit() cho tương thích với scikit-learn API
            
            Parameters
            ----------
            X : numpy.ndarray
                Dữ liệu đầu vào
            y : numpy.ndarray
                Nhãn cluster
            """
            return self.train(X, y)
            
        def predict(self, X):
            """Dự đoán cluster
            
            Parameters
            ----------
            X : numpy.ndarray
                Dữ liệu đầu vào
                
            Returns
            -------
            numpy.ndarray
                Nhãn cluster được dự đoán
            """
            if self.model is None:
                raise ValueError("Mô hình chưa được huấn luyện")
                
            # Đảm bảo X có định dạng đúng
            if isinstance(X, np.ndarray) and len(X.shape) == 1:
                X = X.reshape(1, -1)
                
            # Bỏ qua cảnh báo về feature names
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return self.model.predict(X)
            
        def predict_proba(self, X):
            """Dự đoán xác suất thuộc về các cluster
            
            Parameters
            ----------
            X : numpy.ndarray
                Dữ liệu đầu vào
                
            Returns
            -------
            numpy.ndarray
                Ma trận xác suất thuộc về từng cluster
            """
            if self.model is None:
                raise ValueError("Mô hình chưa được huấn luyện")
                
            if isinstance(X, np.ndarray) and len(X.shape) == 1:
                X = X.reshape(1, -1)
                
            return self.model.predict_proba(X)
            
        def save(self, path):
            """Lưu mô hình
            
            Parameters
            ----------
            path : str
                Đường dẫn thư mục lưu mô hình
            """
            model_path = os.path.join(path, "xgb_classifier_model.pkl")
            
            # Lưu model và thông tin
            model_info = {
                'model': self.model,
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
            XGBClusterClassifier
                Instance của lớp với mô hình đã tải
            """
            model_info = joblib.load(path)
            instance = cls(
                n_estimators=model_info['n_estimators'],
                learning_rate=model_info['learning_rate'],
                max_depth=model_info['max_depth'],
                random_state=model_info['random_state']
            )
            instance.model = model_info['model']
            return instance
except ImportError:
    logger.warning("Không thể import xgboost. XGBClusterClassifier sẽ không được sử dụng.") 