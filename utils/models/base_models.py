from abc import ABC, abstractmethod
import numpy as np

class BaseClusterClassifier(ABC):
    """Lớp cơ sở cho tất cả các bộ phân loại cluster"""
    
    @abstractmethod
    def train(self, X, cluster_assignments):
        """Huấn luyện bộ phân loại
        
        Parameters
        ----------
        X : numpy.ndarray
            Dữ liệu đầu vào
        cluster_assignments : numpy.ndarray
            Nhãn cluster
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def save(self, path):
        """Lưu mô hình
        
        Parameters
        ----------
        path : str
            Đường dẫn lưu mô hình
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        BaseClusterClassifier
            Instance của lớp với mô hình đã tải
        """
        pass

class BaseCoordinatePredictor(ABC):
    """Lớp cơ sở cho tất cả các bộ dự đoán tọa độ"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def save(self, path):
        """Lưu mô hình
        
        Parameters
        ----------
        path : str
            Đường dẫn lưu mô hình
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path):
        """Tải mô hình từ file
        
        Parameters
        ----------
        path : str
            Đường dẫn tới file mô hình
            
        Returns
        -------
        BaseCoordinatePredictor
            Instance của lớp với mô hình đã tải
        """
        pass 