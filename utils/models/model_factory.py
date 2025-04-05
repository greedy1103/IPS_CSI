import logging
from .classifiers import KNNClusterClassifier, RandomForestClusterClassifier, SVMClusterClassifier, GBClusterClassifier
from .predictors import KNNCoordinatePredictor, RFCoordinatePredictor, SVRCoordinatePredictor, GBCoordinatePredictor

logger = logging.getLogger(__name__)

def create_classifier(classifier_type, **kwargs):
    """
    Tạo bộ phân loại dựa vào loại và tham số
    
    Parameters
    ----------
    classifier_type : str
        Loại phân loại, là một trong: 'knn', 'rf', 'svm', 'gb', 'xgb'
    **kwargs : dict
        Tham số cho bộ phân loại
        
    Returns
    -------
    BaseClusterClassifier
        Bộ phân loại được tạo
        
    Raises
    ------
    ValueError
        Nếu loại phân loại không được hỗ trợ
    """
    logger.info(f"Tạo bộ phân loại cluster loại: {classifier_type}")
    
    # Lọc các tham số phù hợp cho từng loại phân loại
    filtered_kwargs = {}
    
    if classifier_type.lower() == 'knn':
        # KNN chỉ chấp nhận n_neighbors
        if 'n_neighbors' in kwargs:
            filtered_kwargs['n_neighbors'] = kwargs['n_neighbors']
        return KNNClusterClassifier(**filtered_kwargs)
    
    elif classifier_type.lower() == 'rf':
        # RandomForest chấp nhận n_estimators, max_depth và random_state
        if 'random_state' in kwargs:
            filtered_kwargs['random_state'] = kwargs['random_state']
        if 'n_estimators' in kwargs:
            filtered_kwargs['n_estimators'] = kwargs['n_estimators']
        if 'max_depth' in kwargs:
            filtered_kwargs['max_depth'] = kwargs['max_depth']
        return RandomForestClusterClassifier(**filtered_kwargs)
    
    elif classifier_type.lower() == 'svm':
        # SVM chấp nhận C, kernel, và random_state
        if 'random_state' in kwargs:
            filtered_kwargs['random_state'] = kwargs['random_state']
        if 'C' in kwargs:
            filtered_kwargs['C'] = kwargs['C']
        if 'kernel' in kwargs:
            filtered_kwargs['kernel'] = kwargs['kernel']
        return SVMClusterClassifier(**filtered_kwargs)
    
    elif classifier_type.lower() == 'gb':
        # Gradient Boosting chấp nhận n_estimators, learning_rate, max_depth và random_state
        if 'random_state' in kwargs:
            filtered_kwargs['random_state'] = kwargs['random_state']
        if 'n_estimators' in kwargs:
            filtered_kwargs['n_estimators'] = kwargs['n_estimators']
        if 'learning_rate' in kwargs:
            filtered_kwargs['learning_rate'] = kwargs['learning_rate']
        if 'max_depth' in kwargs:
            filtered_kwargs['max_depth'] = kwargs['max_depth']
        return GBClusterClassifier(**filtered_kwargs)
    
    elif classifier_type.lower() == 'xgb':
        try:
            from .classifiers import XGBClusterClassifier
            # XGBoost chấp nhận n_estimators, learning_rate, max_depth và random_state
            if 'random_state' in kwargs:
                filtered_kwargs['random_state'] = kwargs['random_state']
            if 'n_estimators' in kwargs:
                filtered_kwargs['n_estimators'] = kwargs['n_estimators']
            if 'learning_rate' in kwargs:
                filtered_kwargs['learning_rate'] = kwargs['learning_rate']
            if 'max_depth' in kwargs:
                filtered_kwargs['max_depth'] = kwargs['max_depth']
            return XGBClusterClassifier(**filtered_kwargs)
        except ImportError:
            logger.error("Không thể import XGBClusterClassifier. Xin hãy cài đặt xgboost.")
            raise ValueError("XGBoost không được cài đặt. Xin hãy cài đặt xgboost để sử dụng XGBClusterClassifier.")
    
    else:
        raise ValueError(f"Không hỗ trợ loại phân loại: {classifier_type}")

def create_predictor(predictor_type, **kwargs):
    """
    Tạo bộ dự đoán tọa độ dựa vào loại và tham số
    
    Parameters
    ----------
    predictor_type : str
        Loại dự đoán, là một trong: 'knn', 'rf', 'svr', 'gb'
    **kwargs : dict
        Tham số cho bộ dự đoán
        
    Returns
    -------
    BaseCoordinatePredictor
        Bộ dự đoán được tạo
        
    Raises
    ------
    ValueError
        Nếu loại dự đoán không được hỗ trợ
    """
    logger.info(f"Tạo bộ dự đoán tọa độ loại: {predictor_type}")
    
    # Lọc các tham số phù hợp cho từng loại dự đoán
    filtered_kwargs = {}
    
    # Trích xuất cluster_id nếu có
    if 'cluster_id' in kwargs:
        filtered_kwargs['cluster_id'] = kwargs['cluster_id']
    
    if predictor_type.lower() == 'knn':
        # KNN chỉ chấp nhận n_neighbors, auto_tune và cv
        if 'n_neighbors' in kwargs:
            filtered_kwargs['n_neighbors'] = kwargs['n_neighbors']
        if 'auto_tune' in kwargs:
            filtered_kwargs['auto_tune'] = kwargs['auto_tune']
        if 'cv' in kwargs:
            filtered_kwargs['cv'] = kwargs['cv']
        # Không truyền random_state cho KNN
        return KNNCoordinatePredictor(**filtered_kwargs)
    
    elif predictor_type.lower() == 'rf':
        # RandomForest chấp nhận random_state, n_estimators và max_depth
        if 'random_state' in kwargs:
            filtered_kwargs['random_state'] = kwargs['random_state']
        if 'n_estimators' in kwargs:
            filtered_kwargs['n_estimators'] = kwargs['n_estimators']
        if 'max_depth' in kwargs:
            filtered_kwargs['max_depth'] = kwargs['max_depth']
        return RFCoordinatePredictor(**filtered_kwargs)
    
    elif predictor_type.lower() == 'svr':
        # SVR chấp nhận C, epsilon và kernel
        if 'C' in kwargs:
            filtered_kwargs['C'] = kwargs['C']
        if 'epsilon' in kwargs:
            filtered_kwargs['epsilon'] = kwargs['epsilon']
        if 'kernel' in kwargs:
            filtered_kwargs['kernel'] = kwargs['kernel']
        # Không truyền random_state cho SVR
        return SVRCoordinatePredictor(**filtered_kwargs)
    
    elif predictor_type.lower() == 'gb':
        # GradientBoosting chấp nhận random_state, n_estimators, learning_rate và max_depth
        if 'random_state' in kwargs:
            filtered_kwargs['random_state'] = kwargs['random_state']
        if 'n_estimators' in kwargs:
            filtered_kwargs['n_estimators'] = kwargs['n_estimators']
        if 'learning_rate' in kwargs:
            filtered_kwargs['learning_rate'] = kwargs['learning_rate']
        if 'max_depth' in kwargs:
            filtered_kwargs['max_depth'] = kwargs['max_depth']
        return GBCoordinatePredictor(**filtered_kwargs)
    
    else:
        raise ValueError(f"Không hỗ trợ loại dự đoán: {predictor_type}") 