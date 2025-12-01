# Tên file: src/models/text_classifier.py

from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Định nghĩa một kiểu (type) cho vectorizer để code rõ ràng hơn
# Dựa trên Lab 2 và 3, chúng ta sẽ dùng TfidfVectorizer hoặc CountVectorizer
Vectorizer = Union[TfidfVectorizer, CountVectorizer]

class TextClassifier:
    
    def __init__(self, vectorizer: Vectorizer):
        """
        Khởi tạo classifier với một vectorizer.
        """
        self.vectorizer = vectorizer
        # self.model sẽ lưu trữ mô hình LogisticRegression sau khi huấn luyện [cite: 32]
        self.model: LogisticRegression = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện (train) mô hình classifier[cite: 34].
        """
        # 1. Dùng vectorizer để fit_transform văn bản thành ma trận đặc trưng X [cite: 35]
        X = self.vectorizer.fit_transform(texts)
        
        # 2. Khởi tạo mô hình LogisticRegression [cite: 36]
        # (solver='liblinear' được gợi ý cho bộ dữ liệu nhỏ)
        self.model = LogisticRegression(solver='liblinear')
        
        # 3. Huấn luyện mô hình [cite: 37]
        self.model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho danh sách văn bản mới[cite: 39].
        """
        if self.model is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Vui lòng gọi fit() trước.")
            
        # 1. Dùng vectorizer để transform văn bản thành ma trận X [cite: 40]
        # (Lưu ý: dùng .transform() chứ không phải .fit_transform()
        # vì từ vựng đã được học ở bước fit)
        X = self.vectorizer.transform(texts)
        
        # 2. Dùng mô hình đã huấn luyện để dự đoán [cite: 41]
        predictions = self.model.predict(X)
        
        # Trả về list các nhãn dự đoán
        return list(predictions)

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá[cite: 43].
        """
        # Sử dụng các hàm từ sklearn.metrics [cite: 44]
        accuracy = accuracy_score(y_true, y_pred)
        # Thêm zero_division=0 để tránh lỗi/cảnh báo khi không có dự đoán positive
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Trả về một dictionary chứa các chỉ số [cite: 44]
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        return metrics