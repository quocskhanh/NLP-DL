# src/core/interfaces.py
from abc import ABC, abstractmethod

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]):
        """Học từ vựng từ danh sách các văn bản (corpus)."""
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        """Chuyển đổi danh sách văn bản thành các vector đếm dựa trên từ vựng đã học."""
        pass

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        """Phương thức tiện ích: thực hiện fit sau đó transform trên cùng dữ liệu."""
        self.fit(corpus)
        return self.transform(corpus)