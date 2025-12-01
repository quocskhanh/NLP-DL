# src/representations/count_vectorizer.py
from src.core.interfaces import Vectorizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        # Constructor nhận vào một instance của Tokenizer từ Lab 1 [cite: 20]
        self.tokenizer = tokenizer
        # Thuộc tính vocabulary lưu mapping từ từ vựng sang index (dict[str, int]) [cite: 21]
        self.vocabulary = {}

    def fit(self, corpus: list[str]):
        """
        Hàm fit thực hiện:
        1. Tạo set rỗng để chứa token duy nhất[cite: 23].
        2. Duyệt qua từng văn bản, tokenize và thêm vào set[cite: 24, 25, 26].
        3. Tạo từ điển vocabulary mapping token -> index[cite: 28].
        """
        unique_tokens = set()
        
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)
        
        # Sắp xếp để index ổn định và tạo dictionary
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

    def transform(self, documents: list[str]) -> list[list[int]]:
        """
        Hàm transform thực hiện:
        1. Tạo vector 0 với độ dài bằng kích thước vocabulary[cite: 31].
        2. Duyệt qua token, nếu có trong vocab thì tăng đếm tại index tương ứng[cite: 32].
        """
        rows = []
        vocab_len = len(self.vocabulary)
        
        for document in documents:
            tokens = self.tokenizer.tokenize(document)
            # Tạo vector toàn số 0
            vector = [0] * vocab_len
            
            for token in tokens:
                if token in self.vocabulary:
                    index = self.vocabulary[token]
                    vector[index] += 1
            
            rows.append(vector)
        
        return rows