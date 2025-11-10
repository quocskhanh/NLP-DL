
import gensim.downloader as api
import numpy as np
import re

class Tokenizer:
    """
    Một trình tokenizer đơn giản để chia tài liệu thành các từ.
    Đây là giả định dựa trên "Tokenizer (từ Lab 1)".
    """
    def tokenize(self, text: str) -> list[str]:
        # Chuyển thành chữ thường và chỉ giữ lại các ký tự chữ cái
        return re.findall(r'\b[a-z]+\b', text.lower())

class WordEmbedder:
    """
    Một lớp bao bọc để tải và sử dụng các mô hình word embedding được đào tạo trước từ gensim.
    """
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        """
        Khởi tạo và tải mô hình word embedding được chỉ định. [cite: 29]
        """
        print(f"Đang tải mô hình '{model_name}'... Điều này có thể mất một lúc.")
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
        print("Tải mô hình hoàn tất.")

    def get_vector(self, word: str) -> np.ndarray | None:
        """
        Trả về vector embedding cho một từ đã cho. [cite: 32]
        Xử lý các từ ngoài từ vựng (OOV) bằng cách trả về None. [cite: 33]
        """
        try:
            return self.model[word]
        except KeyError:
            # Từ không có trong từ vựng của mô hình
            return None

    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Trả về độ tương tự cosine giữa hai từ. [cite: 34]
        """
        return self.model.similarity(word1, word2)

    def get_most_similar(self, word: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Tìm N từ tương tự nhất với một từ đã cho. [cite: 35]
        """
        return self.model.most_similar(word, topn=top_n)

    def embed_document(self, document: str) -> np.ndarray:
        """
        Chuyển đổi một tài liệu văn bản thành một vector embedding duy nhất
        bằng cách lấy trung bình các vector của các từ trong đó. [cite: 38, 39]
        """
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(document)

        word_vectors = []
        for token in tokens:
            vector = self.get_vector(token)
            if vector is not None:
                word_vectors.append(vector)
        
        # Nếu tài liệu không chứa từ nào đã biết, trả về một vector không [cite: 43]
        if not word_vectors:
            return np.zeros(self.vector_size, dtype=np.float32)

        # Tính trung bình theo từng phần tử của các vector từ [cite: 44]
        return np.mean(word_vectors, axis=0)