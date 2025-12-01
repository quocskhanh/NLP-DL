# test/lab2_test.py
import sys
import os
import re

# Thêm thư mục gốc vào path để import được các module trong src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.count_vectorizer import CountVectorizer

# --- Giả lập Tokenizer từ Lab 1 (để code có thể chạy được) ---
class RegexTokenizer:
    def tokenize(self, text):
        # Tách từ đơn giản bằng regex, chuyển về chữ thường để chuẩn hóa
        return re.findall(r'\b\w+\b', text.lower())

# --- Kịch bản test theo yêu cầu đề bài ---
if __name__ == "__main__":
    # 1. Khởi tạo Tokenizer [cite: 36]
    tokenizer = RegexTokenizer()
    
    # 2. Khởi tạo CountVectorizer với tokenizer [cite: 37]
    vectorizer = CountVectorizer(tokenizer)
    
    # 3. Định nghĩa sample corpus [cite: 38-43]
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    
    # 4. Sử dụng fit_transform và in kết quả [cite: 44]
    print("--- Bắt đầu xử lý ---")
    matrix = vectorizer.fit_transform(corpus)
    
    print("\nVocabulary (Từ vựng đã học):")
    # In ra dictionary mapping word -> index
    print(vectorizer.vocabulary)
    
    print("\nDocument-Term Matrix (Các vector kết quả):")
    for doc, vec in zip(corpus, matrix):
        print(f"Doc: '{doc}' -> Vector: {vec}")