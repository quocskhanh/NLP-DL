# test/lab4_test.py

import sys
import os
import numpy as np

# Thêm thư mục gốc của dự án vào sys.path để cho phép import từ src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder

def main():
    """
    Hàm chính để chạy các bài kiểm tra đánh giá cho WordEmbedder.
    """
    # Khởi tạo WordEmbedder với mô hình được chỉ định [cite: 24, 47]
    embedder = WordEmbedder('glove-wiki-gigaword-50')

    # 1. Lấy vector cho từ 'king' [cite: 49]
    print("\n--- Vector cho 'king' ---")
    king_vector = embedder.get_vector('king')
    # In 5 chiều đầu tiên để ngắn gọn
    print(f"Vector (5 chiều đầu tiên): {king_vector[:5]}")
    print(f"Tổng số chiều: {len(king_vector)}")

    # 2. Lấy độ tương tự giữa các từ [cite: 50]
    print("\n--- Độ tương tự từ ---")
    sim_king_queen = embedder.get_similarity('king', 'queen')
    sim_king_man = embedder.get_similarity('king', 'man')
    print(f"Độ tương tự giữa 'king' và 'queen': {sim_king_queen:.4f}")
    print(f"Độ tương tự giữa 'king' và 'man': {sim_king_man:.4f}")

    # 3. Lấy 10 từ tương tự nhất với 'computer' [cite: 50]
    print("\n--- 10 từ tương tự nhất với 'computer' ---")
    similar_words = embedder.get_most_similar('computer', top_n=10)
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")

    # 4. Embed câu "The queen rules the country." [cite: 51]
    print("\n--- Embedding cho tài liệu ---")
    sentence = "The queen rules the country."
    doc_vector = embedder.embed_document(sentence)
    print(f"Câu: \"{sentence}\"")
    # In 5 chiều đầu tiên của vector tài liệu
    print(f"Vector tài liệu (5 chiều đầu tiên): {doc_vector[:5]}")
    print(f"Tổng số chiều: {len(doc_vector)}")

if __name__ == "__main__":
    main()