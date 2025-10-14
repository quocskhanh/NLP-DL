# test/lab4_embedding_training_demo.py

import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def main():
    """
    Minh họa quy trình đào tạo, lưu và sử dụng mô hình Word2Vec tùy chỉnh. [cite: 60]
    """
    # Thiết lập đường dẫn
    data_folder = 'data/UD_English-EWT'
    input_file = os.path.join(data_folder, 'en_ewt-ud-train.txt')
    output_folder = 'results'
    output_model_path = os.path.join(output_folder, 'word2vec_ewt.model')

    # Tạo các thư mục giả nếu chúng chưa tồn tại
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Tạo tệp dữ liệu đào tạo giả
    sample_data = [
        "the quick brown fox jumps over the lazy dog",
        "a king is a powerful man",
        "a queen is a powerful woman",
        "the queen rules with grace",
        "the king rules with power",
        "computers process data very quickly",
        "a laptop is a portable computer",
    ]
    with open(input_file, 'w') as f:
        for line in sample_data:
            f.write(line + '\n')

    print(f"Đang đọc dữ liệu từ: {input_file}")
    # 1. Truyền dữ liệu một cách hiệu quả về bộ nhớ [cite: 57]
    sentences = LineSentence(input_file)

    print("Bắt đầu đào tạo mô hình Word2Vec...")
    # 2. Đào tạo mô hình Word2Vec [cite: 58]
    # vector_size: Số chiều của embedding
    # window: Khoảng cách tối đa giữa từ hiện tại và từ được dự đoán trong một câu
    # min_count: Bỏ qua tất cả các từ có tổng tần suất thấp hơn giá trị này
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    print("Đào tạo hoàn tất.")

    # 3. Lưu mô hình đã đào tạo [cite: 59]
    model.save(output_model_path)
    print(f"Mô hình đã được lưu tại: {output_model_path}")

    # 4. Tải và minh họa cách sử dụng mô hình [cite: 60]
    print("\n--- Minh họa mô hình đã đào tạo tùy chỉnh ---")
    loaded_model = Word2Vec.load(output_model_path)
    
    # Tìm các từ tương tự
    word_to_check = 'queen'
    similar_words = loaded_model.wv.most_similar(word_to_check, topn=5)
    print(f"Các từ tương tự nhất với '{word_to_check}': {similar_words}")

    # Giải các phép loại suy: king - man + woman = ?
    try:
        analogy_result = loaded_model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
        print(f"Kết quả loại suy (king - man + woman): {analogy_result}")
    except KeyError as e:
        print(f"Không thể thực hiện phép loại suy, từ '{e.args[0]}' không có trong từ vựng.")

if __name__ == '__main__':
    main()