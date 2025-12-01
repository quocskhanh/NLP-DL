📘 NLP & Deep Learning – Laboratory Exercises

Repository này tổng hợp toàn bộ các bài thực hành (lab) của môn Natural Language Processing & Deep Learning, bao gồm các chủ đề từ xử lý ngôn ngữ tự nhiên cơ bản, biểu diễn văn bản, word embeddings, đến mô hình học sâu và transformers.
NLP&DL/
│
├── data/                 # Dữ liệu dùng trong các bài lab
│   └── data_report.md
│
├── report/               # Báo cáo cho từng bài lab
│   ├── lab1_report.md
│   ├── lab2_report.md
│   ├── lab4_report.md
│   ├── lab5_report.md
│   ├── lab5p2_report.md
│   └── lab6_report.md
│
├── src/                  # Mã nguồn chính
│   ├── lab1_interfaces.py
│   ├── lab1_simple_tokenizer.py
│   ├── lab2_count_vectorizer.py
│   ├── lab2_interfaces.py
│   ├── lab4_embedding_training_demo.py
│   ├── lab4_spark_word2vec_demo.py
│   ├── lab4_word_embedder.py
│   ├── lab5_text_classifier.py
│   ├── lab5p2_pytorch_basic.py
│   ├── lab5p2_rnns_text_classification.py
│   └── lab6_transformers_intro.py
│
├── test/                 # Unit tests cho mỗi phần
│   ├── lab1_test.py
│   ├── lab2_test.py
│   ├── lab4_test.py
│   ├── lab5_test.py
│   └── lab5_naive_bayes_test.py
│
└── requirements.txt      # Thư viện Python cần cài đặt

🧪 Nội dung các bài lab
🔹 Lab 1 – Tokenization & Text Preprocessing

Implement tokenizer đơn giản.

Thực hành interface chuẩn cho tokenizer.

Xử lý văn bản cơ bản: tách từ, lowercase, loại bỏ dấu câu.

Kiểm thử bằng lab1_test.py.

File quan trọng
src/lab1_simple_tokenizer.py
src/lab1_interfaces.py

🔹 Lab 2 – Bag of Words & Count Vectorizer

Cài đặt CountVectorizer thủ công.

Xây dựng vocabulary, vector hóa văn bản.

Tiền xử lý: tokenization, stopwords, normalization.

Kiểm thử bằng lab2_test.py.

File quan trọng
src/lab2_count_vectorizer.py

🔹 Lab 4 – Word Embedding (Word2Vec)

Tự xây dựng mô hình Word Embedding mini bằng Python.

Demo với Spark Word2Vec.

Thực hành nhúng văn bản và trực quan hóa.

Kiểm thử bằng lab4_test.py.

File quan trọng
src/lab4_word_embedder.py
src/lab4_embedding_training_demo.py
src/lab4_spark_word2vec_demo.py

🔹 Lab 5 – Text Classification (Naive Bayes & Neural Nets)

Phân loại văn bản sử dụng:

Naive Bayes

Mô hình mạng neuron cơ bản

Huấn luyện classifier với dữ liệu mẫu.

Kiểm thử bằng:
lab5_test.py – classifier
lab5_naive_bayes_test.py – naive bayes

File quan trọng
src/lab5_text_classifier.py

🔹 Lab 5 Part 2 – PyTorch & RNNs

Làm quen với PyTorch: tensor, autograd, optimizer.

Xây dựng RNN cho phân loại văn bản.

LSTM, GRU, huấn luyện mô hình.

File quan trọng
src/lab5p2_pytorch_basic.py
src/lab5p2_rnns_text_classification.py

🔹 Lab 6 – Introduction to Transformers

Hiểu pipeline của HuggingFace transformers.

Tokenization với mô hình pretrained.

Text classification cơ bản bằng BERT.

File quan trọng
src/lab6_transformers_intro.py

📌 Ghi chú

Repo phục vụ mục đích học tập và thực hành NLP & DL.

Các phần có thể mở rộng: training pipeline, visualization, dataset loader, benchmarking.

Trong các bài lab được giao không có lab 3 nên trong repo sẽ không có lab 3