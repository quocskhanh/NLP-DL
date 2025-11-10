# Lab 5: Phân loại văn bản (Text Classification)

## 🎯 Mục tiêu (Objective)

[cite_start]Mục tiêu của bài lab này là xây dựng một pipeline (quy trình) phân loại văn bản hoàn chỉnh[cite: 6], đi từ văn bản thô đến một mô hình máy học đã được huấn luyện. Chúng ta sẽ áp dụng các kỹ thuật tokenization và vectorization đã học.

Các khái niệm chính bao gồm:
* [cite_start]**Vectorization**: Sử dụng **TF-IDF** để chuyển đổi văn bản thành các đặc trưng số[cite: 25].
* [cite_start]**Training**: Huấn luyện một mô hình tuyến tính (`LogisticRegression`) trên dữ liệu đã được vector hóa[cite: 12].
* [cite_start]**Evaluation**: Đánh giá hiệu suất mô hình bằng các chỉ số như Accuracy, Precision, Recall, và F1-score[cite: 13].

---

## 📂 Cấu trúc thư mục

```
Lab5/
├── src/
│   └── models/
│       └── text_classifier.py  # (Task 2) Định nghĩa lớp TextClassifier
├── test/
│   ├── lab5_test.py            # (Task 3) Huấn luyện & đánh giá TextClassifier
│   └── lab5_naive_bayes_test.py # (Task 4) Thử nghiệm với mô hình Naive Bayes
└── README.md                   # File này
```

### Chi tiết các File

* **`src/models/text_classifier.py`**:
    * [cite_start]Định nghĩa lớp `TextClassifier`[cite: 30].
    * [cite_start]Bao gồm các phương thức `fit()`, `predict()`, và `evaluate()`[cite: 33, 38, 42].
    * [cite_start]Sử dụng mô hình `LogisticRegression` của scikit-learn làm mô hình phân loại[cite: 32, 36].

* **`test/lab5_test.py`**:
    * File thực thi chính cho Task 1 và 3.
    * [cite_start]Tải bộ dữ liệu mẫu (6 câu) [cite: 16-24].
    * [cite_start]Chia dữ liệu thành tập train và test[cite: 48].
    * Huấn luyện `TextClassifier` (với `LogisticRegression`) và in ra các chỉ số đánh giá.

* **`test/lab5_naive_bayes_test.py`**:
    * File thử nghiệm cho Task 4.
    * [cite_start]Thực hiện cùng một quy trình nhưng thay thế mô hình `LogisticRegression` bằng `MultinomialNB` (Naive Bayes) để so sánh kết quả[cite: 118].

---

## 🚀 Cách chạy (How to Run)

Bạn có thể chạy các file test từ thư mục gốc của dự án (`NLP-DL/`).

**1. Chạy bài lab chính (Logistic Regression):**
```bash
python Lab5/test/lab5_test.py
```

**2. Chạy bài thử nghiệm (Naive Bayes):**
```bash
python Lab5/test/lab5_naive_bayes_test.py
```

---

## ✨ Ví dụ nâng cao (Advanced Example)

[cite_start]Tài liệu lab cũng cung cấp một ví dụ nâng cao sử dụng **PySpark** (`test/lab5_spark_sentiment_analysis.py`)[cite: 57]. [cite_start]Ví dụ này minh họa cách xây dựng một pipeline tương tự nhưng có khả năng mở rộng để xử lý các bộ dữ liệu rất lớn (Big Data) không thể vừa trong bộ nhớ của một máy[cite: 56].