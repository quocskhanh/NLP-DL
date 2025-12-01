# Lab 5: Phân loại văn bản (Text Classification)

## 🎯 Mục tiêu (Objective)

Mục tiêu của bài lab này là xây dựng một pipeline (quy trình) phân loại văn bản hoàn chỉnh, đi từ văn bản thô đến một mô hình máy học đã được huấn luyện. Chúng ta sẽ áp dụng các kỹ thuật tokenization và vectorization đã học.

Các khái niệm chính bao gồm:
* **Vectorization**: Sử dụng **TF-IDF** để chuyển đổi văn bản thành các đặc trưng số.
* **Training**: Huấn luyện một mô hình tuyến tính (`LogisticRegression`) trên dữ liệu đã được vector hóa.
* **Evaluation**: Đánh giá hiệu suất mô hình bằng các chỉ số như Accuracy, Precision, Recall, và F1-score.

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
    * Định nghĩa lớp `TextClassifier`.
    * Bao gồm các phương thức `fit()`, `predict()`, và `evaluate()`.
    * Sử dụng mô hình `LogisticRegression` của scikit-learn làm mô hình phân loại.

* **`test/lab5_test.py`**:
    * File thực thi chính cho Task 1 và 3.
    * Tải bộ dữ liệu mẫu (6 câu).
    * Chia dữ liệu thành tập train và test.
    * Huấn luyện `TextClassifier` (với `LogisticRegression`) và in ra các chỉ số đánh giá.

* **`test/lab5_naive_bayes_test.py`**:
    * File thử nghiệm cho Task 4.
    * Thực hiện cùng một quy trình nhưng thay thế mô hình `LogisticRegression` bằng `MultinomialNB` (Naive Bayes) để so sánh kết quả.

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

Tài liệu lab cũng cung cấp một ví dụ nâng cao sử dụng **PySpark** (`test/lab5_spark_sentiment_analysis.py`). [cite_start]Ví dụ này minh họa cách xây dựng một pipeline tương tự nhưng có khả năng mở rộng để xử lý các bộ dữ liệu rất lớn (Big Data) không thể vừa trong bộ nhớ của một máy.

## Sử dụng AI
Ở bài này, em có sử dụng AI (Gemini) để chỉnh sửa và tối ưu code của mình.