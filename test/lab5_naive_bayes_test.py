# Tên file: test/lab5_naive_bayes_test.py

import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB # Import mô hình Naive Bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict

# 1. Định nghĩa bộ dữ liệu (từ Task 1)
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

# 2. Chia dữ liệu
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

print("--- Dữ liệu Huấn luyện (Train) ---")
for t, l in zip(X_train_texts, y_train):
    print(f"Nhãn: {l} | Văn bản: {t}")
    
print("\n--- Dữ liệu Kiểm tra (Test) ---")
for t, l in zip(X_test_texts, y_test):
    print(f"Nhãn: {l} | Văn bản: {t}")

# 3. Khởi tạo và Huấn luyện Vectorizer
vectorizer = TfidfVectorizer()
# Dùng fit_transform trên data train
X_train_vec = vectorizer.fit_transform(X_train_texts)
# Chỉ dùng transform trên data test
X_test_vec = vectorizer.transform(X_test_texts)

# 4. Khởi tạo và Huấn luyện mô hình Naive Bayes
# (Đây là điểm thay đổi so với Task 3)
print("\n--- Bắt đầu huấn luyện với MultinomialNB ---")
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("Đã huấn luyện xong.")

# 5. Đưa ra dự đoán (Predict)
print("\n--- Bắt đầu dự đoán ---")
y_pred = model.predict(X_test_vec)

# 6. Đánh giá (Evaluate)
def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

metrics = evaluate(y_test, y_pred)

# 7. In kết quả
print("\n--- Kết quả Dự đoán (Naive Bayes) ---")
for i in range(len(X_test_texts)):
    print(f"Văn bản:   \"{X_test_texts[i]}\"")
    print(f"Nhãn thật: {y_test[i]} | Dự đoán: {y_pred[i]}")
    print("-" * 20)

print("\n--- Chỉ số Đánh giá (Naive Bayes) ---")
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-score:  {metrics['f1_score']:.4f}")