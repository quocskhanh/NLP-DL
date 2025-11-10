# Tên file: test/lab5_test.py

import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Cấu hình đường dẫn để import ---
# Thêm thư mục gốc (project root) vào sys.path
# Giả sử file test này nằm trong 'test/' và code class nằm trong 'src/'
# Đường dẫn này trỏ lên một cấp (từ 'test' lên thư mục cha)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import class từ Task 2 (từ file src/models/text_classifier.py)
try:
    from src.models.text_classifier import TextClassifier
except ImportError:
    print("Lỗi: Không tìm thấy file 'src/models/text_classifier.py'.")
    print("Hãy chắc chắn rằng bạn đã lưu file từ Task 2 vào đúng vị trí.")
    sys.exit(1)
# -----------------------------------


# 1. Định nghĩa bộ dữ liệu (từ Task 1) [cite: 16-24, 47]
texts = [
    "This movie is fantastic and I love it!", #
    "I hate this film, it's terrible.", #
    "The acting was superb, a truly great experience.", #
    "What a waste of time, absolutely boring.", #
    "Highly recommend this, a masterpiece.", #
    "Could not finish watching, so bad." #
]
labels = [1, 0, 1, 0, 1, 0]

# 2. Chia dữ liệu (Split the data) [cite: 48, 49]
# Bộ dữ liệu rất nhỏ (6 mẫu). Tỷ lệ 80/20 [cite: 48] sẽ cho 1-2 mẫu test.
# Chúng ta dùng test_size=0.33 (sẽ lấy 2 mẫu) và random_state để lặp lại kết quả.
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

print("--- Dữ liệu Huấn luyện (Train) ---")
for t, l in zip(X_train_texts, y_train):
    print(f"Nhãn: {l} | Văn bản: {t}")
    
print("\n--- Dữ liệu Kiểm tra (Test) ---")
for t, l in zip(X_test_texts, y_test):
    print(f"Nhãn: {l} | Văn bản: {t}")

# 3. Khởi tạo Vectorizer [cite: 50]
# Chúng ta sẽ dùng TfidfVectorizer như đã dùng ở Task 1
vectorizer = TfidfVectorizer()

# 4. Khởi tạo TextClassifier (truyền vectorizer vào) [cite: 51]
classifier = TextClassifier(vectorizer=vectorizer)

# 5. Huấn luyện (Train) mô hình [cite: 52]
# Phương thức .fit() của classifier sẽ tự động gọi vectorizer.fit_transform()
print("\n--- Bắt đầu huấn luyện ---")
classifier.fit(X_train_texts, y_train)
print(f"Đã huấn luyện xong.")
print(f"Số lượng từ vựng đã học từ data train: {len(vectorizer.get_feature_names_out())}")

# 6. Đưa ra dự đoán (Predict) trên dữ liệu test [cite: 53]
print("\n--- Bắt đầu dự đoán ---")
y_pred = classifier.predict(X_test_texts)

# 7. Đánh giá (Evaluate) kết quả [cite: 54]
metrics = classifier.evaluate(y_test, y_pred)

# 8. In kết quả
print("\n--- Kết quả Dự đoán trên Test Set ---")
for i in range(len(X_test_texts)):
    print(f"Văn bản:   \"{X_test_texts[i]}\"")
    print(f"Nhãn thật: {y_test[i]} | Dự đoán: {y_pred[i]}")
    print("-" * 20)

print("\n--- Chỉ số Đánh giá Mô hình ---")
# Dùng f-string để format output đẹp hơn
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-score:  {metrics['f1_score']:.4f}")