# Lab 5: Recurrent Neural Networks (RNNs) & Text Classification

Dự án này bao gồm các bài tập thực hành về **PyTorch cơ bản** và xây dựng pipeline **Phân loại văn bản (Text Classification)** sử dụng các mô hình học sâu (Deep Learning) như RNN/LSTM trên bộ dữ liệu HWU64.

## 📂 Cấu trúc dự án

```text
Lab5_Part2/
├── hwu/                             # Thư mục dữ liệu (giải nén từ hwu.tar.gz)
│   ├── train.csv                    # Tập huấn luyện [cite: 162]
│   ├── val.csv                      # Tập kiểm định [cite: 163]
│   ├── test.csv                     # Tập kiểm tra [cite: 164]
│   └── categories.json              # Danh sách các nhãn (intents)
├── pytorch_basic.py                 # Script thực hành các thao tác Tensor & Autograd
├── lab5_rnns_text_classification.py # Script chính cho bài toán phân loại văn bản
└── README.md                        # File hướng dẫn này

🛠 Yêu cầu cài đặt (Prerequisites)
Để chạy được mã nguồn, bạn cần cài đặt các thư viện Python sau: pip install torch numpy pandas scikit-learn gensim tensorflow

PyTorch: Cho phần thực hành tensor và autograd.

TensorFlow/Keras: Xây dựng và huấn luyện các mô hình LSTM.

Gensim: Huấn luyện mô hình Word2Vec.

Scikit-learn: Tiền xử lý dữ liệu (LabelEncoder) và đánh giá mô hình (Classification Report).

📝 Phần 1: PyTorch Cơ bản (pytorch_basic.py)
Script này giới thiệu các khái niệm cốt lõi trong PyTorch:

Tensor Manipulation:

Tạo tensor từ list, numpy array .

Các phép toán: Cộng, nhân vô hướng, nhân ma trận (@), chuyển vị .

Indexing & Slicing (truy xuất dữ liệu) .

Reshaping (view) .


Autograd: Cơ chế tự động tính đạo hàm (backward()) phục vụ cho lan truyền ngược .

Neural Network Modules:


nn.Linear: Lớp kết nối đầy đủ (Linear transformation).


nn.Embedding: Lớp biểu diễn từ vector (Lookup table).


Custom Model: Định nghĩa lớp MyFirstModel hoàn chỉnh với luồng xử lý: Embedding -> Linear -> ReLU -> Output .

🧠 Phần 2: Phân loại văn bản (lab5_rnns_text_classification.py)
Script này giải quyết bài toán phân loại ý định (Intent Classification) trên bộ dữ liệu HWU64 (64 loại ý định khác nhau như alarm_set, music_query, v.v.).

Dữ liệu được xử lý qua 4 phương pháp tiếp cận khác nhau để so sánh hiệu quả:

1. Baseline: TF-IDF + Logistic Regression
Sử dụng TfidfVectorizer (giới hạn 5000 features) để trích xuất đặc trưng.

Mô hình phân loại tuyến tính LogisticRegression.

2. Word2Vec Average + Dense Neural Network
Huấn luyện mô hình Word2Vec (bằng thư viện Gensim) trên tập dữ liệu train để tạo embedding riêng .

Biểu diễn mỗi câu bằng cách tính trung bình cộng các vector từ trong câu đó.

Mô hình Keras đơn giản: Input -> Dense (ReLU) -> Dropout -> Output (Softmax) .

3. LSTM với Pre-trained Embeddings (Frozen)
Sử dụng trọng số từ mô hình Word2Vec đã huấn luyện ở bước 2 để khởi tạo lớp Embedding của Keras .


Đóng băng (Trainable=False) lớp Embedding để giữ nguyên trọng số Word2Vec.

Mô hình: Embedding -> LSTM (128 units) -> Dense (Softmax) .

4. LSTM với Embeddings Train from Scratch
Lớp Embedding được khởi tạo ngẫu nhiên và huấn luyện cùng lúc với toàn bộ mạng (Trainable=True).

Cho phép mô hình tự học cách biểu diễn từ tối ưu nhất cho tác vụ phân loại cụ thể này.

🔍 Phân tích định tính (Qualitative Analysis)
Cuối script thực hiện kiểm tra nhanh trên 3 câu mẫu khó (phủ định, câu ghép) để xem dự đoán thực tế của từng mô hình.

🚀 Hướng dẫn chạy (Usage)
Bước 1: Chuẩn bị dữ liệu Đảm bảo bạn đã giải nén file hwu.tar.gz và đặt thư mục hwu vào đúng đường dẫn. Lưu ý: Trong file code, đường dẫn đang là tuyệt đối (D:/NLP&DL/...), bạn nên sửa lại thành đường dẫn tương đối nếu chạy trên máy khác.

Bước 2: Chạy PyTorch Basic: python pytorch_basic.py
Bước 3: Chạy Phân loại văn bản: python lab5_rnns_text_classification.py