# Lab 2: Count Vectorization

## Tổng quan
Dự án này cài đặt mô hình **Bag-of-Words** thông qua class `CountVectorizer`. [cite_start]Mục tiêu chính là biểu diễn các tài liệu văn bản dưới dạng các vector số, cho phép sử dụng dữ liệu văn bản trong các mô hình học máy[cite: 5, 6, 7].

[cite_start]Dự án tái sử dụng `Tokenizer` đã được phát triển từ Lab 1 để xử lý việc tách từ[cite: 8].

## Cấu trúc dự án
[cite_start]Mã nguồn được tổ chức theo cấu trúc sau[cite: 10, 16, 35]:

```text
.
├── src/
│   ├── core/
│   │   └── interfaces.py           # Chứa abstract base class Vectorizer
│   └── representations/
│       └── count_vectorizer.py     # Cài đặt chính của CountVectorizer
└── test/
    └── lab2_test.py                # Script kiểm thử và đánh giá

Tính năng
1. Vectorizer Interface (src/core/interfaces.py)
Định nghĩa lớp trừu tượng cơ sở Vectorizer với các phương thức chuẩn:


fit(corpus): Học từ vựng từ danh sách văn bản.


transform(documents): Chuyển đổi văn bản thành vector đếm.


fit_transform(corpus): Thực hiện fit và sau đó transform trên cùng một dữ liệu.

2. CountVectorizer (src/representations/count_vectorizer.py)
Lớp này kế thừa từ Vectorizer và cài đặt logic Bag-of-Words:


Khởi tạo (__init__): Nhận một đối tượng Tokenizer và khởi tạo từ điển từ vựng.


Học từ vựng (fit): Duyệt qua corpus, thu thập các token duy nhất và gán chỉ số (index) cho chúng.



Tạo vector (transform): Tạo các vector chứa số lần xuất hiện của từng từ trong từ vựng đối với mỗi văn bản.

Dưới đây là nội dung file README.md chuyên nghiệp, đầy đủ và phù hợp với cấu trúc bài làm của bạn. Bạn có thể lưu nội dung này vào file có tên README.md trong thư mục gốc của dự án.

Markdown

# Lab 2: Count Vectorization

## Tổng quan
Dự án này cài đặt mô hình **Bag-of-Words** thông qua class `CountVectorizer`. [cite_start]Mục tiêu chính là biểu diễn các tài liệu văn bản dưới dạng các vector số, cho phép sử dụng dữ liệu văn bản trong các mô hình học máy[cite: 5, 6, 7].

[cite_start]Dự án tái sử dụng `Tokenizer` đã được phát triển từ Lab 1 để xử lý việc tách từ[cite: 8].

## Cấu trúc dự án
[cite_start]Mã nguồn được tổ chức theo cấu trúc sau[cite: 10, 16, 35]:

```text
.
├── src/
│   ├── core/
│   │   └── interfaces.py           # Chứa abstract base class Vectorizer
│   └── representations/
│       └── count_vectorizer.py     # Cài đặt chính của CountVectorizer
└── test/
    └── lab2_test.py                # Script kiểm thử và đánh giá
Tính năng
1. Vectorizer Interface (src/core/interfaces.py)
Định nghĩa lớp trừu tượng cơ sở Vectorizer với các phương thức chuẩn:


fit(corpus): Học từ vựng từ danh sách văn bản.


transform(documents): Chuyển đổi văn bản thành vector đếm.


fit_transform(corpus): Thực hiện fit và sau đó transform trên cùng một dữ liệu.

2. CountVectorizer (src/representations/count_vectorizer.py)
Lớp này kế thừa từ Vectorizer và cài đặt logic Bag-of-Words:


Khởi tạo (__init__): Nhận một đối tượng Tokenizer và khởi tạo từ điển từ vựng.


Học từ vựng (fit): Duyệt qua corpus, thu thập các token duy nhất và gán chỉ số (index) cho chúng.



Tạo vector (transform): Tạo các vector chứa số lần xuất hiện của từng từ trong từ vựng đối với mỗi văn bản.


Hướng dẫn cài đặt và chạy
Yêu cầu
Python 3.x

Module Tokenizer (đã được giả lập sẵn trong file test nếu chưa có code Lab 1).

Chạy kiểm thử
Để chạy kịch bản kiểm thử mẫu (Evaluation) được định nghĩa trong đề bài, hãy chạy lệnh sau từ thư mục gốc:

Bash

python test/lab2_test.py
Dữ liệu mẫu (Corpus)
Kịch bản test sử dụng corpus mẫu sau để kiểm tra:


Python

corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
Kết quả mong đợi
Chương trình sẽ in ra:

Vocabulary: Dictionary ánh xạ từ token sang index (ví dụ: {'ai': 0, 'i': 1...}).

Document-Term Matrix: Danh sách các vector đếm tương ứng với từng câu trong corpus.