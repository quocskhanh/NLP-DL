📝 Simple Tokenizer – Mini NLP Preprocessing Demo

Dự án này minh hoạ cách xây dựng một Tokenizer đơn giản trong Python, sử dụng OOP và Abstract Class. Tokenizer thực hiện các bước cơ bản như:

Chuyển chữ viết thường

Tách dấu câu bằng regex

Loại bỏ khoảng trắng thừa

Tách từ theo khoảng trắng

Dự án rất phù hợp để nhập môn xử lý ngôn ngữ tự nhiên (NLP).

src/
├── core/
│   └── interfaces.py        # Định nghĩa abstract class Tokenizer
└── preprocessing/
    └── simple_tokenizer.py  # Triển khai SimpleTokenizer
main.py                      # Chạy thử tokenizer

📚 Giải thích các file chính
1. interfaces.py

Định nghĩa abstract class Tokenizer với một phương thức bắt buộc

→ Bảo đảm mọi tokenizer khác đều phải implement tokenize().

2. simple_tokenizer.py

Triển khai SimpleTokenizer:

Lowercase toàn bộ text

Tách dấu . , ? !

Loại bỏ khoảng trắng thừa bằng regex

Tách tokens bằng split
3. main.py

Script demo cách dùng tokenizer:

Tạo đối tượng SimpleTokenizer

Tokenize một list câu mẫu

In kết quả ra màn hình

⭐ Mục tiêu học được

Hiểu cách dùng ABC để tạo interface trong Python

Nắm được logic tiền xử lý văn bản cơ bản

Biết dùng regex để tách dấu câu

Tổ chức code dạng module theo chuẩn dự án NLP

📌 Hướng phát triển thêm

Bạn có thể mở rộng tokenizer để:

Loại stopwords

Stemming / Lemmatization

Tách từ viết dính (Vietnamese word segmentation)

Tách emoji, URL, email, hashtag

Trong bài làm em có dùng ChatGpt(free) để làm readme.md