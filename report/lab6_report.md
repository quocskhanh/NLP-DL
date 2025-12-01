# 📘 Lab 6 – Introduction to Transformers

---

## 🧠 Mục tiêu bài lab

File `lab6_transformers_intro.py` giúp bạn làm quen với:

- Kiến trúc Transformer và các loại mô hình phổ biến (BERT, GPT).
- Cách sử dụng thư viện **Hugging Face Transformers**.
- 3 tác vụ NLP cơ bản:
  1. Masked Language Modeling (Fill-mask)
  2. Text Generation (Sinh văn bản)
  3. Sentence Embedding (Vector biểu diễn câu)

---

## 📂 Nội dung chương trình

File code gồm 3 phần chính:

### 1️⃣ Masked Language Modeling (Fill-mask)

Sử dụng mô hình **BERT (Encoder-only)** để dự đoán từ bị che `[MASK]`.

Ví dụ: Hanoi is the [MASK] of Vietnam.

Mô hình sẽ dự đoán các từ phù hợp như: `capital`, `center`,...

👉 Phù hợp cho các bài toán:
- Dự đoán từ bị thiếu  
- Hiểu ngữ cảnh hai chiều  

---

### 2️⃣ Text Generation (Sinh văn bản)

Sử dụng mô hình **GPT (Decoder-only)** để sinh văn bản từ một câu mồi.

Ví dụ: The best thing about learning NLP is

👉 Ứng dụng:
- Chatbot  
- Sinh nội dung tự động  
- Mô hình tiếp văn bản  

---

### 3️⃣ Sentence Embedding (Vector biểu diễn câu)

Sử dụng **BERT** để chuyển một câu thành vector số bằng phương pháp **Mean Pooling**.

Ví dụ: This is a sample sentence.

Kết quả vector có kích thước: torch.Size([1, 768])


👉 Con số **768** tương ứng với **hidden size** của mô hình `bert-base-uncased`.

---

## ⚙️ Yêu cầu cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install transformers torch

## Trợ giúp
Bài làm của em có sự trợ giúp của AI (Gemini) 