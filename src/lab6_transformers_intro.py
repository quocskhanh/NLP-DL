from transformers import pipeline

# 1. Tải pipeline "fill-mask"
# Pipeline này sẽ tự động tải một mô hình mặc định phù hợp (thường là một biến thể của BERT)
mask_filler = pipeline("fill-mask", model="bert-base-uncased")

# 2. Câu đầu vào với token [MASK]
input_sentence = "Hanoi is the [MASK] of Vietnam."

# 3. Thực hiện dự đoán (top_k=5 yêu cầu 5 dự đoán hàng đầu)
predictions = mask_filler(input_sentence, top_k=5)

# 4. In kết quả
print(f"Câu gốc: '{input_sentence}'")
for pred in predictions:
    print(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
    print(f" -> Câu hoàn chỉnh: {pred['sequence']}")

#Câu hỏi:
#Mô hình đã dự đoán đúng từ "capital" không? 
#Trả lời: Rất có thể có. Mô hình BERT được huấn luyện trên một lượng lớn dữ liệu và sẽ dễ dàng nhận ra mối quan hệ ngữ nghĩa rằng "Hanoi" là thủ đô ("capital") của "Vietnam". Dự đoán hàng đầu (token_str) thường là "capital".
#Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này? 
#Trả lời: Các mô hình Encoder-only (ví dụ: BERT) được huấn luyện để hiểu sâu sắc ngữ cảnh của một câu và có khả năng nhìn hai chiều (bidirectional). Điều này có nghĩa là khi xử lý token [MASK], mô hình có thể xem xét cả các từ đứng trước (Hanoi is the) và các từ đứng sau (of Vietnam) để đưa ra dự đoán chính xác nhất, điều này là cốt lõi của MLM.

from transformers import pipeline

# 1. Tải pipeline "text-generation"
generator = pipeline("text-generation")
# 2. Đoạn văn bản mồi
prompt = "The best thing about learning NLP is"

# 3. Sinh văn bản
# max_length: tổng độ dài của câu mồi và phần được sinh ra
# num_return_sequences: số lượng chuỗi kết quả muốn nhận (đã sửa thành 3 thay vì 13 như code mẫu để tránh quá dài)
generated_texts = generator(prompt, max_length=50, num_return_sequences=3)
# 4. In kết quả
print(f"Câu mồi: '{prompt}'")
for text in generated_texts:
    print("Văn bản được sinh ra:")
    print(text['generated_text'])

#Câu hỏi:
#Kết quả sinh ra có hợp lý không? 
#Trả lời: Có. Các mô hình Decoder-only như GPT được huấn luyện để tạo ra văn bản tự nhiên. Văn bản sinh ra (ví dụ: "The best thing about learning NLP is that it helps you understand how people communicate.") thường có cấu trúc ngữ pháp và ngữ nghĩa hợp lý, mặc dù đôi khi nội dung có thể lặp lại hoặc không logic.
#Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này? 
#Trả lời: Các mô hình Decoder-only (ví dụ: GPT) được huấn luyện chuyên biệt để dự đoán từ tiếp theo (Next Token Prediction) trong một chuỗi. Chúng chỉ có khả năng nhìn một chiều (unidirectional), tức là chỉ xem xét các từ đã xuất hiện trước đó để sinh ra token hiện tại, mô phỏng quá trình viết của con người.

import torch
from transformers import AutoTokenizer, AutoModel

# 1. Chọn một mô hình BERT (Encoder-only)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Câu đầu vào
sentences = ["This is a sample sentence."]

# 3. Tokenize câu
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# 4. Đưa qua mô hình để lấy hidden states
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state 

# 5. Thực hiện Mean Pooling (Bỏ qua padding tokens)
attention_mask = inputs['attention_mask'] 
# Mở rộng mask từ (batch_size, seq_len) thành (batch_size, seq_len, hidden_size)
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float() 

# Tính tổng các embedding (chỉ tính những token không bị mask)
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1) 

# Tính tổng của mask để chia trung bình, đảm bảo không chia cho 0
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) 

# Trung bình cộng
sentence_embedding = sum_embeddings / sum_mask 

# 6. In kết quả
print("Vector biểu diễn của câu:") 
print(sentence_embedding) 
print("\nKích thước của vector:", sentence_embedding.shape)

#Câu hỏi:
#Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT? 
#Trả lời: Kích thước (chiều) của vector biểu diễn là 768 (shape sẽ là torch.Size([1, 768])). Con số này tương ứng với tham số Hidden Size (hoặc embedding_dim) của mô hình bert-base-uncased.
#Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling? 
#Trả lời: Chúng ta cần sử dụng attention_mask để bỏ qua các token đệm (padding tokens) khi tính trung bình cộng. Khi tokenizer đệm các câu ngắn hơn để chúng có cùng độ dài (padding=True), các token đệm này không mang ý nghĩa ngữ nghĩa. Nếu tính trung bình trên tất cả các token (bao gồm cả padding), vector cuối cùng sẽ bị lệch (skewed) về giá trị 0, làm giảm chất lượng biểu diễn của câu.