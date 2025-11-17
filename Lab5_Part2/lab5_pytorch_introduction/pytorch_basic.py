import torch
import numpy as np

# Tạo tensor từ list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor từ list:\n {x_data}\n")

# Tạo tensor từ NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor từ NumPy array:\n {x_np}\n")

# Tạo tensor với các giá trị ngẫu nhiên hoặc hằng số
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # 
print(f"Random Tensor: \n {x_rand}\n") # 

# In ra shape, dtype, và device của tensor
print(f"Shape của tensor: {x_rand.shape}") # 
print(f"Datatype của tensor: {x_rand.dtype}") # 
print(f"Device lưu trữ tensor: {x_rand.device}") #

# Sử dụng x_data = tensor([[1, 2], [3, 4]]) đã tạo ở Task 1.1

# 1. Cộng x_data với chính nó
print(f"x_data + x_data:\n {x_data + x_data}\n") # 

# 2. Nhân x_data với 5 (Nhân vô hướng)
print(f"x_data * 5:\n {x_data * 5}\n") # 

# 3. Nhân ma trận x_data với x_data.T
# x_data.T (Chuyển vị): tensor([[1, 3], [2, 4]])
# Kết quả: [[1*1+2*2, 1*3+2*4], [3*1+4*2, 3*3+4*4]] = [[5, 11], [11, 25]]
print(f"Phép nhân ma trận x_data @ x_data.T:\n {x_data @ x_data.T}") #

# Sử dụng x_data = tensor([[1, 2], [3, 4]])

# 1. Lấy ra hàng đầu tiên
print(f"Hàng đầu tiên: {x_data[0]}") # 

# 2. Lấy ra cột thứ hai
print(f"Cột thứ hai: {x_data[:, 1]}") # 

# 3. Lấy ra giá trị ở hàng thứ hai, cột thứ hai
print(f"Giá trị (2, 2): {x_data[1, 1]}") #

# Tạo tensor có shape (4, 4)
tensor_4x4 = torch.rand(4, 4)
print(f"Shape ban đầu: {tensor_4x4.shape}")

# Biến nó thành (16, 1) bằng view (hoặc reshape)
tensor_16x1 = tensor_4x4.view(16, 1)
# tensor_16x1 = tensor_4x4.reshape(16, 1)

print(f"Shape mới (view): {tensor_16x1.shape}")

# Tạo một tensor và yêu cầu tính đạo hàm cho nó
x = torch.ones(1, requires_grad=True) # 
print(f"x: {x}")

# Thực hiện một phép toán
y = x + 2 # 
print(f"y: {y}")

# y được tạo ra từ một phép toán có x (requires_grad=True), nên nó cũng có grad_fn
print(f"grad_fn của y: {y.grad_fn}")

# Thực hiện thêm các phép toán
z = y * y * 3 # 

# Tính đạo hàm của z theo x
z.backward()

# Đạo hàm được lưu trong thuộc tính grad
# dz/dx = 6*(x+2). Với x=1, dz/dx = 18
print(f"Đạo hàm của z theo x: {x.grad}")

# Khởi tạo một lớp Linear biến đổi từ 5 chiều -> 2 chiều
linear_layer = torch.nn.Linear(in_features=5, out_features=2) # 

# Tạo một tensor đầu vào mẫu
input_tensor = torch.randn(3, 5) # 3 mẫu, mỗi mẫu 5 chiều 

# Truyền đầu vào qua lớp linear
output = linear_layer(input_tensor) 

print(f"Input shape: {input_tensor.shape}") 
print(f"Output shape: {output.shape}")
print(f"Output: \n {output}")

# Khởi tạo lớp Embedding cho từ điển 10 từ, mỗi từ 3 chiều
embedding_layer = torch.nn.Embedding(num_embeddings=10, embedding_dim=3) # 

# Tạo một tensor đầu vào chứa các chỉ số của từ (ví dụ: một câu)
input_indices = torch.LongTensor([1, 5, 0, 8]) 

# Lấy ra các vector embedding tương ứng
embeddings = embedding_layer(input_indices) 

print(f"Input shape: {input_indices.shape}")
print(f"Output shape: {embeddings.shape}")
print(f"Embeddings: \n {embeddings}")

from torch import nn 

class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyFirstModel, self).__init__()
        # Định nghĩa các lớp (layer) bạn sẽ dùng
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.linear = nn.Linear(embedding_dim, hidden_dim) 
        self.activation = nn.ReLU() # Hàm kích hoạt 
        self.output_layer = nn.Linear(hidden_dim, output_dim) 

    def forward(self, indices):
        # Định nghĩa luồng dữ liệu đi qua các lớp [cite: 97]
        # 1. Lấy embedding
        embeds = self.embedding(indices)
        # 2. Truyền qua lớp linear và hàm kích hoạt
        # Lớp Linear chỉ được áp dụng cho chiều cuối cùng (embedding_dim)
        hidden = self.activation(self.linear(embeds))
        # 3. Truyền qua lớp output
        output = self.output_layer(hidden)
        return output
    
    # Khởi tạo và kiểm tra mô hình
model = MyFirstModel(vocab_size=100, embedding_dim=16, hidden_dim=8, output_dim=2)

# input_data: (batch_size=1, sequence_length=4)
input_data = torch.LongTensor([[1, 2, 5, 9]]) # một câu gồm 4 từ 
output_data = model(input_data)

# Đầu ra sau khi qua embedding sẽ là (1, 4, 16)
# Đầu ra sau khi qua Linear/ReLU/Output sẽ là (1, 4, 2)
print(f"Model output shape: {output_data.shape}") # [cite: 109, 110]