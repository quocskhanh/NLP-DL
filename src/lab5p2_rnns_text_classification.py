import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Cần chạy lệnh shell trước khi chạy Python code: tar -xzvf data/hwu.tar.gz

# Đọc dữ liệu
df_train = pd.read_csv("D:/NLP&DL/Lab5_Part2/hwu/train.csv", sep=',', header=0) #
df_val = pd.read_csv("D:/NLP&DL/Lab5_Part2/hwu/val.csv", sep=',', header=0)
df_test = pd.read_csv("D:/NLP&DL/Lab5_Part2/hwu/test.csv", sep=',', header=0)
print("Train shape:", df_train.shape) #
print("Validation shape:", df_val.shape) #
print("Test shape:", df_test.shape) #
print(df_train.head()) #

# --- BƯỚC 2: TIỀN XỬ LÝ NHÃN (LABEL ENCODING) ---
le = LabelEncoder()
all_intents = pd.concat([df_train['category'], df_val['category'], df_test['category']])
le.fit(all_intents) 

# Dòng này phải chạy TRƯỚC khi bạn kiểm tra y_train
y_train = le.transform(df_train['category']) 
y_val = le.transform(df_val['category']) 
y_test = le.transform(df_test['category']) 

num_classes = len(le.classes_)

# --- BƯỚC 3: MÃ KIỂM TRA (ĐẶT Ở ĐÂY LÀ ĐÚNG) ---
print(f"\nKiểm tra y_train:")
print(f"Tổng số mẫu trong y_train: {len(y_train)}") # y_train đã được định nghĩa
unique_classes = np.unique(y_train)
print(f"Các lớp duy nhất trong y_train: {unique_classes}")
print(f"Số lượng lớp duy nhất: {len(unique_classes)}") # Phải lớn hơn 1
# ----------------------------------------------------

# Khởi tạo và fit LabelEncoder trên toàn bộ tập intent
le = LabelEncoder() #
all_intents = pd.concat([df_train['category'], df_val['category'], df_test['category']])
le.fit(all_intents) #

# Chuyển đổi nhãn sang dạng số
y_train = le.transform(df_train['category']) #
y_val = le.transform(df_val['category']) #
y_test = le.transform(df_test['category']) #

num_classes = len(le.classes_) #
print(f"\nSố lượng lớp (num_classes): {num_classes}")

from sklearn.feature_extraction.text import TfidfVectorizer #
from sklearn.linear_model import LogisticRegression #
from sklearn.pipeline import make_pipeline #
from sklearn.metrics import classification_report #

# 1. Tạo một pipeline với TfidfVectorizer và LogisticRegression
tfidf_lr_pipeline = make_pipeline( #
    TfidfVectorizer(max_features=5000), # Giới hạn số lượng features
    LogisticRegression(max_iter=1000) #
)

# 2. Huấn luyện pipeline trên tập train
X_train = df_train['text']
tfidf_lr_pipeline.fit(X_train, y_train) #

# 3. Đánh giá trên tập test
X_test = df_test['text']
y_pred_lr = tfidf_lr_pipeline.predict(X_test) #
report_lr = classification_report(y_test, y_pred_lr, target_names=le.classes_, output_dict=True)
print("\n--- Kết quả Nhiệm vụ 1 (TF-IDF + LR) ---")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
# Lưu F1-score (macro) cho bảng so sánh
f1_lr = report_lr['macro avg']['f1-score']

from gensim.models import Word2Vec #

# 1. Huấn luyện mô hình Word2Vec trên dữ liệu text
sentences = [text.split() for text in df_train['text']] #
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4) #
embedding_dim = w2v_model.vector_size # 100

# 2. Viết hàm để chuyển mỗi câu thành vector trung bình
def sentence_to_avg_vector(text, model, size): #
    words = text.split()
    # Lấy vector của các từ có trong model, nếu không có thì bỏ qua
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(size)
    return np.mean(word_vectors, axis=0) #

# 3. Tạo dữ liệu train/val/test X_train_avg, X_val_avg, X_test_avg
X_train_avg = np.array([sentence_to_avg_vector(text, w2v_model, embedding_dim) for text in df_train['text']]) #
X_val_avg = np.array([sentence_to_avg_vector(text, w2v_model, embedding_dim) for text in df_val['text']]) #
X_test_avg = np.array([sentence_to_avg_vector(text, w2v_model, embedding_dim) for text in df_test['text']]) #

import tensorflow as tf
from tensorflow.keras.models import Sequential #
from tensorflow.keras.layers import Dense, Dropout #
from tensorflow.keras.utils import to_categorical

# One-hot encode nhãn cho Keras
Y_train_oh = to_categorical(y_train, num_classes=num_classes)
Y_val_oh = to_categorical(y_val, num_classes=num_classes)
Y_test_oh = to_categorical(y_test, num_classes=num_classes)

# 4. Xây dựng mô hình Sequential của Keras
model_avg_w2v = Sequential([ #
    Dense(128, activation='relu', input_shape=(embedding_dim,)), #
    Dropout(0.5), #
    Dense(num_classes, activation='softmax') #
])

# 5. Compile, huấn luyện và đánh giá mô hình
model_avg_w2v.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #

history_w2v = model_avg_w2v.fit(
    X_train_avg, Y_train_oh,
    epochs=20,
    batch_size=32,
    validation_data=(X_val_avg, Y_val_oh),
    verbose=0 # Đặt thành 1 để xem quá trình huấn luyện
) #

# Đánh giá trên tập test
loss_w2v, _ = model_avg_w2v.evaluate(X_test_avg, Y_test_oh, verbose=0)
y_pred_w2v = np.argmax(model_avg_w2v.predict(X_test_avg, verbose=0), axis=1)

report_w2v = classification_report(y_test, y_pred_w2v, target_names=le.classes_, output_dict=True)
print("\n--- Kết quả Nhiệm vụ 2 (Word2Vec Avg + Dense) ---")
print(classification_report(y_test, y_pred_w2v, target_names=le.classes_))
f1_w2v = report_w2v['macro avg']['f1-score']

from tensorflow.keras.preprocessing.text import Tokenizer #
from tensorflow.keras.preprocessing.sequence import pad_sequences #

max_len = 50 # Chiều dài chuỗi tối đa [cite: 257]

# a. Tokenizer: Tạo vocab và chuyển text thành chuỗi chỉ số
tokenizer = Tokenizer(num_words=None, oov_token="<UNK>") # num_words=None để giữ tất cả các từ
all_text = pd.concat([df_train['text'], df_val['text'], df_test['text']])
tokenizer.fit_on_texts(all_text) #

train_sequences = tokenizer.texts_to_sequences(df_train['text']) #
val_sequences = tokenizer.texts_to_sequences(df_val['text'])
test_sequences = tokenizer.texts_to_sequences(df_test['text'])

# b. Padding: Đảm bảo các chuỗi có cùng độ dài
X_train_pad = pad_sequences(train_sequences, maxlen=max_len, padding='post') #
X_val_pad = pad_sequences(val_sequences, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(test_sequences, maxlen=max_len, padding='post')

# Thông tin cần thiết cho Embedding Layer
vocab_size = len(tokenizer.word_index) + 1 #
print(f"Vocab size: {vocab_size}")

from tensorflow.keras.layers import Embedding, LSTM #
from tensorflow.keras.callbacks import EarlyStopping #

# 2. Tạo ma trận trọng số cho Embedding Layer từ Word2Vec
# embedding_dim đã được xác định là 100 ở Nhiệm vụ 2
embedding_matrix = np.zeros((vocab_size, embedding_dim)) #
for word, i in tokenizer.word_index.items(): #
    if i >= vocab_size:
        continue
    if word in w2v_model.wv: #
        embedding_matrix[i] = w2v_model.wv[word] #

# 3. Xây dựng mô hình Sequential với LSTM
lstm_model_pretrained = Sequential([ #
    Embedding( #
        input_dim=vocab_size, #
        output_dim=embedding_dim, #
        weights=[embedding_matrix], # Khởi tạo trọng số [cite: 285]
        input_length=max_len, #
        trainable=False # Dóng băng lớp Embedding 
    ), #
    LSTM(128, dropout=0.2, recurrent_dropout=0.2), #
    Dense(num_classes, activation='softmax') #
])

# 4. Compile, huấn luyện và đánh giá
lstm_model_pretrained.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #

es_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) #

history_lstm_pre = lstm_model_pretrained.fit(
    X_train_pad, Y_train_oh,
    epochs=50, # Chọn số epoch lớn, dùng EarlyStopping để dừng sớm
    batch_size=32,
    validation_data=(X_val_pad, Y_val_oh),
    callbacks=[es_callback],
    verbose=0
) #

# Đánh giá trên tập test
loss_lstm_pre, _ = lstm_model_pretrained.evaluate(X_test_pad, Y_test_oh, verbose=0)
y_pred_lstm_pre = np.argmax(lstm_model_pretrained.predict(X_test_pad, verbose=0), axis=1)

report_lstm_pre = classification_report(y_test, y_pred_lstm_pre, target_names=le.classes_, output_dict=True)
print("\n--- Kết quả Nhiệm vụ 3 (Embedding Pre-trained + LSTM) ---")
print(classification_report(y_test, y_pred_lstm_pre, target_names=le.classes_))
f1_lstm_pre = report_lstm_pre['macro avg']['f1-score']

# 1. Xây dựng mô hình
lstm_model_scratch = Sequential([ #
    Embedding( #
        input_dim=vocab_size, #
        output_dim=100, # Chọn embedding_dim = 100 [cite: 303]
        input_length=max_len #
        # Không có weights, trainable=True (mặc định) [cite: 310]
    ), #
    LSTM(128, dropout=0.2, recurrent_dropout=0.2), #
    Dense(num_classes, activation='softmax') #
])

# 2. Compile, huấn luyện và đánh giá
lstm_model_scratch.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #

history_lstm_scratch = lstm_model_scratch.fit(
    X_train_pad, Y_train_oh,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_pad, Y_val_oh),
    callbacks=[es_callback],
    verbose=0
) #

# Đánh giá trên tập test
loss_lstm_scratch, _ = lstm_model_scratch.evaluate(X_test_pad, Y_test_oh, verbose=0)
y_pred_lstm_scratch = np.argmax(lstm_model_scratch.predict(X_test_pad, verbose=0), axis=1)

report_lstm_scratch = classification_report(y_test, y_pred_lstm_scratch, target_names=le.classes_, output_dict=True)
print("\n--- Kết quả Nhiệm vụ 4 (Embedding Scratch + LSTM) ---")
print(classification_report(y_test, y_pred_lstm_scratch, target_names=le.classes_))
f1_lstm_scratch = report_lstm_scratch['macro avg']['f1-score']

test_cases = [
    "can you remind me to not call my mom", # Có từ phủ định, phụ thuộc xa
    "is it going to be sunny or rainy tomorrow", # Câu hỏi phức tạp, 2 ý tưởng đối lập
    "find a flight from new york to london but not through paris" # Phủ định trong mệnh đề phụ
]

X_test_cases_pad = pad_sequences(tokenizer.texts_to_sequences(test_cases), maxlen=max_len, padding='post')

print("\n--- Phân tích Định Tính ---")
for text in test_cases:
    idx = test_cases.index(text)
    
    # Dự đoán (nhãn số)
    pred_lr = tfidf_lr_pipeline.predict([text])[0]
    pred_w2v = np.argmax(model_avg_w2v.predict(X_test_avg[idx].reshape(1, -1), verbose=0), axis=1)[0]
    pred_lstm_pre = np.argmax(lstm_model_pretrained.predict(X_test_cases_pad[idx].reshape(1, -1), verbose=0), axis=1)[0]
    pred_lstm_scratch = np.argmax(lstm_model_scratch.predict(X_test_cases_pad[idx].reshape(1, -1), verbose=0), axis=1)[0]
    
    # Dự đoán (nhãn chuỗi)
    label_lr = le.inverse_transform([pred_lr])[0]
    label_w2v = le.inverse_transform([pred_w2v])[0]
    label_lstm_pre = le.inverse_transform([pred_lstm_pre])[0]
    label_lstm_scratch = le.inverse_transform([pred_lstm_scratch])[0]
    
    print(f"\nCâu: {text}")
    print(f"  LR (TF-IDF): {label_lr}")
    print(f"  W2V Avg: {label_w2v}")
    print(f"  LSTM Pre-trained: {label_lstm_pre}")
    print(f"  LSTM Scratch: {label_lstm_scratch}")