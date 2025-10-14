Lab 4: Word Embeddings với Word2Vec
Dự án này tập trung vào việc khám phá và ứng dụng Word Embeddings, một kỹ thuật biểu diễn từ dưới dạng vector dày đặc (dense vector) để nắm bắt các mối quan hệ ngữ nghĩa, cú pháp và ngữ cảnh. Chúng ta sẽ sử dụng thư viện gensim để làm việc với các mô hình được huấn luyện trước (pre-trained) và tự huấn luyện một mô hình nhỏ.

🚀 Các bước thực hiện
Xây dựng Lớp WordEmbedder: Xây dựng một lớp Python để đóng gói các chức năng của mô hình word embedding, bao gồm việc tải model, lấy vector của từ, tính độ tương đồng cosine, và tìm các từ đồng nghĩa.

Tải và Sử dụng Model Pre-trained: Tải model glove-wiki-gigaword-50 từ gensim-data. Đây là model 50 chiều được huấn luyện trên kho dữ liệu lớn từ Wikipedia.

Embedding cho Tài liệu: Triển khai phương pháp đơn giản nhưng hiệu quả để biểu diễn một tài liệu bằng cách lấy trung bình vector của các từ trong tài liệu đó.

Huấn luyện Model Tùy chỉnh (Bonus): Xây dựng một script (lab4_embedding_training_demo.py) để huấn luyện một mô hình Word2Vec từ đầu trên một tập dữ liệu nhỏ, cụ thể để so sánh với model pre-trained.

Mở rộng với Spark (Advanced): Triển khai một ví dụ sử dụng PySpark MLlib để huấn luyện mô hình Word2Vec trên dữ liệu lớn, minh họa khả năng mở rộng của kỹ thuật này.

⚙️ Hướng dẫn chạy code
Để thực thi lại các thử nghiệm trong project này, vui lòng làm theo các bước sau:

Clone Repository

Bash

git clone https://github.com/quocskhanh/NLP-DL.git
cd NLP-DL/Lab\ 4
Cài đặt Dependencies
Đảm bảo bạn đã cài đặt tất cả các thư viện cần thiết.

Bash

pip install -r requirements.txt
Thực thi Script Chính
Script chính sẽ thực hiện các tác vụ như tìm từ đồng nghĩa, tính độ tương đồng và embedding câu văn sử dụng model pre-trained.

Bash

python -m test.lab4_test
(Lần chạy đầu tiên sẽ mất thời gian để tải model glove-wiki-gigaword-50 về máy.)

📊 Phân tích kết quả
1. Độ tương đồng và Từ đồng nghĩa (Model Pre-trained)
Model glove-wiki-gigaword-50 cho thấy khả năng nắm bắt ngữ nghĩa rất tốt:

Độ tương đồng: Kết quả cho thấy king có độ tương đồng cao với queen (quan hệ ngữ nghĩa) và cũng khá tương đồng với man (quan hệ thuộc tính). Điều này chứng tỏ model đã học được các mối quan hệ phức tạp từ dữ liệu lớn.

similarity('king', 'queen') ≈ 0.8

similarity('king', 'man') ≈ 0.6

Từ đồng nghĩa: Khi tìm các từ tương tự nhất với computer, model trả về các từ như software, technology, internet, machine. Đây không chỉ là các từ đồng nghĩa trực tiếp mà còn là các từ liên quan chặt chẽ trong cùng một trường ngữ nghĩa, cho thấy khả năng nắm bắt ngữ cảnh của model.

2. Phân tích Biểu đồ Trực quan hóa (Giả định)
Nếu thực hiện trực quan hóa các vector từ bằng t-SNE hoặc PCA, chúng ta có thể kỳ vọng thấy các kết quả sau:

Các cụm từ có ý nghĩa: Các từ có quan hệ gần gũi sẽ được nhóm lại với nhau. Ví dụ, các quốc gia (vietnam, japan, korea) sẽ tạo thành một cụm; các từ chỉ hoàng gia (king, queen, prince, princess) sẽ tạo thành một cụm khác.

Cụm từ thú vị: Một quan sát thú vị là các vector thường phản ánh "thiên kiến" (bias) trong dữ liệu huấn luyện. Ví dụ, trong không gian vector, phép toán king - man + woman thường cho ra kết quả gần với queen. Điều này cho thấy model đã học được quan hệ về giới tính một cách tường minh.

3. So sánh Model Pre-trained và Model Tự Huấn luyện
Model Pre-trained (GloVe): Có vốn từ vựng khổng lồ và kiến thức tổng quát rộng. Nó hoạt động rất tốt trên các tác vụ phổ thông và các phép loại suy (analogy) kinh điển.

Model Tự Huấn luyện: Có vốn từ vựng giới hạn trong tập dữ liệu huấn luyện. Do đó, nó có thể không thực hiện tốt các tác vụ tổng quát. Tuy nhiên, ưu điểm của nó là có thể học được các ngữ nghĩa đặc trưng và chuyên biệt của lĩnh vực đó. Ví dụ, nếu huấn luyện trên kho văn bản y tế, từ "cell" sẽ có ngữ nghĩa gần với "tissue", "organ" hơn là "phone".

🚧 Khó khăn và Giải pháp
Vấn đề: ModuleNotFoundError khi chạy script

Khó khăn: Ban đầu, việc chạy script từ các thư mục con (test/) gây ra lỗi import vì Python không tìm thấy thư mục src/.

Giải pháp: Áp dụng hai giải pháp tiêu chuẩn cho dự án Python:

Thêm các tệp __init__.py trống vào tất cả các thư mục (src, test, src/representations) để khai báo chúng là các "package".

Chạy script dưới dạng module từ thư mục gốc của dự án (python -m test.lab4_test) để đảm bảo Python path được thiết lập chính xác.

Vấn đề: Quy trình Git khi đẩy code lên repository đã có sẵn

Khó khăn: Repository trên GitHub đã có sẵn nội dung và nhánh master, tronGemini
