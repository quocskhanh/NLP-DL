# test/lab4_spark_word2vec_demo.py

import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace

def main():
    """
    Minh họa cách đào tạo mô hình Word2Vec bằng PySpark. [cite: 72]
    """
    # 1. Khởi tạo Spark Session [cite: 83]
    # 'local[*]' sử dụng tất cả các lõi CPU có sẵn trên máy cục bộ
    spark = SparkSession.builder \
        .appName("SparkWord2VecDemo") \
        .master("local[*]") \
        .getOrCreate()

    # Tạo tệp dữ liệu C4 giả
    data_folder = 'data'
    input_file = os.path.join(data_folder, 'c4-train.00000-of-01024-30K.json')
    os.makedirs(data_folder, exist_ok=True)
    
    # Dữ liệu JSON trên mỗi dòng [cite: 76, 86]
    sample_json_data = [
        '{"text": "Apache Spark is a unified analytics engine for large-scale data processing."}',
        '{"text": "It provides high-level APIs in Java, Scala, Python and R."}',
        '{"text": "Spark\'s machine learning library, MLlib, provides a scalable Word2Vec implementation."}',
        '{"text": "A modern personal computer contains a CPU, RAM, and a storage device."}',
        '{"text": "My new laptop is a powerful computer with a fast processor."}',
        '{"text": "The software runs on any modern computing device."}'
    ]
    with open(input_file, 'w') as f:
        for line in sample_json_data:
            f.write(line + '\n')

    # 2. Tải tập dữ liệu từ tệp JSON [cite: 85]
    print(f"Đang đọc dữ liệu từ {input_file}")
    df = spark.read.json(input_file)
    
    # Chỉ quan tâm đến cột 'text' [cite: 88]
    text_df = df.select("text")

    # 3. Tiền xử lý văn bản [cite: 90]
    # Chuyển đổi thành chữ thường, loại bỏ dấu câu và ký tự đặc biệt [cite: 91, 92]
    cleaned_df = text_df.withColumn("cleaned_text", lower(col("text"))) \
                        .withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r'[\p{Punct}]', ''))
    
    # Tokenize (chia) văn bản thành các từ [cite: 108]
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    words_df = tokenizer.transform(cleaned_df)

    # 4. Cấu hình và đào tạo mô hình Word2Vec [cite: 95]
    print("Bắt đầu đào tạo mô hình Word2Vec trên Spark...")
    word2Vec = Word2Vec(vectorSize=100, minCount=1, inputCol="words", outputCol="result")
    model = word2Vec.fit(words_df)
    print("Đào tạo trên Spark hoàn tất.")

    # 5. Minh họa mô hình [cite: 97]
    # Tìm các từ đồng nghĩa (từ tương tự nhất) [cite: 98]
    synonyms = model.findSynonyms("computer", 5)

    print("\n--- 5 từ tương tự nhất với 'computer' (từ mô hình Spark) ---")
    synonyms.show()

    # 6. Dừng phiên Spark [cite: 100]
    spark.stop()
    print("\nPhiên Spark đã dừng.")

if __name__ == "__main__":
    main()