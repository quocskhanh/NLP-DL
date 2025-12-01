import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        # Đưa text về lowercase
        text = text.lower()

        # Tách các dấu câu cơ bản ra khỏi từ
        # Regex: thêm khoảng trắng trước và sau các dấu . , ? !
        text = re.sub(r"([.,?!])", r" \1 ", text)

        # Loại bỏ khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()

        # Split theo khoảng trắng
        tokens = text.split(" ")
        return tokens
