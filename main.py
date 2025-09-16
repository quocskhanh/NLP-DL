from src.preprocessing.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()

    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    for s in sentences:
        print(f"\nOriginal: {s}")
        print(f"Tokens: {simple_tokenizer.tokenize(s)}")
