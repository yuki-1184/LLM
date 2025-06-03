import fugashi
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    japanese_text = "走れメロスは、太宰治が1940年に発表した短編小説です。王に人質として差し出された親友を救うため、困難な状況でも走り続けるメロスの姿を描いています。友情や信頼の大切さを伝える作品として知られています。"
    english_text = (
        "Run, Melos is a short story written by Osamu Dazai and published in 1940. "
        "It depicts the figure of Melos, who continues to run despite difficult circumstances "
        "in order to save his friend who was offered as a hostage to the king. "
        "The story is known for conveying the importance of friendship and trust."
    )
    
    tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v2")
    print(tokenizer.tokenize(japanese_text))
    print(f"The number of token length {len(tokenizer.tokenize(japanese_text))}")
    
    print(tokenizer.tokenize(english_text))
    print(f"The number of token length {len(tokenizer.tokenize(english_text))}")
    
    
if __name__ == "__main__":
    main()