from datasets import load_dataset


def main():
    dataset = load_dataset("AhmedSSabir/Japanese-wiki-dump-sentence-dataset", split="train")
    print(dataset)
    

if __name__ == "__main__":
    main()