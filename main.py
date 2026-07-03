import torch

def main():
    print("Hello from theo-conductor!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


if __name__ == "__main__":
    main()
