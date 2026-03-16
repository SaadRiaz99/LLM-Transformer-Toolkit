# def main():
#     print("Hello from minigpt!")


# if __name__ == "__main__":
#     main()
import torch 
import torch.nn as nn
import torch.nn.functional as f
import random

# from transformer import SelfAttentionHead, MultiHeadAttention, FeedForward, Block

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

