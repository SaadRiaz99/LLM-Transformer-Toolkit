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

data = [
    "Pakistan is a beautiful country with a rich history and culture.",
    "he people of Pakistan are very kind and hospitable.",
    "Pakistan has stunning mountains, rivers, and valleys.",
    "The country has a strong and brave army.",
    "Pakistani youth are talented and hardworking.",
    "Pakistan has unique traditions and cultural diversity.",
    "The land of Pakistan is fertile and good for agriculture.",
    "Many languages and cultures live together in harmony in Pakistan.",
    "Pakistan has shown courage and strength in difficult times.",
    "We are proud of Pakistan and hope for its bright future. 🇵🇰"
]

datas = [s + "<End>" for s in data ]
text = "".join(datas)
print(text)