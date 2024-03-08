import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import pytesseract
from torchvision.io import read_image

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("model loaded, device: ", device)

model = model.to(device)

# Move the model to the deivec

# Function to convert image to text using OCR
def imagetotext(image_path):
    image = Image.open(image_path)
    text = pytesseract.imagetostring(image)
    return text

# Function to process text through GPT-J
def processwithgpt(text):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
# image_path = os.listdir("C:/Users/noodle/Documents/GitHub/ai---project/test_v2/test")
# text_from_image = imagetotext(image_path)
#
# generated_input = processwithgpt(text_from_image)
# print("Generated input from GPT-J:", generated_input)


other_text = processwithgpt("what is the best drug ")
print("Other text from GPT-J: ", other_text)
