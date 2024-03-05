import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import pytesseract

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# Function to convert image to text using OCR
def imagetotext(image_path):
    image = Image.open(image_path)
    text = pytesseract.imagetostring(image)
    return text

# Function to process text through GPT-J
def processwithgpt(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
image_path = "path_to_your_handwritten_image.jpg"
text_from_image = imagetotext(image_path)
generated_input = processwithgpt(text_from_image)
print("Generated input from GPT-J:", generated_input)
