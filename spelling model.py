import torch
from transformers import GPTJForCausalLM, GPTJTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-J 6B model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = GPTJTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)

# Define path to your grammar correction dataset
train_file = "grammar_correction_train.txt"
val_file = "grammar_correction_val.txt"

# Load dataset and preprocess
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=val_file,
    block_size=128
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gptj_grammar_correction",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    warmup_steps=500,
    save_total_limit=2,
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Create Trainer instance and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Save the fine-tuned model
trainer.save_model("./gptj_grammar_correction_finetuned")

# Example usage of the fine-tuned model for grammar correction
text_with_errors = "He do not likes apples."
inputs = tokenizer(text_with_errors, return_tensors="pt")
outputs = model.generate(**inputs)
corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Original Text:", text_with_errors)
print("Corrected Text:", corrected_text)
