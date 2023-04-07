from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the path to the custom model
model_path = "./models/LLaMa/13B"

print("Tokenizer.....")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Sharder.....")

model = AutoModelForCausalLM.from_pretrained(model_path)


# Function to generate text using the model
def generate_text(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print("Input IDs:", input_ids)
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_p=0.9,  # Add top-p nucleus sampling
        do_sample=True,  # Enable sampling
    )

    generated_texts = []
    for generated_sequence in output_sequences:
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        generated_texts.append(text)

    return generated_texts

# Example usage
if __name__ == "__main__":
    prompt = "write me a short story"
    generated_texts = generate_text(prompt, max_length=100, num_return_sequences=3)

    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i + 1}:")
        print(text)
        print("\n")