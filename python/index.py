from transformers import AutoTokenizer, AutoModel

# Load the pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('/models/LLaMa/7B')

# Encode some input text using the BERT tokenizer and model
input_text = 'Hello, how are you today?'
input_ids = tokenizer.encode(input_text, return_tensors='pt')
outputs = model(input_ids)

# Print the model outputs
print(outputs)
