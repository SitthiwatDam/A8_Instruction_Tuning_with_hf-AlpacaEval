import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ",device)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    './app/model',
    device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained('./app/model')

# create a text generation pipeline
text_generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    pad_token_id = tokenizer.eos_token_id,
    max_new_tokens = 200,
    device_map = 'auto',
)

# generate prompt for the model
def Generate_prompt_input(text):
	
	if 'input' in text.keys():
		return f"""
Below is an instruction for an instruction-tuning task, paired with an input that provides further context. Write a response that appropriately aligns with the provided instruction.

### Instruction:
{text['instruction']}

### Input:
{text['input']}

### Response:
""".strip()
			
	else:
		return f"""
Below is an instruction for an instruction-tuning task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{text['instruction']}

### Response:
""".strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get-response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    print("User message:   ",user_message)
    
    # Generate a response using the text generation pipeline
    bot_response = text_generator(Generate_prompt_input(user_message))

    # Extracting relevant information from source documents
    source_docs_info = []
    if bot_response['source_documents']:
        for doc in bot_response['source_documents']:
            doc_info = {
                'source': doc.metadata['source'],
                'file_path': doc.metadata['file_path'],
                'page': doc.metadata['page']
            }
            source_docs_info.append(doc_info)
    print("ChatBot is response:   ",bot_response['answer'],"\n", source_docs_info)
    return jsonify({'message': bot_response['answer'], 'source_documents': source_docs_info})




if __name__ == '__main__':
    app.run(debug=True)
