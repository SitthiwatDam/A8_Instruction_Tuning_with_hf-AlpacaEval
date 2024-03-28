import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ",device)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    './model',
    device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained('./model')

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
def Generate_prompt_input(inst,input = None):
	
	if inst:
		return f"""
Below is an instruction for an instruction-tuning task, paired with an input that provides further context. Write a response that appropriately aligns with the provided instruction.

### Instruction:
{inst}

### Input:
{input}

### Response:
""".strip()
			
	else:
		return f"""
Below is an instruction for an instruction-tuning task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{inst}

### Response:
""".strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get-response', methods=['POST'])
def get_response():
    user_instruction = request.json['instruction']
    user_input = request.json['prompt']
    print("User inputs:   ",user_instruction,'\n',user_input)
    
    # Generate a response using the text generation pipeline
    result = text_generator(Generate_prompt_input(user_instruction,user_input))
    bot_response = result[0]['generated_text'].split("### Response:\n")[-1]

    print("ChatBot is response:   ",bot_response,"\n")
    return jsonify({'message': bot_response})




if __name__ == '__main__':
    app.run(debug=True)
