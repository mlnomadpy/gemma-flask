import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from flask import Flask, request, jsonify
import os
import threading
import queue
import time
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

model = None
tokenizer = None
app = Flask(__name__)

access_token = 'your hf token'

request_queue = queue.Queue()
response_dict = {}
batch_size = 4
batch_interval = 0.1  # Time to wait before processing the batch

def load_model_and_tokenizer(strategy, model_path="google/gemma-2b-it"):
    global model
    global tokenizer
    global access_token

    print("Loading the Model")
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
    
    if not model:
        if strategy == "cpu":
            model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
        elif strategy == "gpu":
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", token=access_token).to("cuda:0")
        elif strategy == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quantization_config, token=access_token)
        elif strategy == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, token=access_token)
        elif strategy == "flash":
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", token=access_token)
        else:
            raise ValueError("Invalid strategy specified")
    model.eval()  
    print("Done loading the model...")
    return model, tokenizer

def generate_text_batch(prompts, model, tokenizer, max_length=512, temperature=0.8):
    input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature)
    return [tokenizer.decode(output_ids[i], skip_special_tokens=True) for i in range(len(prompts))]

def process_batch():
    while True:
        time.sleep(batch_interval)
        if not request_queue.empty():
            batch = []
            while not request_queue.empty() and len(batch) < batch_size:
                batch.append(request_queue.get())
            
            prompts = [item['prompt'] for item in batch]
            request_ids = [item['request_id'] for item in batch]
            max_length = batch[0]['max_length']
            temperature = batch[0]['temperature']
            
            try:
                model, tokenizer = load_model_and_tokenizer(args.strategy)
                generated_texts = generate_text_batch(prompts, model, tokenizer, max_length=max_length, temperature=temperature)
                for request_id, generated_text in zip(request_ids, generated_texts):
                    response_dict[request_id] = {"completion": {"text": generated_text}}
            except Exception as e:
                for request_id in request_ids:
                    response_dict[request_id] = {"error": str(e)}

threading.Thread(target=process_batch, daemon=True).start()

@app.route('/complete', methods=['POST'])
def complete():
    data = request.json
    prompt = data['text']
    max_length = int(data.get('length', 512))
    temperature = float(data.get('temperature', 0.8))
    request_id = len(response_dict)
    
    request_queue.put({
        'prompt': prompt,
        'max_length': max_length,
        'temperature': temperature,
        'request_id': request_id
    })

    while request_id not in response_dict:
        time.sleep(1)
        
    # this is just a demo, not a production code
    # we are not saving this in any database
    # another strategy will be to sub to your db 
    # push the answer to the db whenever your model finished processing the text
    # one their is a change in the databse it will send the response to the user
    # check firebase real-time database 
    # you will use
        # chat_id
        # message_id
        # owner_id
    return jsonify(response_dict.pop(request_id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server with different model loading strategies")
    parser.add_argument('--strategy', type=str, choices=['cpu', 'gpu', '8bit', '4bit', 'flash'], required=True, help='The strategy to load the model')
    args = parser.parse_args()
    app.run(port=5010, host="0.0.0.0", debug=False)
