import argparse
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6" #set this from your env using export 

model = None
tokenizer = None
app = Flask(__name__)

access_token = 'your hf token ' #os.environ["HF_ACCESS_TOKEN"] 


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
            print("GPU available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("Current device:", torch.cuda.current_device())
                print("Device name:", torch.cuda.get_device_name())
                print("Device capability:", torch.cuda.get_device_capability())
                print("Device memory (GB):", torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3))

            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", token=access_token).to("cuda:0")
        elif strategy == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quantization_config, token=access_token)
        elif strategy == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, token=access_token)
        elif strategy == "flash":
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", token=access_token).to("cuda:0")
        else:
            raise ValueError("Invalid strategy specified")
    model.eval()  # Set the model to evaluation mode
    print("Done loading the model...")
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=512, temperature=0.8):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.route('/complete', methods=['POST'])
def complete():
    data = request.json
    prompt = data['text']
  # process yopur prompt here, add prompt engineering, chain-of-thought...
    max_length = int(data.get('length', 512))
    temperature = float(data.get('temperature', 0.8))

    try:
        model, tokenizer = load_model_and_tokenizer(args.strategy)
        generated_text = generate_text(prompt, model, tokenizer, max_length=max_length, temperature=temperature)
        return jsonify({"completion": {"text": generated_text}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server with different model loading strategies")
    parser.add_argument('--strategy', type=str, choices=['cpu', 'gpu', '8bit', '4bit', 'flash'], required=True, help='The strategy to load the model')
    args = parser.parse_args()
    print("Listening...")
    app.run(port=5010, host="0.0.0.0", debug=False)
