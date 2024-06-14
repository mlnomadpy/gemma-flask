# Google Gemma simple Flask API with huggingface's transformers

This repository contains a Flask API that uses Google Gemma with Hugging Face's Transformers to generate text based on a given prompt. The API allows you to load and serve a language model with different strategies (CPU, 8bit, 4bit, GPU, flash attention).

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Flask App](#running-the-flask-app)
  - [Communicating with the API](#communicating-with-the-api)
- [Code Explanation](#code-explanation)
  - [Importing Required Libraries](#importing-required-libraries)
  - [Initializing Variables and Flask App](#initializing-variables-and-flask-app)
  - [Loading the Model and Tokenizer](#loading-the-model-and-tokenizer)
  - [Generating Text](#generating-text)
  - [Creating the API Endpoint](#creating-the-api-endpoint)

## Introduction

The goal of this project is to create an API endpoint that takes a text prompt and returns a generated continuation of the text using a transformer model. This example demonstrates how to load a language model with different strategies and serve it through a simple web API.

## Prerequisites

Ensure you have the following libraries installed:

```bash
pip install torch flask transformers
```

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/skywolfmo/gemma-flask.git
cd gemma-flask
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Flask App

To run the Flask app, use the following command with the desired model loading strategy:

```bash
python app.py --strategy <strategy>
```

Replace `<strategy>` with one of the following options: `cpu`, `gpu`, `8bit`, `4bit`, `flash`.

### API Endpoint

The API has a single endpoint `/complete` that accepts a POST request with a JSON payload containing the text prompt and optional parameters like length and temperature.

**Request:**

```json
POST /complete
{
  "text": "Once upon a time",
  "length": 100,
  "temperature": 0.8
}
```

**Response:**

```json
{
  "completion": {
    "text": "Once upon a time, there was a magical kingdom..."
  }
}
```

## Usage

### Running the Flask App

To run the Flask app, first, set your GPU ID (if applicable) and then use the following command with the desired model loading strategy:

```bash
export CUDA_VISIBLE_DEVICES="7"  # This is your GPU ID; get it from nvidia-smi or gpustat
python app.py --strategy gpu  # You can set this to cpu, gpu, 8bit, 4bit, flash depending on your needs
```

### Communicating with the API

You can communicate with the API using a tool like `curl`. Here’s an example of how to send a POST request to the `/complete` endpoint:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "the earth is flat because", "length": 10, "temperature": 0.8}' http://localhost:5010/complete
```

This command sends a JSON payload with a text prompt, and optional parameters like length and temperature, to the API and returns the generated text.



## Code Explanation

### Importing Required Libraries

First, we import the necessary libraries.

```python
import argparse
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
```

### Initializing Variables and Flask App

We initialize the global variables for the model and tokenizer, and create an instance of the Flask application.

```python
model = None
tokenizer = None
app = Flask(__name__)

access_token = 'your_hf_token'  # Replace with your Hugging Face access token
```

### Loading the Model and Tokenizer

The function `load_model_and_tokenizer` loads the model and tokenizer based on the specified strategy. It supports loading models on CPU, GPU, and with different quantization techniques.

```python
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
            if torch.cuda.is_available():
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
```

### Generating Text

The `generate_text` function takes a prompt and generates text using the loaded model and tokenizer.

```python
def generate_text(prompt, model, tokenizer, max_length=512, temperature=0.8):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### Creating the API Endpoint

We define an endpoint `/complete` that takes a POST request with the text prompt and other optional parameters like length and temperature. It returns the generated text.

```python
@app.route('/complete', methods=['POST'])
def complete():
    data = request.json
    prompt = data['text']
    max_length = int(data.get('length', 512))
    temperature = float(data.get('temperature', 0.8))

    try:
        model, tokenizer = load_model_and_tokenizer(args.strategy)
        generated_text = generate_text(prompt, model, tokenizer, max_length=max_length, temperature=temperature)
        return jsonify({"completion": {"text": generated_text}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### Running the Flask App

Finally, we parse the command-line arguments to specify the strategy and run the Flask app.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server with different model loading strategies")
    parser.add_argument('--strategy', type=str, choices=['cpu', 'gpu', '8bit', '4bit', 'flash'], required=True, help='The strategy to load the model')
    args = parser.parse_args()
    print("Listening...")
    app.run(port=5010, host="0.0.0.0", debug=False)
```

