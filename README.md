# gemma-flask

This one of the flask apis we talked about in the talk 
I will improve it later but just use it right now

pip install flask torch transformers bitsandbytes accelerate

run this in your terminal

export CUDA_VISIBLE_DEVICES="7" # this is your gpu id get it either from nvidia-smi or gpustat

python app.py --strategy gpu # you can set this to cpu gpu 8bit 4bit flash depending on your needs chack the talk

communicate with it from the terminal using

curl -X POST -H "Content-Type: application/json" -d '{"text": "the earth is flat because", "length": 10, "temperature": 0.8}' http://localhost:5010/complete

or check my other vs-gemma
