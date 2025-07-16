#!/usr/bin/env python
# coding: utf-8

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
max_seq_length = 4096
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "microsoft/Phi-3.5-mini-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
#adapter_path = "../../Models/lora_model_phi3_chandler"
#adapter_path = "../../Models/lora_model_phi3_phoebe"
#adapter_path = "../../Models/lora_model_phi3_rachel"
#model.load_adapter(adapter_path, adapter_name="chandler")
#model.set_adapter("chandler")
#model.enable_adapters()

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
   # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass



from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import uvicorn
from threading import Thread


# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


FastLanguageModel.for_inference(model) # Enable native 2x faster inference


# Assuming `FastLanguageModel` is the inference model wrapper you're using
#from your_model_library import FastLanguageModel, tokenizer  # Adjust the imports as necessary

# Initialize the model and tokenizer for inference
#model = FastLanguageModel.for_inference("path_to_your_model")  # Adjust the model loading if needed
#model = model.to("cuda")  # Make sure to send the model to the GPU if available
print("Model loaded successfully!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# Create the FastAPI app
app = FastAPI()

# Define Pydantic model for message input
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.post("/generate/chat/completions")
async def generate_chat_response(request: ChatRequest):
    try:
        # Convert incoming messages to the format your tokenizer expects
        messages_0 = request.messages
        
        # Convert Message objects to JSON-style dicts
        messages = [{"role": message.role, "content": message.content} for message in messages_0]

      #  print("\nXXXXXXXXX messages: ", messages, "XXXXXXXX\n")


        # Apply the chat template (tokenizing, adding generation prompts, etc.)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Ensures it's ready for generation
            return_tensors="pt"
        ).to("cuda")
     #   print("\nYYYYYYYYY messages: ", inputs, "YYYYYYYYYY\n")

       # texts = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
     #   print("\nZZZZZZZZZZZZZZZZ texts: ", texts, "ZZZZZZZZZZ\n")
        
     #   print("inputs: ", inputs)
        tokenizer.eos_token_id = 32007
        # Perform model inference (generate response)
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=500,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id= 32007,
            repetition_penalty=1.01,
            do_sample=True,
            top_p=0.9
        )
      #  input_length = inputs.shape[1]
     #   outputs = model.generate(**inputs, max_length=input_length+100, pad_token_id=tokenizer.eos_token_id, eos_token_id= 32007, temperature = 1.8)
     #   full_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
      #  full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Decode the output tokens to text
       # decoded_response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)

        xx = outputs[0][inputs.shape[-1]:]
        decoded_response = tokenizer.decode(xx, skip_special_tokens=True)
        
     #   print("\nPPPPPPP full_output: ", full_output, "PPPPPP\n")
     #   print("decoded_response: ", decoded_response)
        # Return the generated response
        print("\nDecoded: ", decoded_response, "--- END_OF_MSG\n")
        return {"choices": [{"message": {"content": decoded_response}}]}

    except Exception as e:
        # Handle errors if any
        raise HTTPException(status_code=500, detail=str(e))


# uvicorn app:app --host 0.0.0.0 --port 8000
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start the server in a separate thread
thread = Thread(target=run_server)
thread.start()