import time
from mlx_lm import load, generate
import os

def run_inference_test():
    """
    Loads a quantized GGUF model from a LOCAL directory using mlx-lm
    and runs a test generation to validate inference capability.
    """
    model_path = "./Meta-Llama-3-8B-Instruct-4bit"

    prompt = """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the core concept behind simulated annealing? Explain it like I'm a programmer trying to understand a new optimization algorithm.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    print("--- Eidetic AI Local Inference Test ---")
    print(f"Model Path: {os.path.abspath(model_path)}")
    
    if not os.path.isdir(model_path):
        print(f"\033[91mError: Model directory not found at '{model_path}'\033[0m")
        print("Please ensure the model folder is in the root of the EideticAI project directory.")
        return

    print("Status: Loading model and tokenizer from local directory...")
    
    start_time = time.time()
    try:
        model, tokenizer = load(model_path)
        load_time = time.time() - start_time
        print(f"Status: Model loaded successfully in {load_time:.2f} seconds.")
    except Exception as e:
        print(f"\033[91mError loading model: {e}\033[0m")
        return

    print("-" * 35)
    print("Prompt:")
    print(prompt.strip())
    print("-" * 35)
    
    print("Status: Generating response...")
    start_gen_time = time.time()
    
    # The final, robust solution is to not specify temperature and let the
    # library use its default value, avoiding the API inconsistency.
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=False
    )
    
    gen_time = time.time() - start_gen_time
    tokens_per_sec = 256 / gen_time if gen_time > 0 else float('inf')

    print("Generated Response:")
    # The model may sometimes continue generating beyond a logical stop.
    # We print the response as-is to see the raw output.
    print(response)
    print("-" * 35)
    print("Inference complete.")
    print(f"Generation Time: {gen_time:.2f} seconds")
    print(f"Performance: {tokens_per_sec:.2f} tokens/second")
    print("-" * 35)

if __name__ == "__main__":
    run_inference_test()