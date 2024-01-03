# Run LLMs on Your CPU with Llama.cpp: A Step-by-Step Guide

[![Watch on GitHub](https://img.shields.io/github/watchers/awinml/llama-cpp-python-bindings.svg?style=social)](https://github.com/awinml/llama-cpp-python-bindings/watchers)
[![Star on GitHub](https://img.shields.io/github/stars/awinml/llama-cpp-python-bindings.svg?style=social)](https://github.com/awinml/llama-cpp-python-bindings/stargazers)
[![Tweet](https://img.shields.io/twitter/url/https/github.com/awinml/llama-cpp-python-bindings.svg?style=social)](https://twitter.com/intent/tweet?url=https%3A%2F%2Fawinml.github.io%2Fllm-ggml-python%2F&via=awinml&text=Run%20LLMs%20on%20Your%20CPU%20with%20Llama.cpp%3A%20A%20Step-by-Step%20Guide&hashtags=llamacpp%2C%20llms)


This respository contains the code for the all the examples mentioned in the article, [How to Run LLMs on Your CPU with Llama.cpp: A Step-by-Step Guide](https://awinml.github.io/llm-ggml-python/
).

A simple example that uses the [Zephyr-7B-β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) LLM for text generation:

```python
import os
import urllib.request
from llama_cpp import Llama


def download_file(file_link, filename):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")


# Dowloading GGML model from HuggingFace
ggml_model_path = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf"
filename = "zephyr-7b-beta.Q4_0.gguf"

download_file(ggml_model_path, filename)


llm = Llama(model_path="zephyr-7b-beta.Q4_0.gguf", n_ctx=512, n_batch=126)


def generate_text(
    prompt="Who is the CEO of Apple?",
    max_tokens=256,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text


def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{input}<|im_end|>"""
    return chat_prompt_template


prompt = generate_prompt_from_template(
    "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
)

generate_text(
    prompt,
    max_tokens=356,
)
```
