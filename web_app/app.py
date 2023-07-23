import time
from aiohttp import web
import aiohttp_jinja2
import jinja2
from llama_cpp import Llama

MODEL_PATH = "../ggml-vicuna-7b-1.1-q4_1.bin"
llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_batch=126)

def generate_text(prompt="Who is the CEO of Apple?", max_tokens=300, temperature=0.1, top_p=0.5, echo=False, stop=[]):
    start_time = time.time()
    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, echo=echo, stop=stop)
    end_time = time.time()
    time_taken = end_time - start_time
    output_text = output["choices"][0]["text"].strip()
    return output_text, time_taken

async def get_inference(request):
    data = await request.json()
    print("Incoming request:\n", data)
    prompt = data.get('prompt')
    max_tokens = data.get('max_tokens', 2000)
    response_text, inference_time = generate_text(prompt, max_tokens=max_tokens)
    minutes, seconds = divmod(inference_time, 60)
    time_taken_formatted = f"{int(minutes)}m {int(seconds)}s"
    response_data = {
        "inference_time": time_taken_formatted,
        "response_text": response_text
    }
    print("response_data:\n",response_data)
    return web.json_response(response_data)

@aiohttp_jinja2.template('index.html')
async def index(request):
    return {}

app = web.Application()
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))

app.router.add_post('/generate_text', get_inference)
app.router.add_get('/', index)
app.router.add_static('/static/', path='static', name='static')

if __name__ == "__main__":
    web.run_app(app, host='0.0.0.0')
