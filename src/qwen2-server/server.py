from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed
from threading import Thread
import random
import os
import uuid

from contextlib import asynccontextmanager
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
import uvicorn

DEFAULT_CKPT_PATH = os.path.dirname(os.path.abspath(__file__))

def load_model_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, resume_download=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map='auto',
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 512
    return model, tokenizer

def chat_stream(model, tokenizer, query, history):
    conversation = [
        {'role': 'system', 'content': ''},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    index = 0
    for new_text in streamer:
        yield index, new_text
        index += 1

config = {}
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config

    config['checkpoint_path'] = DEFAULT_CKPT_PATH
    config['seed'] = random.randint(0, 2**32 - 1)

    model, tokenizer = load_model_tokenizer(config['checkpoint_path'])
    set_seed(config['seed'])

    config['model'] = model
    config['tokenizer'] = tokenizer
    config['history'] = []

    yield

app = FastAPI(lifespan=lifespan)

class NewContinueParam(BaseModel):
    text: str

class NewContinueResponse(BaseModel):
    token: str

class ReadStreamParam(BaseModel):
    token: str

class ReadStreamResponse(BaseModel):
    index: int
    text: str

@app.post('/novel/new')
def new_continue(param: NewContinueParam):
    global config, sessions

    session_id = uuid.uuid4().hex
    history = config['history']
    stream = chat_stream(config['model'], config['tokenizer'], param.text, history)

    sessions[session_id] = {
        'history': history,
        'stream': stream,
        'query': param.text,
        'partial_text': '',
    }

    return NewContinueResponse(token=session_id)

@app.post('/novel/read')
def read_stream(param: ReadStreamParam):
    global config, sessions

    if param.token not in sessions:
        raise HTTPException(status.HTTP_400_BAD_REQUEST)

    session = sessions[param.token]

    index, text = -1, ''

    try:
        index, text = next(session['stream'])
        sessions[param.token]['partial_text'] += text
    except StopIteration:
        text = session['partial_text']
        sessions[param.token]['history'].append((session['query'], text))

    return ReadStreamResponse(index=index, text=text)

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')
