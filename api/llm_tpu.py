from fastapi import Form
from fastapi.responses import JSONResponse, StreamingResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper
from typing import Optional
import argparse
import os,sys
import re
from pydantic import BaseModel, Field

app_name = "llm_tpu"

def match_model(model_name, patterns):
    model_name = re.sub(r'\W', '', model_name.lower().replace('_', ''))
    normalized_patterns = [re.sub(r'\W', '', pattern.lower().replace('_', '')) for pattern in patterns]
    for x in range(len(normalized_patterns)):
        if normalized_patterns[x] in model_name:
            return x
    return None

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        import importlib

        self.models = {}
        self.models_list = os.listdir("bmodels")
        self.tokenizer_dict = {}

        for root, dirs, files in os.walk('./models'):
            if 'token_config' in dirs:
                full_path = os.path.join(root, 'token_config')
                # 分割路径
                parts = root.split(os.sep)
                # 假设模型目录总是位于 './models' 目录下的第一级子目录
                if len(parts) > 2 and parts[1] == 'models':
                    model_name = parts[2]
                else:
                    model_name = os.path.basename(root)
                self.tokenizer_dict[model_name] = full_path


        args = argparse.Namespace(
            devid='0',
            temperature=1.0,
            top_p=1.0,
            repeat_penalty=1.0,
            repeat_last_n=32,
            max_new_tokens=1024,
            generation_mode="greedy",
            prompt_mode="prompted",
            enable_history=False,
            lib_path='',
            decode_mode='basic'
        )

        mm = list(self.tokenizer_dict.keys())
        nn = list(self.tokenizer_dict.values())

        for model_name in self.models_list:
            id = match_model(model_name, mm)
            if id is None:
                print(f"Model {model_name} does not match any available model.")
                continue
            tokenizer_path = nn[id]
            args.model_path = f"bmodels/{model_name}"
            args.tokenizer_path = tokenizer_path

            if 'chat' in sys.modules:
                del sys.modules['chat']

            sys.path.insert(0, f"models/{mm[id]}/python_demo")

            module_name = f"models.{mm[id]}.python_demo.pipeline"
            module = importlib.import_module(module_name)

            model_class = getattr(module, mm[id])

            self.models[f"{model_name}"] = model_class(args)

            sys.path.remove(f"models/{mm[id]}/python_demo")

        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.models

router = AppInitializationRouter(app_name=app_name)

class ChatRequest(BaseModel):
    model: str = Field("qwen1.5-1.8b_int8_1dev_seq1280.bmodel", description="bmodel file name")
    messages: list = Field([{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"hello"}], description="Chat history")
    stream: bool = Field(False, description="Stream response")

@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest):
    slm = router.models[request.model]
    tokens = slm.tokenizer.apply_chat_template(request.messages, tokenize=True, add_generation_prompt=True)


    token = slm.model.forward_first(tokens)
    output_tokens = [token]

    if not isinstance(slm.EOS, list):
        slm.EOS = [slm.EOS]

    if request.stream:
        def generate_responses():
            yield '{"choices": [{"delta": {"role": "assistant", "content": "'
            while True:
                token = slm.model.forward_next()
                if token in slm.EOS or slm.model.token_length >= slm.SEQLEN:
                    break
                output_tokens.append(token)
                response_text = slm.tokenizer.decode([token])
                yield response_text
            yield '"}}]}'

        return StreamingResponse(generate_responses(), media_type="application/json")
    
    else:
        while True:
            token = slm.model.forward_next()
            if token in slm.EOS or slm.model.token_length >= slm.SEQLEN:
                break
            output_tokens += [token]
        slm.answer_cur = slm.tokenizer.decode(output_tokens)

        return JSONResponse({"choices": [{"delta": {"role": "assistant", "content": slm.answer_cur}}]})
    

# curl --no-buffer -X 'POST' \
#   'http://localhost:8000/llm_tpu/v1/chat/completions' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "model": "qwen1.5-1.8b_int8_1dev_seq1280.bmodel",
#   "messages": [
#     {
#       "role": "system",
#       "content": "You are a helpful assistant."
#     },
#     {
#       "role": "user",
#       "content": "hello"
#     }
#   ],
#   "stream": true
# }'