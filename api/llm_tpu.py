from fastapi import Form
from fastapi.responses import JSONResponse, StreamingResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper
from typing import Optional
import argparse
import os,sys
import re
from pydantic import BaseModel, Field
from difflib import get_close_matches

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
        self.models_list = os.listdir("llm_bmodels")
        tokenizer_dict = {}

        for root, dirs, files in os.walk('./llm_models'):
            if 'token_config' in dirs:
                full_path = os.path.join(root, 'token_config')
                # 分割路径
                parts = root.split(os.sep)
                # 假设模型目录总是位于 './models' 目录下的第一级子目录
                if len(parts) > 2 and parts[1] == 'models':
                    model_name = parts[2]
                else:
                    model_name = os.path.basename(root)
                tokenizer_dict[model_name] = full_path


        args = argparse.Namespace(
            devid='0',
            temperature=1.0,
            top_p=1.0,
            repeat_penalty=1.0,
            repeat_last_n=32,
            max_new_tokens=1024,
            generation_mode="greedy",
            prompt_mode="prompted",
            enable_history=True,
            lib_path=''
        )

        mm = list(tokenizer_dict.keys())
        nn = list(tokenizer_dict.values())

        for model_name in self.models_list:
            id = match_model(model_name, mm)
            if id is None:
                print(f"Model {model_name} does not match any available model.")
                continue
            tokenizer_path = nn[id]
            args.model_path = f"llm_bmodels/{model_name}"
            args.tokenizer_path = tokenizer_path

            # if 'chat' in sys.modules:
            #     del sys.modules['chat']

            # sys.path.insert(0, f"models/{mm[id]}/python_demo")

            module_name = f"llm_models.{mm[id]}.python_demo.pipeline"
            module = importlib.import_module(module_name)

            model_class = getattr(module, mm[id])

            self.models[f"{model_name}"] = model_class(args)

            # sys.path.remove(f"models/{mm[id]}/python_demo")

        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.models, self.models_list

router = AppInitializationRouter(app_name=app_name)

class ChatRequest(BaseModel):
    model: str = Field("minicpm3-4b_int4_seq512_1dev.bmodel", description="bmodel file name")
    messages: list = Field([{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"hello"}], description="Chat history")
    stream: bool = Field(False, description="Stream response")

@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest, ):
    best_match = get_close_matches(request.model, router.models_list, n=1, cutoff=0.0)
    if best_match:
        slm = router.models[best_match[0]]
    else:
        slm = router.models['minicpm3-4b_int4_seq512_1dev.bmodel']
    
    if isinstance(request.messages[-1]['content'], list):
        content = request.messages[-1]['content']
        for x in content:
            if x['type'] == 'text':
                slm.input_str = x['text']
            elif x['type'] == 'image_url':
                url = x['image_url']['url']
                png = url.split('.')[-1]
                os.system(f"wget {url} -O /data/tmpdir/image.{png}")
                slm.image_str = f"/data/tmpdir/image.{png}"
            elif x['type'] == 'image_path':
                slm.image_str = x['image_path']['path'] if isinstance(x['image_path'], dict) else x['image_path']
            else:
                slm.image_str = ''

    else:
        slm.input_str = request.messages[-1]['content']
        slm.history = request.messages
        tokens = slm.tokenizer.apply_chat_template(slm.history, tokenize=True, add_generation_prompt=True)

    if "minicpmv2" in request.model.lower():
        try:
            image_path = slm.image_str
        except:
            slm.image_str = ''

        if slm.image_str:
            if not os.path.exists(slm.image_str):
                print("Can't find image: {}".format(slm.image_str))

        slm.encode()
        token = slm.model.forward_first(slm.input_ids, slm.pixel_values, slm.image_offset)
        EOS = [slm.ID_EOS, slm.ID_IM_END]
    else:
        token =  slm.model.forward_first(tokens)
        EOS = slm.EOS if isinstance(slm.EOS, list) else [slm.EOS]

    if request.stream:
        def generate_responses(token):
            yield '{"choices": [{"delta": {"role": "assistant", "content": "'
            output_tokens = []
            while True:
                output_tokens.append(token)
                if token in EOS or slm.model.token_length >= slm.model.SEQLEN:
                    break
                word = slm.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(output_tokens) == 1:
                        pre_word = word
                        word = slm.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
                    yield word
                    output_tokens = []
                token = slm.model.forward_next()
            yield '"}}]}'
        return StreamingResponse(generate_responses(token), media_type="application/json")
    
    else:
        output_tokens = [token]
        while True:
            token = slm.model.forward_next()
            if token in EOS or slm.model.token_length >= slm.model.SEQLEN:
                break
            output_tokens += [token]
        slm.answer_cur = slm.tokenizer.decode(output_tokens)
        slm.history = []
        return JSONResponse({"choices": [{"delta": {"role": "assistant", "content": slm.answer_cur}}]})
    
### 常规测试
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


### 图片测试
# curl -X 'POST' \
#   'http://localhost:8000/llm_tpu/v1/chat/completions' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "model": "minicpmv2",
#   "messages": [
#     {
#       "role": "system",
#       "content": "You are a helpful assistant."
#     },
#     {
#       "role": "user",
#       "content": [{"type":"text","text":"what is it?"},{"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}]
#     }
#   ],
#   "stream": true
# }'