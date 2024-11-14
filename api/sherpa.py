import subprocess
import re
import json
import time
from api.base_api import BaseAPIRouter, change_dir, init_helper
from typing import Optional
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse


app_name = "sherpa"

def run_shell_command(command):
    pattern = re.compile(r'\{.*?\}')
    dict_obj = {"text":""} 
    with subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            matches = pattern.findall(line)
            for match in matches:
                try:
                    # 将匹配的字符串转换为字典
                    dict_obj = json.loads(match)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {match}")
    return dict_obj

# def run_shell_command_tofile(command, output_file):
#     with open(output_file, "w") as file:
#         try:
#             result = subprocess.run(command, shell=True, stdout=file, stderr=subprocess.STDOUT)
#             if result.returncode != 0:
#                 print(f"Command failed with return code {result.returncode}")
#         except Exception as e:
#             print(f"An error occurred: {e}")

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        self.cmd = "build/bin/sherpa-onnx --tokens=./models/tokens.txt --zipformer2-ctc-model=./models/zipformer2_ctc_F32.bmodel "
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.models

router = AppInitializationRouter(app_name=app_name)



### ASR；兼容openai api，audio/transcriptions
@router.post("/v1/audio/transcriptions")
@change_dir(router.dir)
async def sherpa(
    file: UploadFile = File(...),
    response_format: Optional[str] = Form("text"),
):

    # save the audio file
    file_tmp_path = "/data/tmpdir/sherpa.wav"
    with open(file_tmp_path, "wb") as buffer:
        buffer.write(file.file.read())
        
    audio_start_time = time.time()
    result = run_shell_command(router.cmd + file_tmp_path)
    total_time = time.time() - audio_start_time
    print(f"Total time: {total_time}")
    if  response_format== "text":
        return PlainTextResponse(content=result["text"])
    else:
        return JSONResponse(content=result)
    

# #### 测试命令
# curl http://localhost:8000/sherpa/v1/audio/transcriptions \
#   -F 'file=@/data/TEST.wav;type=audio/wav' \
#   -F 'response_format=json'
