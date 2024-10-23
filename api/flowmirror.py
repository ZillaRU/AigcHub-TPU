from pydantic import BaseModel, Field
import base64
import os
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import tempfile
import soundfile as sf
import logging
import numpy as np

app_name = "flowmirror"
logging.basicConfig(level=logging.INFO)
SEQ_LENGTH = 512
IDS_LENGTH = 196

def fm_main(router, audio_path):
    input_ids = router.hubert.get_input_ids(audio_path)
    input_ids = np.pad(input_ids, ((0,0),(IDS_LENGTH - input_ids.shape[1], 0)))

    conti = True
    while conti:
        answer, _ = router.model.generate(prompt_input_ids=input_ids, speaker_embedding=router.speaker_embedding)
        if (answer == 0.0).all(): conti = True
        else:  conti = False

    answer = answer.squeeze()
    answer = answer - np.min(answer)
    max_audio=np.max(answer)
    answer/=max_audio
    answer = (answer * 32768).astype(np.int16)
    return answer


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        from repo.flowmirror.src_sail.modeling_flow_mirror_bmodel import CNHubert, FlowmirrorForConditionalGeneration, Config
        flomirror_dir = "repo/flowmirror"
        self.hubert = CNHubert("models")
        self.model = FlowmirrorForConditionalGeneration(model_dir = "models", config=Config("configs/config.json"), device_id=0)
        self.speaker_embedding = np.load("models/speaker_embedding.npz")['speaker_embedding_1']
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.hubert, self.model, self.speaker_embedding

router = AppInitializationRouter(app_name=app_name)

class TTSRequest(BaseModel):
    q_audio_path: str = Field(..., description="提问语音的绝对路径")

@router.post("/flowmirror_path")
@change_dir(router.dir)
async def gptsovits_api(request: TTSRequest):  
    try:
        answer = fm_main(router, request.q_audio_path)

        if not os.path.exists("/data/tmpdir/aigchub"):
            os.makedirs("/data/tmpdir/aigchub")
        audio_path = "/data/tmpdir/aigchub/flowmirror_output.wav"
        sf.write(audio_path, answer, 16000)
        logging.info("语音回答已生成")
        return JSONResponse(content=jsonable_encoder(audio_path), media_type="application/json") # 为方便复制，只返回音频地址
    except Exception as e:
        return {"message": "处理过程中出现错误", "error": str(e)}

class TTSBase64Request(BaseModel):
    q_audio_base64: str = Field(..., description="提问语音的base64编码")

@router.post("/flowmirror_base64")
@change_dir(router.dir)
async def gptsovits_api_base64(request: TTSBase64Request):  
    try:
        # 解码 base64 编码的语音
        q_audio_base64 = base64.b64decode(request.q_audio_base64)
        
        # 使用临时文件处理解码后的数据
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(q_audio_base64)
            audio_path = tmp.name

        answer = fm_main(router, audio_path)

        # 手动删除文件
        os.remove(audio_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, answer, 16000)
            with open(tmp_out.name, 'rb') as file:
                audio_data = file.read()
            os.remove(tmp_out.name)
        audio_base64 = base64.b64encode(audio_data).decode()
        logging.info("base64语音回答已生成")
        # return audio_base64 # 为方便复制，只返回 base64 编码的音频
        return JSONResponse(content=jsonable_encoder(audio_base64), media_type="application/json") # 为方便复制，只返回 base64 编码的音频
    except Exception as e:
        return {"message": "处理过程中出现错误", "error": str(e)}