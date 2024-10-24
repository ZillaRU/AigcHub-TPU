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
from typing import Optional

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


### 语音对话；兼容openai api，audio/translation
class ConversionRequest(BaseModel):
    ## 有意义的兼容参数
    file: str = Field(..., description="提问语音的路径")
    ## 无意义的兼容参数
    prompt: Optional[str] = Field(..., description="（形式参数无意义）") 
    response_format: Optional[str] = Field('wav', description="（形式参数无意义）")
    model: Optional[str] = Field("emotivoice", description="（形式参数无意义）")
    temperature: Optional[float] = Field(0.0, description="（形式参数无意义）")

@router.post("/v1/audio/translation")
@change_dir(router.dir)
async def gptsovits_api(request: ConversionRequest):  
    try:
        answer = fm_main(router, request.file)

        if not os.path.exists("/data/tmpdir/aigchub"):
            os.makedirs("/data/tmpdir/aigchub")
        audio_path = "/data/tmpdir/aigchub/flowmirror_output.wav"
        sf.write(audio_path, answer, 16000)
        logging.info("语音回答已生成")
        content = {"text": audio_path, 'info': 'text is the answer audio path'}
        return content

    except Exception as e:
        return {"error": str(e), "info": "处理过程中出现错误"}