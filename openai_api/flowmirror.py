from fastapi import File, UploadFile
import base64
import os
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import aiofiles
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
@router.post("/v1/audio/translation")
@change_dir(router.dir)
async def gptsovits_api(
    file: UploadFile = File(...)
):  
    try:
        file_path = f"/data/tmpdir/aigchub/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        async with aiofiles.open(file_path, "wb") as buffer:
            data = await file.read()
            await buffer.write(data)

        answer = fm_main(router, file_path)

        os.remove(file_path)
        audio_path = "/data/tmpdir/aigchub/flowmirror_output.wav"
        sf.write(audio_path, answer, 16000)
        logging.info("语音回答已生成")
        content = {"text": audio_path, 'info': 'text is the answer audio path'}
        return content

    except Exception as e:
        return {"error": str(e), "info": "处理过程中出现错误"}