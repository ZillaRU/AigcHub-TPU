from fastapi import File, Form, UploadFile
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional

app_name = "img2txt"

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        from repo.img2txt.img_speaking_pipeline import ImageSpeakingPipeline as ISPipeline
        self.models['pipeline'] = ISPipeline()
        return {"message": f"Application {self.app_name} has been initialized successfully."}
    
    async def destroy_app(self):
        del self.models['pipeline']
    
router = AppInitializationRouter(app_name=app_name)


### img2txt；兼容openai api，image/variations
@router.post("/v1/images/variations")
@change_dir(router.dir)
async def get_img_caption(
    image: UploadFile = File(...),
    num_of_description: Optional[int] = Form(1),
):
    ori_image_bytes = await image.read()
    image_bytes = BytesIO(ori_image_bytes)
    Image.open(image_bytes).save("temp.jpg")
    captions, tags = router.models['pipeline']("temp.jpg", num_return_sequences=num_of_description)
    content = {
                "created": 1589478378, "captions": captions, "tags": tags,
                "data": []
                }
    return content
