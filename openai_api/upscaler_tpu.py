from fastapi import File, Form, UploadFile
import base64
from io import BytesIO
from PIL import Image
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from repo.upscaler_tpu.pipeline import UpscaleModel
from typing import Optional

app_name = "upscaler_tpu"

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        self.models = UpscaleModel(model='./resrgan4x.bmodel', padding=20)
        return {"message": f"Application {self.app_name} has been initialized successfully."}
    
    async def destroy_app(self):
        del self.models
    

router = AppInitializationRouter(app_name=app_name)


### 图像超分；兼容openai api，image/variations
@router.post("/v1/images/variations")
@change_dir(router.dir)
async def face_swap(
    image: UploadFile = File(...),
    upscale_ratio: Optional[float] = Form(1.0),
):
    ori_image_bytes = await image.read()
    src_image = Image.open(BytesIO(ori_image_bytes))
    pil_res = router.models.extract_and_enhance_tiles(src_image, upscale_ratio=upscale_ratio)
    # pil to base64
    buffer = BytesIO()
    pil_res.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {
                "data": [
                    {
                        "b64_json": ret_img_b64
                    }
                        ]
                }
    return JSONResponse(content=jsonable_encoder(content), media_type="application/json")