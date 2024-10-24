from pydantic import BaseModel, Field
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
class UpscaleRequest(BaseModel):
    ## 有意义的兼容参数
    image: str = Field(..., description="Base64 encoded image to upscale")

    ## 无意义的兼容参数
    model: Optional[str] = Field("roop_face", description="(形式参数无意义)")
    n : Optional[int] = Field(1, description="(形式参数无意义)")
    response_format: Optional[str] = Field("b64_json", description="(形式参数无意义)")
    size: Optional[str] = Field("1024x1024", description="(形式参数无意义)")
    user: Optional[str] = Field("", description="(形式参数无意义)")

    ## 专有参数
    upscale_ratio: float = Field(1.0, description="Upscale ratio")

@router.post("/v1/images/variations")
@change_dir(router.dir)
async def face_swap(request: UpscaleRequest):
    src_image_bytes = BytesIO(base64.b64decode(request.image))
    src_image = Image.open(src_image_bytes)
    pil_res = router.models.extract_and_enhance_tiles(src_image, upscale_ratio=request.upscale_ratio)
    # pil to base64
    buffer = BytesIO()
    pil_res.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {
                "created": 1589478378, 
                "data": [
                    {
                        "b64_json": ret_img_b64
                    }
                        ]
                }
    return content