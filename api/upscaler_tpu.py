from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from repo.upscaler_tpu.pipeline import UpscaleModel

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

class UpscaleRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image to upscale")
    upscale_ratio: float = Field(..., description="Upscale ratio")

@router.post("/upscale")
@change_dir(router.dir)
async def face_swap(request: UpscaleRequest):
    src_image_bytes = BytesIO(base64.b64decode(request.image))
    src_image = Image.open(src_image_bytes)
    pil_res = router.models.extract_and_enhance_tiles(src_image, upscale_ratio=request.upscale_ratio)
    # pil to base64
    buffer = BytesIO()
    pil_res.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {'ret_img': [ret_img_b64], 'message': 'success'}
    return JSONResponse(content=jsonable_encoder(content), media_type="application/json")