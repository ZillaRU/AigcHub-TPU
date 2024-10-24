from pydantic import BaseModel, Field
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
class Img2TxtRequest(BaseModel):
    ## 有意义的兼容参数
    image: str = Field(..., description="Base64 encoded source image")
    ## 无意义的兼容参数
    model: Optional[str] = Field("roop_face", description="(形式参数无意义)")
    n : Optional[int] = Field(1, description="(形式参数无意义)")
    response_format: Optional[str] = Field("b64_json", description="(形式参数无意义)")
    size: Optional[str] = Field("1024x1024", description="(形式参数无意义)")
    user: Optional[str] = Field("", description="(形式参数无意义)")

    ## 专有参数
    num_of_description: int = Field(1, description="Number of description")

@router.post("/v1/images/variations")
@change_dir(router.dir)
async def get_img_caption(request: Img2TxtRequest):
    image_bytes = BytesIO(base64.b64decode(request.image))
    Image.open(image_bytes).save("temp.jpg")
    captions, tags = router.models['pipeline']("temp.jpg", num_return_sequences=request.num_of_description)
    content = {
                "created": 1589478378, "captions": captions, "tags": tags,
                "data": []
                }
    return content