import base64
import io
import os
from typing import Literal

from pydantic import BaseModel
from PIL import Image,ImageEnhance,ImageFilter

ImageTypes = Literal["jpg", "jpeg", "png", "gif"]


class ImageData(BaseModel):
    

    data: str
    type: ImageTypes

    @classmethod
    def load_from_file(cls, file_path: str) -> "ImageData":
        img = Image.open(file_path)
        # Convert to grayscale
        img = img.convert('L')
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        # Reduce noise
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # Save and return as ImageData
        output = io.BytesIO()
        img.save(output, format='PNG')
        data = base64.b64encode(output.getvalue()).decode('utf-8')
        return ImageData(data=data, type='png')

    def merge(self, other: "ImageData") -> "ImageData":
        img1 = Image.open(io.BytesIO(base64.b64decode(self.data)))
        img2 = Image.open(io.BytesIO(base64.b64decode(other.data)))
        dst = Image.new("RGB", (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        output = io.BytesIO()
        dst.save(output, format=self.type)
        return ImageData(data=base64.b64encode(output.getvalue()).decode("utf-8"), type=self.type)

    def convert(self, type: ImageTypes) -> "ImageData":
        img = Image.open(io.BytesIO(base64.b64decode(self.data)))
        output = io.BytesIO()
        img.save(output, format=type)
        return ImageData(data=base64.b64encode(output.getvalue()).decode("utf-8"), type=type)
