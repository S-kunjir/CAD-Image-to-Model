import os
import base64
import shutil
from typing import List

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageEnhance
from pydantic import BaseModel, Field

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class PartsBoundingBox(BaseModel):
    name: str = Field(..., description="name of part")
    bbox: list[int]
    confidence: float

class PartList(BaseModel):
    parts: list[PartsBoundingBox]
    image_height: int
    image_width: int

class PartImageSeparator():
    def __init__(self, image_path, output_dir):
        self.image_path = image_path
        self.output_dir = output_dir

    def run(self):
        parts_bbox = self.get_part_bbox()
        parts_bbox_fix = self.expand_bboxes(parts_bbox)
        self.create_or_empty_dir(self.output_dir)
        partlist = self.tile_output(self.output_dir, parts_bbox_fix)
        enhanced_partlist = self.image_enhance(partlist)
        return enhanced_partlist


    def create_or_empty_dir(self, dir_path: str):
        """
        Create directory if it doesn't exist.
        If it exists, remove all contents inside it.
        """
        if os.path.exists(dir_path):
            # remove all files and subdirectories
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                else:
                    shutil.rmtree(item_path)
        else:
            os.makedirs(dir_path, exist_ok=True)

    def encode_image(self):
        with open(self.image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        
    def get_part_bbox(self):
        image_base64 = self.encode_image()

        response = client.chat.completions.parse(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a computer vision system. "
                        "Detect mechanical parts in engineering drawings and "
                        "return bounding boxes in pixel coordinates."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Identify all mechanical parts (flange, shaft, bolt, nut, key, "
                                "assembly views)."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            response_format=PartList,
        )
        parsed = response.choices[0].message.parsed
        return parsed.model_dump()
        
    def expand_bboxes(self, data, margin=100):
        """
        Expands bounding boxes by a given margin while keeping them within image bounds.

        Args:
            data (dict): input dictionary with 'parts'
            image_width (int): width of image
            image_height (int): height of image
            margin (int): pixels to expand on each side

        Returns:
            dict: updated dictionary with expanded bounding boxes
        """
        image_width = data.get("image_width", 0)
        image_height = data.get("image_height", 0)
        for part in data.get("parts", []):
            xmin, ymin, xmax, ymax = part["bbox"]

            # Skip invalid / empty boxes
            if xmin == xmax == ymin == ymax == 0:
                continue

            new_xmin = max(0, xmin - margin)
            new_ymin = max(0, ymin - margin)
            new_xmax = min(image_width - 1, xmax + margin)
            new_ymax = min(image_height - 1, ymax + margin)

            part["bbox"] = [new_xmin, new_ymin, new_xmax, new_ymax]

        return data
    
    def tile_output(self, output_dir, bounding_boxes):
        img = cv2.imread(self.image_path)
        partlist = []
        if img is None:
            raise ValueError("Image could not be loaded")

        h, w, _ = img.shape
        os.makedirs(output_dir, exist_ok=True)

        # ---- CROP & SAVE ----
        for item in bounding_boxes["parts"]:
            name = item["name"]
            x1, y1, x2, y2 = item["bbox"]

            # Safety clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"Skipped {name} (empty crop)")
                continue

            output_path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(output_path, crop)
            partlist.append(output_path)
            print(f"Saved {name}: {crop.shape}")

        print("All bounding-box tiles saved successfully.")
        return partlist
    
    def image_enhance(self, partlist):
        if len(partlist)>0:
            for imgfile in partlist:
                img = Image.open(imgfile).convert("RGB")
                img_np = np.array(img)
    
                scale_percent = 7.0
                height, width = img_np.shape[:2]
                
                new_width = int(width * scale_percent)
                new_height = int(height * scale_percent)
                
                zoomed_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                zoomed_pil = Image.fromarray(zoomed_np)
                
                contrast_enhancer = ImageEnhance.Contrast(zoomed_pil)
                contrast_enhanced = contrast_enhancer.enhance(1.5)  
                
                sharpness_enhancer = ImageEnhance.Sharpness(contrast_enhanced)
                sharpness_enhanced = sharpness_enhancer.enhance(2.0)  
                
                sharpness_enhanced.save(imgfile)
        return partlist 


def segment_assembly_parts(image_path: str, output_dir: str) -> List[str]:
    part_image_separator = PartImageSeparator(image_path, output_dir)
    part_list = part_image_separator.run()
    return part_list




 
