import os

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from .Segmentator import Segmentator
from .SAMSegmentator import SAMSegmentator

from .helper_functions import preview_methods

def segmentatimg_test():
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(os.sep.join(current_dir.split(os.sep)[:4]), "images", "jet_images", "jet1.jpg")
    preview_methods(image_path)
    
    downloads_path = os.path.join(os.sep.join(current_dir.split(os.sep)[:3]), "Downloads")
    ckpt_paht = [f for f in os.listdir(downloads_path) if "sam" == f[:3] and ".pth" == f[-4:]][0]
    ckpt_paht = os.path.join(downloads_path, ckpt_paht)
    sam_config = sam_model_registry["vit_b"](checkpoint=ckpt_paht).to("cpu")
    sam_with_prompt = SamPredictor(sam_config)
    sam_auto = SamAutomaticMaskGenerator(sam_config)
    SAM_segmentator = SAMSegmentator(sam_with_prompt, "cpu")

    sgmt = Segmentator(image_folder=os.path.split(image_path)[0], 
                        method="graph", template_threshold=0.1, SAMSegmentator=SAM_segmentator)
    sgmt()

if __name__ == "__main__":
    segmentatimg_test()