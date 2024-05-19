from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from Segmentator import Segmentating
from SAMSegmentator import SAMSegmentator

sam_config = sam_model_registry["vit_b"](checkpoint="/home/mericdemirors/Downloads/sam_vit_b_01ec64.pth").to("cpu")
sam_with_prompt = SamPredictor(sam_config)
#sam_auto = SamAutomaticMaskGenerator(sam_config)
SAM_segmentator = SAMSegmentator(sam_with_prompt, "cpu")

sgmt = Segmentating(image_folder="/home/mericdemirors/Pictures/titles", 
                    method="grabcut", template_threshold=0.1,
                    templates_path="/home/mericdemirors/Pictures/templates", attentions_path="/home/mericdemirors/Pictures/attentions",
                    segments_path="/home/mericdemirors/Pictures/segments", masks_path="/home/mericdemirors/Pictures/masks",
                    SAMSegmentator = SAM_segmentator)
sgmt()




