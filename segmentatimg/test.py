from Segmentator import Segmentating

attentions = "/home/mericdemirors/Pictures/attentions"
masks = "/home/mericdemirors/Pictures/masks"
segments = "/home/mericdemirors/Pictures/segments"
templates = "/home/mericdemirors/Pictures/templates"

sgmt = Segmentating(image_folder="/home/mericdemirors/Pictures/araba", 
                    method="superpixel", template_threshold=0.1,
                    templates_path=templates, attentions_path=segments,
                    segments_path=attentions, masks_path=masks)
sgmt()




