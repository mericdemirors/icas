from segmentator import Segmentating
import cv2
import numpy as np

temp1 = cv2.imread("/home/mericdemirors/Pictures/temp1.png")
segment1 = cv2.imread("/home/mericdemirors/Pictures/seg1.png")
attention1 = cv2.imread("/home/mericdemirors/Pictures/att1.png")
mask1 = cv2.imread("/home/mericdemirors/Pictures/pp1.png")

temp2 = cv2.imread("/home/mericdemirors/Pictures/temp2.png")
segment2 = cv2.imread("/home/mericdemirors/Pictures/seg2.png")
attention2 = cv2.imread("/home/mericdemirors/Pictures/att2.png")
mask2 = cv2.imread("/home/mericdemirors/Pictures/pp2.png")


sgmt = Segmentating(image_folder="/home/mericdemirors/Pictures/titles", method="kmeans",
                    templates=[temp1, temp2], template_threshold=0.1, segments=[segment1, segment2])#, attentions=[attention1, attention2], masks=[mask1, mask2])
sgmt()