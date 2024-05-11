# origin of the code: https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py
import numpy as np
import cv2

class App():
    GREEN = [0,255,0]          # rectangle color
    DARK_GRAY = [40,40,40]    # PR BG
    LIGHT_GRAY = [200,200,200]# PR FG
    BLACK = [0,0,0]           # sure BG
    WHITE = [255,255,255]     # sure FG
    thickness = 3

    DRAW_BG = {'color' : BLACK, 'val' : 0} # right click
    DRAW_FG = {'color' : WHITE, 'val' : 1} # left click
    DRAW_PR_BG = {'color' : DARK_GRAY, 'val' : 2} # ctrl + right click
    DRAW_PR_FG = {'color' : LIGHT_GRAY, 'val' : 3} # ctrl + left click
    paint_dict = None

    rect = (0,0,0,0)                # rect x,y,w,h
    display_rects = []              # selected rectangles for displaying
    segment_rects = []              # selected rectangles for segmenting
    currently_drawing = False       # flag for drawing action
    currently_drawing_rect = False  # flag for rectangle action
    rect_or_mask = -1               # flag for selecting rect or mask mode

    def onmouse(self, event, x, y, flags, param):
        global paint_dict
        # Draw Rectangle
        if event == cv2.EVENT_MBUTTONDOWN:
            self.currently_drawing_rect = True
            self.ix, self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.currently_drawing_rect:
                self.img = self.altered.copy()
                for r in self.display_rects:
                    cv2.rectangle(self.img, (r[0], r[1]), (r[2], r[3]), self.GREEN, 1)
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.GREEN, 1)
        elif event == cv2.EVENT_MBUTTONUP:
            self.currently_drawing_rect = False
            for r in self.display_rects:
                cv2.rectangle(self.altered, (r[0], r[1]), (r[2], r[3]), self.GREEN, 1)
            cv2.rectangle(self.altered, (self.ix, self.iy), (x, y), self.GREEN, 1)
            self.display_rects.append((self.ix, self.iy, x, y))
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.segment_rects.append(self.rect)
            self.rect_or_mask = 0
            self.altered = self.img.copy()
        if self.currently_drawing_rect:
            return

        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.currently_drawing = True
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                paint_dict = self.DRAW_PR_FG
            else:
                paint_dict = self.DRAW_FG

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.currently_drawing = True
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                paint_dict = self.DRAW_PR_BG
            else:
                paint_dict = self.DRAW_BG

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.currently_drawing = False

        if self.currently_drawing:
            cv2.circle(self.altered, (x,y), self.thickness, paint_dict['color'], -1)
            self.img = self.altered.copy()
            cv2.circle(self.mask, (x,y), self.thickness, paint_dict['val'], -1)

    def on_trackbar_change(self, value):
            self.thickness = value
   
    def run(self, filename):
        self.img = cv2.imread(cv2.samples.findFile(filename))
        self.original = self.img.copy()                             # original copy
        self.altered = self.img.copy()                              # copy to store annotations
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8)  # background initialized mask
        self.display = np.zeros(self.img.shape, np.uint8)           # display image

        # annotations and display windows
        cv2.namedWindow('display')
        cv2.namedWindow('annotations', flags= cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback('annotations', self.onmouse)
        cv2.createTrackbar('brush size','annotations',self.thickness,100, self.on_trackbar_change)
        cv2.moveWindow('annotations', self.img.shape[1]+10,90)
        

        while True:
            cv2.imshow('display', self.display)
            cv2.imshow('annotations', self.img)
            key = cv2.waitKey(1)

            # key bindings
            if key == ord('q'):
                # throw a custom exception to end program termination
                6/0
            if key == ord('f'):
                # return segmented image
                6/0
                break
            elif key == ord('r'): # reset everything
                print("resetting \n")
                self.rect = (0,0,1,1)
                self.display_rects = []
                self.segment_rects = []
                self.currently_drawing = False
                self.currently_drawing_rect = False
                self.rect_or_mask = -1
                self.img = self.original.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
                self.display = np.zeros(self.img.shape, np.uint8)           # display image to be shown
            elif key == ord(' '): # segment the image
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                if (self.rect_or_mask == 0):         # grabcut with rect
                    merged_mask = self.mask.copy()
                    for r in self.segment_rects:
                        temp_mask = np.zeros_like(merged_mask)
                        cv2.grabCut(self.original, temp_mask, r, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

                        merged_mask[(temp_mask == 2) & ((merged_mask == 2) | (merged_mask == 0))] = 2
                        merged_mask[(temp_mask == 3) & ((merged_mask == 2) | (merged_mask == 3) | (merged_mask == 0))] = 3
                    self.mask = merged_mask.copy()

                    self.rect_or_mask = 1
                elif (self.rect_or_mask == 1):       # grabcut with mask
                    cv2.grabCut(self.original, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.display = cv2.bitwise_and(self.original, self.original, mask=mask2)

if __name__ == '__main__':
    print(__doc__)
    App().run("/home/mericdemirors/Pictures/araba/araba.jpg")
    cv2.destroyAllWindows()
