import cv2
import numpy as np

class img_contour:
    def __init__(self):
        pass

    def find_contour(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        cv2.bitwise_not(thresh, thresh)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        box = max(cnts, key=cv2.contourArea)
        left = tuple(box[box[:, :, 0].argmin()][0])
        right = tuple(box[box[:, :, 0].argmax()][0])
        return box, thresh

    def find_center(img, contour, thresh):
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
#         cv2.drawContours(img, [contour], -1, (0,0,255), 6)
        mask = np.zeros(thresh.shape[:2], np.uint8)
        cv2.fillPoly(mask, pts=[np.asarray(contour)], color=(1))
        M = cv2.moments(mask, binaryImage=True)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return x, y, w, h, cx, cy

    def extend_line(p1, p2, distance):
        diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        p3_x = int(p1[0] + distance*np.cos(diff))
        p3_y = int(p1[1] + distance*np.sin(diff))
        p4_x = int(p1[0] - distance*np.cos(diff))
        p4_y = int(p1[1] - distance*np.sin(diff))
        return (p3_x, p3_y), (p4_x, p4_y)
    
    def draw_contour(img, x, y, w, h, cx, cy, angle):
    

        
        # print(angle)
        actual_angle = 360-angle
        # print( actual_angle)
        if (angle + 90) > 360:
            orth_angle = (actual_angle + 90)-360
        else:
            orth_angle = actual_angle + 90
        # print(orth_angle)
        length = 150
        distance = 10000
                         
        # cv2.putText(img, "Q1", (cx + ((w+angle)//4), y + ((h+angle)//4)), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "Q2", (x + ((w+angle)//4), y + ((h-angle)//4)), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "Q3", (x + ((w-angle)//4), cy + ((h-angle)//4)), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "Q4", (cx + ((w-angle)//4), cy + ((h+angle)//4)), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)

        # cv2.putText(img, "NE", (cx + int(length * np.cos(actual_angle * np.pi / 180.0)), y + int(length * np.sin(actual_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "NW", (x + int(length * np.cos(actual_angle * np.pi / 180.0)), y + int(length * np.sin(actual_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "SW", (x + int(length * np.cos(actual_angle * np.pi / 180.0)), cy + int(length * np.sin(actual_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "SE", (cx + int(length * np.cos(actual_angle * np.pi / 180.0)), cy + int(length * np.sin(actual_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "NE", (cx + int((w * np.cos(angle * np.pi / 180.0))//4), y + int(length * np.sin(angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)

        # print("NE", cx + int((w//4) * np.cos(angle * np.pi / 180.0)), y + int(length * np.sin(angle * np.pi / 180.0)))
        # cv2.putText(img, "NW", (x + int(length * np.cos(angle * np.pi / 180.0)), y + int(length * np.sin(angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # print("NW", x + int(length * np.cos(angle * np.pi / 180.0)), y + int(length * np.sin(angle * np.pi / 180.0)))            
        # cv2.putText(img, "SW", (x + int(length * np.cos(angle * np.pi / 180.0)), cy + int(length * np.sin(angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # print("SW", x + int(length * np.cos(angle * np.pi / 180.0)), cy + int(length * np.sin(angle * np.pi / 180.0)))            
        # cv2.putText(img, "SE", (cx + int(length * np.cos(angle * np.pi / 180.0)), cy + int(length * np.sin(angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # print("SE", cx + int(length * np.cos(angle * np.pi / 180.0)), cy + int(length * np.sin(angle * np.pi / 180.0)))            
        # cv2.putText(img, "NE", (cx + int(length * np.cos(orth_angle * np.pi / 180.0)), y + int(length * np.sin(orth_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "NW", (x + int(length * np.cos(orth_angle * np.pi / 180.0)), y + int(length * np.sin(orth_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "SW", (x + int(length * np.cos(orth_angle * np.pi / 180.0)), cy + int(length * np.sin(orth_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
        # cv2.putText(img, "SE", (cx + int(length * np.cos(orth_angle * np.pi / 180.0)), cy + int(length * np.sin(orth_angle * np.pi / 180.0))), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 0), 4)
                    
        
        x2 =  int((cx + length * np.cos(actual_angle * np.pi / 180.0)))
        y2 =  int((cy + length * np.sin(actual_angle * np.pi / 180.0)))
        x3 =  int((cx + length * np.cos(orth_angle * np.pi / 180.0)))
        y3 =  int((cy + length * np.sin(orth_angle * np.pi / 180.0)))
        # print(cx,cy)
        # print(x2,y2)
        # print(x3,y3)
        
        p1 = (cx,cy)
        p2 = (x2,y2)
        p3 = (x3,y3)
        
        points_1 = img_contour.extend_line(p1,p2,distance)
        cv2.line(img, points_1[0],points_1[1], (255,0,0), 4, cv2.LINE_AA)

        points_2 = img_contour.extend_line(p1,p3,distance)
        cv2.line(img, points_2[0],points_2[1], (255,0,0), 4, cv2.LINE_AA)        

        return img
        
        
            