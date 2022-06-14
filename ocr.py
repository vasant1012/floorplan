import pytesseract
import time
import pathlib

PATH = pathlib.Path(__file__).parent

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

class ocr:
    def __init__(self):
        pass

    def img_to_str(image):
        custom_config = r'--oem 3 --psm 6'
        time.sleep(1)
        extracted_text = pytesseract.image_to_string(image, config=custom_config)
        extracted_text = extracted_text.replace(" |","").replace("| ","").replace("\"","\'")
        list_extracted = extracted_text.split("\n")
        return list_extracted

    def convert(value):
        lst=[i.split(' ', 1)[0] for i in value[1]]
        for i in range(len(lst)):
            if (lst[i]=='S'):
                lst[i]=str(5)
            elif (lst[i]=='T'):
                lst[i]=str(7)
            else:
                pass
        lst = "".join(lst)
        value[1] = lst
        return value

    def summary(lst):
        if len(lst) <= 3:
            lst = ocr.convert(lst)
            time.sleep(1)
            room_type = lst[0]
            room_type = room_type.replace('_','').replace('—','').replace("'","").replace(" ","")
            room_size = lst[1]
            room_size = room_size.replace('(','').replace(')','').replace("'","")
            room_size = room_size.split("X")
            room_sz_list_L, room_sz_list_B = room_size[0], room_size[1]
            room_sz_list_L = room_sz_list_L.split("-")
            room_sz_list_B = room_sz_list_B.split("-")
            total_ft_L = float(room_sz_list_L[0]) + (float(room_sz_list_L[1])/12)
            total_ft_B = float(room_sz_list_B[0]) + (float(room_sz_list_B[1])/12)
            dimensionL = round(total_ft_L,2)
            dimensionB = round(total_ft_B,2)
            rmtype = room_type.title()
            area = round((total_ft_L * total_ft_B),2)
            return dimensionL, dimensionB, rmtype, area
        else:
            time.sleep(1)
            room_type = lst[0]
            room_type = room_type.replace('_','').replace('—','').replace("'","").replace(" ","")
            room_size = lst[2]
            room_size = room_size.replace('(','').replace(')','').replace(' ','')
            room_sz_list = room_size.split("x")
            room_sz_list_L, room_sz_list_B = room_sz_list[0], room_sz_list[1]
            room_sz_list_L = room_sz_list_L.split("'")
            room_sz_list_B = room_sz_list_B.split("'")
            room_sz_list_L_ft, room_sz_list_L_in = float(room_sz_list_L[0]), float(room_sz_list_L[1])
            room_sz_list_B_ft, room_sz_list_B_in = float(room_sz_list_B[0]), float(room_sz_list_B[1])
            total_ft_L = room_sz_list_L_ft + room_sz_list_L_in/12
            total_ft_B = room_sz_list_B_ft + room_sz_list_B_in/12
            dimensionL = round(total_ft_L,2)
            dimensionB = round(total_ft_B,2)
            rmtype = room_type.title()
            area = round((total_ft_L * total_ft_B),2)
            return dimensionL, dimensionB, rmtype, area