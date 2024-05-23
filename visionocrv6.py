import io
# from google.cloud import vision
import os
import re
import json
from google.cloud import vision_v1
from google.cloud.vision_v1 import AnnotateImageResponse
import string
import pandas as pd

# credentials = service_account.Credentials.from_service_account_file('floorplankey.json')
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "floorplankey.json"
# client = vision.ImageAnnotatorClient()


class vision_api:

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ibizmetricapicred.json"

    global keywords

    keywords = [
        "balcony",
        "bedroom",
        "m.bedroom",
        "m.",
        "m.bed",
        "m.toilet",
        "bed",
        "bath",
        'drawing',
        'dress',
        'service',
        "bathroom",
        "utility",
        "kitchen",
        "foyer",
        "toilet",
        "corridor",
        "puja",
        "pooja",
        "terrace",
        "dressing",
        "living/dining",
        "living",
        "dining",
        "passage",
        "master",
        "store",
        "lobby",
        "wash",
        "area",
        "parking",
        "corridor",
        "drybalcony",
        "wc",
        'servant',
    ]

    def __init__(self):
        pass

    def api(path):
        """Detects text in the file."""
        try:
            client = vision_v1.ImageAnnotatorClient()

            with io.open(path, 'rb') as image_file:
                content = image_file.read()

            image = vision_v1.Image(content=content)

            response = client.text_detection(image=image)
            texts = response.text_annotations

            serialized_proto_plus = AnnotateImageResponse.serialize(response)
            response = AnnotateImageResponse.deserialize(serialized_proto_plus)
            #     print(response.full_text_annotation.text)

            # serialize / deserialize json
            response_json = AnnotateImageResponse.to_json(response)
            response = json.loads(response_json)
        except:
            response = '{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'.format(
                response.error.message)

        return response

    def temp(data):

        temp = []

        for i in range(len(data['fullTextAnnotation']['pages'][0]['blocks'])):
            block = []
            cords = []
            for j in range(
                    len(data['fullTextAnnotation']['pages'][0]['blocks'][i]
                        ['paragraphs'])):
                for k in range(
                        len(data['fullTextAnnotation']['pages'][0]['blocks'][i]
                            ['paragraphs'][j]['words'])):
                    word = ' '
                    for l in range(
                            len(data['fullTextAnnotation']['pages'][0]
                                ['blocks'][i]['paragraphs'][j]['words'][k]
                                ['symbols'])):
                        #                 print(i, j, k, l)
                        block.append(data['fullTextAnnotation']['pages'][0]
                                     ['blocks'][i]['paragraphs'][j]['words'][k]
                                     ['symbols'][l]['text'])
                    block.append(word)
            my_lst_str = ''.join(map(str, block))
            x1 = data['fullTextAnnotation']['pages'][0]['blocks'][i]['boundingBox']['vertices'][0]['x']
            y1 = data['fullTextAnnotation']['pages'][0]['blocks'][i]['boundingBox']['vertices'][0]['y']
            x2 = data['fullTextAnnotation']['pages'][0]['blocks'][i]['boundingBox']['vertices'][2]['x']
            y2 = data['fullTextAnnotation']['pages'][0]['blocks'][i]['boundingBox']['vertices'][2]['y']
        #     print(my_lst_str, cords)
            temp.append([my_lst_str, [x1, y1, x2, y2]])


        for i in range(len(temp)):
            if i < (len(temp) - 1):
                text = temp[i][0].split(" ")
        #         print(text)
                if len(text) >= 2 and text[0].isdigit():
                    temp[i - 1][0] = temp[i - 1][0].strip() + ' ' + text[0]
                    text.pop(0)
                    my_lst_str = ' '.join(map(str, text))
                    temp[i][0] = my_lst_str

        new = []
        for j in range(len(temp)):
            text = temp[j][0].lower()
            ind = []
            for i in range(len(keywords)):
                index = 0
                while index < len(text):
                    index = text.find(keywords[i], index)
                    if index == -1:
                        break
                    ind.append(index)
                    index += 2  # +2 because len('ll') == 2

            ind = [*set(ind)]
            ind.sort()
            for i in range(len(ind)):
                if i < (len(ind) - 1):
                    if max(ind) < 8:
                        #                 print('0', temp[j][ind[i]:])
                        new.append([temp[j][0][ind[i]:], temp[j][1]])
                        break
                    else:
                        #                 print('1', temp[j][ind[i]:(ind[i+1]-1)])
                        new.append([temp[j][0][ind[i]:(ind[i + 1] - 1)], temp[j][1]])
                else:
                    #             print('2', temp[j][ind[i]:])
                    new.append([temp[j][0][ind[i]:], temp[j][1]])

        temp_new = []

        for i in range(len(new)):
            if ('''"''' in new[i][0] or """'""" in new[i][0] or "x" in new[i][0]
                    or "X" in new[i][0] or "M" in new[i][0] or "mm" in new[i][0] or "."
                    in new[i][0] or "0" in new[i][0]) and any(chr.isdigit()
                                                              for chr in new[i][0]):
                temp_new.append([new[i][:]])

        temp_new = [i for i in temp_new if i != '']

        temp_new = [[i[0][0].replace(' L', ''), i[0][1]] for i in temp_new]

        temp = temp_new.copy()

        temp_new = []

        if '(' in temp[-1][0][0]:
            for i in range(len(temp)):
                if len(re.findall(r"\d{4} X \d{4}", temp[i][0][0])) > 0:
                    temp_new.append([temp[i][0][:]])
        else:
            temp_new = temp
        temp = temp_new.copy()
        return temp

    def feature(temp):

        # feature building

        ALPHA = string.ascii_letters
        final = []

        for i in temp:
            if " " in i[0]:
        #         print(i[0])
                text = i[0].replace(".", " ").replace("-", " -").split(' ')
                if text[0].startswith(tuple(ALPHA)) or text[1].startswith(
                        tuple(ALPHA)):
                    if text[0].lower() in keywords or text[1].lower(
                    ) in keywords or text[2].lower() in keywords:
                        final.append(i)
                    if text[0].lower() + ' ' + text[1].lower() in keywords:
                        final.append(i)

        # print(final)

        my_dict = {"Features":[], "Dimension":[], "x1":[], "y1":[], "x2":[], "y2":[]}

        for i in final:
            text = i[0].replace('BEDROOM-', 'BEDROOM -').split(' ')
            for j in range(len(text)):
                #         print(text[j])
                if j < (len(text) - 1):
                    if (text[j] + text[j + 1]).lower() in keywords:
                        text[j] = text[j] + ' ' + text[j + 1]
                        #                 print('1', text[j])
                        text.pop(j + 1)
                        my_dict["Features"].append(text[j].replace("BED ROOM ", "BEDROOM"))
                        my_dict["Dimension"].append(text[j + 1:])
                        my_dict["x1"].append(i[1][0])
                        my_dict["y1"].append(i[1][1])
                        my_dict["x2"].append(i[1][2])
                        my_dict["y2"].append(i[1][3])
                        continue
                    if text[j].lower() in keywords:
                        my_dict["Features"].append(text[j].replace("BED ROOM ", "BEDROOM"))
                        my_dict["Dimension"].append(text[j + 1:])
                        my_dict["x1"].append(i[1][0])
                        my_dict["y1"].append(i[1][1])
                        my_dict["x2"].append(i[1][2])
                        my_dict["y2"].append(i[1][3])
                        
        df = pd.DataFrame(my_dict, columns=my_dict.keys())

        for i in range(len(df['Dimension'])):
            if len(df['Dimension'][i]) == 0:
                df.drop(i, inplace=True)
                df.reset_index(drop=True)

        df = df.reset_index(drop=True)

        for i in range(len(df['Features'])):
            for j in range(len(df['Dimension'])):
                #         print(i , j)
                if i < (len(df['Features'])) and j < (len(
                        df['Dimension'])) and i == j:
                    text = df['Dimension'][j]
                    if len(text) >= 2:
                        if (df['Features'][i]
                                == 'BEDROOM') and (text[0] == '1' or text[0]
                                                   == '2' or text[0] == '3'):
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[0]
                            df['Dimension'][j] = text[1:]
                            continue
                        if (df['Features'][i]
                                == 'TOILET') and (text[0] == '1' or text[0]
                                                  == '2' or text[0] == '3'):
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[0]
                            df['Dimension'][j] = text[1:]
                            continue
                        if (df['Features'][i]
                                == 'BED ROOM') and (text[0] == '1' or text[0]
                                                    == '2' or text[0] == '3'):
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[0]
                            df['Dimension'][j] = text[1:]
                            continue
                        if (df['Features'][i]
                                == 'BEDROOM') and (text[1] == '1' or text[1]
                                                   == '2' or text[1] == '3'):
                            #                 print(text)
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i]
                                == 'BED ROOM') and (text[1] == '1' or text[1]
                                                    == '2' or text[1] == '3'):
                            #                 print(text)
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i]
                                == 'BED ROOM') and (text[1] == '01' or text[1]
                                                    == '02' or text[1] == '03'):
                            #                 print(text)
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[0] + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i]
                                == 'MASTER BED ROOM') and (text[1] == '01' or text[1]
                                                    == '02' or text[1] == '03'):
                            #                 print(text)
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[0] + text[1]
                            df['Dimension'][j] = text[2:]  
                        if (df['Features'][i]
                                == 'BEDROOM') and (text[1] == '01' or text[1]
                                                   == '02' or text[1] == '03'):
                            #                     print(text)
                            df['Features'][i] = df['Features'][i] + \
                                text[0] + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i]
                                == 'TOILET') and (text[1] == '01' or text[1]
                                                  == '02' or text[1] == '03'):
                            #                     print(text)
                            df['Features'][i] = df['Features'][i] + \
                                text[0] + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i] == 'TOILET') and (
                                text[1] == '1' or text[1] == '2'
                                or text[1] == '3'):
                            #                     print(text)
                            df['Features'][i] = df['Features'][i] + \
                                text[0] + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i]
                                == 'BATHROOM') and (text[1] == '01'
                                                    or text[1] == '02'
                                                    or text[1] == '03'):
                            #                     print(text)
                            df['Features'][i] = df['Features'][i] + \
                                text[0] + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i]
                                == 'BATH ROOM') and (text[1] == '1' or text[1]
                                                     == '2' or text[1] == '3'):
                            #                     print(text)
                            df['Features'][i] = df['Features'][i] + \
                                text[0] + text[1]
                            df['Dimension'][j] = text[2:]
                        if (df['Features'][i]
                                == 'BATHROOM') and (text[0] == '01'
                                                    or text[0] == '02'
                                                    or text[0] == '03'):
                            #                     print(text)
                            df['Features'][i] = df['Features'][i] + text[0]
                            df['Dimension'][j] = text[1:]

        for i in range(len(df['Features'])):
            for j in range(len(df['Dimension'])):
                #         print(i , j)
                if i < (len(df['Features']) -
                        1) and j < (len(df['Dimension']) - 1) and i == j:
                    #             print(df['Features'][i+1] , df['Dimension'][j])
                    text = df['Dimension'][j]
                    #             print(text)
                    if len(text) >= 2:
                        if (text[0] + " " + text[1]) == df['Features'][i + 1]:
                            #                     print("Two words in area are matching")
                            df['Features'][i] = df['Features'][i] + \
                                ' ' + text[0] + ' ' + text[1]
                            df['Dimension'][j] = df['Dimension'][j + 1]
                            df.drop(i + 1)
                            continue
                        if text[0] == df['Features'][i + 1]:
                            #                     print("First word of area list")
                            df['Features'][
                                i] = df['Features'][i] + ' ' + text[0]
                            df['Dimension'][j] = df['Dimension'][j + 1]
                            df.drop(i + 1)
                            continue
                        if text[1] == df['Features'][i + 1]:
                            if (text[1] + " " + text[2]) == df['Features'][i +
                                                                           1]:
                                #                         print("Second word and Third word of area is matching")
                                df['Features'][i] = df['Features'][i] + ' ' + \
                                    text[0] + ' ' + text[1] + ' ' + text[2]
                                df['Dimension'][j] = df['Dimension'][j + 1]
                                df.drop(i + 1)
                                continue
                            elif (text[1] == df['Features'][i + 1]) and (
                                    text[0].isalnum()):
                                #                         print("Elif")
                                df['Features'][i] = df['Features'][i] + \
                                    ' ' + text[0] + ' ' + text[1]
                                df['Dimension'][j] = [text[0]]

                            else:
                                #                         print("Second word of area is matching")
                                df['Features'][i] = df['Features'][i] + \
                                    ' ' + text[0] + ' ' + text[1]
                                df['Dimension'][j] = df['Dimension'][j + 1]
                                df.drop(i + 1)
                        if text[0] != df['Features'][i + 1]:
                            #                     print(text[0], "text is not same as feature")
                            if text[0].isalpha() and text[0] in keywords:
                                #                                 print("if first word is alpha")
                                df['Features'][i] = df['Features'][i] + \
                                    ' ' + text[0]
                                df['Dimension'][j] = df['Dimension'][j + 1]
                                continue
                            else:
                                #                         print('in last else', df['Features'][i], df['Dimension'][j])
                                df['Features'][i] = df['Features'][i]
                                df['Dimension'][j] = df['Dimension'][j]

        for i in range(len(df['Features'])):
            for j in range(len(df['Dimension'])):
                #         print(i , j)
                if i < (len(df['Features']) -
                        1) and j < (len(df['Dimension']) - 1) and i == j:
                    if df['Dimension'][j][1:] == df['Dimension'][j + 1]:
                        if len(df['Dimension'][j]) == 0:
                            continue
                        else:
                            #                             print(i, j, df['Dimension'][j])
                            df['Features'][i] = df['Features'][i] + \
                                ' ' + df['Dimension'][j][0]
                            df['Dimension'][j] = df['Dimension'][j + 1]
                            df.drop(i + 1)
                            df = df.reset_index(drop=True)

        for i in range(len(df['Features'])):
            for j in range(len(df['Dimension'])):
                #         print(i , j)
                if i < (len(df['Features']) -
                        1) and j < (len(df['Dimension']) - 1) and i == j:
                    if " " in df['Features'][i]:
                        text = df['Features'][i].split(' ')
                        if (text[-1] == df['Features'][i + 1]) and (
                                df['Dimension'][j] == df['Dimension'][j + 1]):
                            #                     print(j+1, df['Dimension'][j+1])
                            df.drop(j + 1, inplace=True)
                            df = df.reset_index(drop=True)
                        elif (df['Features'][i] == df['Features'][i + 1]) and (
                                df['Dimension'][j] == df['Dimension'][j + 1]):
                            #                     print(j+1, df['Dimension'][j+1])
                            df.drop(j + 1, inplace=True)
                            df = df.reset_index(drop=True)

        scrap_word = [
            'X', 'XS', 'x', 'WIDE', 'DO', 'M', 'a', 'PASSAGE', 'UTILITY', ')1',
            'I', 'mm', 'D', 'BED3350', 'O', 'BATHROOM', 'SERVICE', 'SLAB'
        ]
        for i in range(len(df['Dimension'])):
            text = df['Dimension'][i]
            for j in text:
                if j.isalpha() and j not in scrap_word:
                    #             print(text.index(j), j)
                    num = df['Dimension'][i].index(j)
                    #             print(i[:(num-1)])
                    df['Dimension'][i] = df['Dimension'][i][:(num - 1)]

        for i in range(len(df['Dimension'])):
            df['Dimension'][i] = " ".join(df['Dimension'][i][::])

        for i in range(len(df['Features'])):
            for j in range(len(df['Dimension'])):
                if i < (len(df['Features']) -
                        1) and j < (len(df['Dimension']) - 1) and i == j:
                    if df['Dimension'][j] == df['Dimension'][
                            j + 1] and df['Features'][i] == df['Features'][i +
                                                                           1]:
                        df['Features'][i] = df['Features'][i] + ' ' + '1'
                        df['Features'][i +
                                       1] = df['Features'][i + 1] + ' ' + '2'

        for i in range(len(df['Features'])):
            for j in range(len(df['Dimension'])):
                if i < (len(df['Features']) -
                        1) and j < (len(df['Dimension']) - 1) and i == j:
                    if df['Dimension'][j] == df['Dimension'][
                            j + 1] and df['Features'][i] == df['Features'][i +
                                                                           1]:
                        df.drop_duplicates(subset=['Dimension'],
                                           keep='first',
                                           inplace=True)
                        df = df.reset_index(drop=True)

        df = df.reset_index(drop=True)

        #         area preprocessing
        for i in range(len(df['Dimension'])):
            text = df['Dimension'][i]
            text = text.replace("-", "").replace("⁰", '''"''').replace(
                "¹", """'""").replace("x", "X").replace("Oo", "").replace(
                    "++", "").replace("Ⓒ", "").replace("° *", " X ").replace(
                        "E1825",
                        "1825").replace(" : ", ".").replace(' 1+', '').replace(
                            "☆", "").replace('ROOM', '').replace(" °", " X ")
            df['Dimension'][i] = text

        df = df[df['Dimension'] != '']

        df = df.reset_index(drop=True)

        for i in range(len(df['Dimension'])):
            text = df['Dimension'][i].replace(' ', '')
            if "WIDE" in df['Dimension'][i]:
                text = df['Dimension'][i].split()
                text = text[:(text.index("WIDE") + 1)]
                text = " ".join(text[::])
                df['Dimension'][i] = text
        #         dimension.append(text)

        for i in range(len(df['Dimension'])):
            text = df['Dimension'][i]
            if "'" not in text and 'WIDE' not in text and 'X' not in text:
                #         print(text)
                df['Dimension'][i] = text.replace(' ', 'X').replace(
                    "XmmX/", " WIDE")

        df['Dimension'] = df['Dimension'].replace(
            '100126X', '100 X 126').replace('100X110XX00XDO', '100 X 110')
        
        feature_df = df.copy()

        if "(" in feature_df.iat[-1, 1]:
            for i in range(len(feature_df['Dimension'])):
                if len(re.findall(r"\d{4} X \d{4}",
                                  feature_df['Dimension'][i])) > 0:
                    #                     print(re.findall(r"\d{4} X \d{4}", feature_df['Dimension'][i])[0])
                    feature_df['Dimension'][i] = re.findall(
                        r"\d{4} X \d{4}", feature_df['Dimension'][i])[0]

        # feature_df.drop_duplicates(subset=['Dimension'], keep='first', inplace=True)
        # feature_df = feature_df.reset_index(drop=True)

        scrap_dim = [
            '''54 " X3 '  11 " ''', 'OX', '/X', '176X', "4'58 ", "4'11X ",
            "ZX 101 '", 'PLATFORM', '15K4.45X', 'SLABX', 'X1X', 'X3X', 'X2X', '62 % X 146 ', 'TOILET'
        ]

        for i in range(len(feature_df['Dimension'])):
            if feature_df['Dimension'][i] in scrap_dim:
                #                 print("Yes", feature_df['Dimension'][i])
                feature_df.drop(i, inplace=True)
#                 feature_df.reset_index(drop=True)

        feature_df = feature_df.sort_values(by=['Features'])
        feature_df = feature_df.reset_index(drop=True)
    
        return feature_df

    def mm_to_ft(length, width):
        feet1 = length / 304.8
        feet2 = width / 304.8
        length = round(feet1, 2)
        width = round(feet2, 2)
        length = str(length).split('.')[0] + """'""" + str(
            int(float(str(length).split('.')[1]) / 100 * 12)) + '''"'''
        width = str(width).split('.')[0] + """'""" + str(
            int(float(str(width).split('.')[1]) / 100 * 12)) + '''"'''
        dimension = length + " X " + width
        return dimension

    def meter_to_ft(length, width):
        feet1 = length / 0.3048
        feet2 = width / 0.3048
        length = round(feet1, 2)
        width = round(feet2, 2)
        length = str(length).split('.')[0] + """'""" + str(
            int(float(str(length).split('.')[1]) / 100 * 12)) + '''"'''
        width = str(width).split('.')[0] + """'""" + str(
            int(float(str(width).split('.')[1]) / 100 * 12)) + '''"'''
        dimension = length + " X " + width
        return dimension

    def dimension(df):

        # area calculation
        if len(re.findall(r"\d{4} X \d{4}", df['Dimension'][int(len(df)/2)])) > 0 or len(
                re.findall(r"\d{4}X\d{4}", df['Dimension'][int(len(df)/2)])) > 0:
            print('MM')
            for i in range(len(df['Dimension'])):
                text = df['Dimension'][i]
                if 'WIDE' in text:
                    index = text.find('WIDE')
                    fttext = text[:index - 1]
                    if """'""" not in fttext and '''"''' in fttext:
                        inch = fttext.find('''"''')
                        if int(fttext[:inch][:2]) > 20:
                            fttext = fttext.replace(' ', '')
                            text = f'''{fttext[0]}'{fttext[1:]} WIDE'''
                            #                             print('wide w/o feet symbol', text)
                            df['Dimension'][i] = text
                        elif '(' in fttext:
                            #                             print('7')
                            text = re.findall('''\d{4} ''', fttext)[0]
                            length = int(text)
                            length = length / 304.8
                            length = round(length, 2)
                            length = str(length).split('.')[0] + """'""" + str(
                                int(
                                    float(str(length).split('.')[1]) / 100 *
                                    12)) + '''"'''
                            text = str(length) + ' WIDE'
                            #                 print('dim 2', text)
                            df['Dimension'][i] = text
                        else:
                            #                             print('3')
                            #                     print('else wide w/o feet symbol', text)
                            df['Dimension'][i] = text

                    elif """'""" in fttext and '''"''' in fttext:
                        if '/' in fttext:
                            #                             print('4')
                            #                             print(fttext)
                            text = re.findall(
                                '''\d{1,2}'\d{1,2} "''', fttext)[0].replace(
                                    " ", "") + ' WIDE'
                            df['Dimension'][i] = text
                        else:
                            #                             print('10')
#                             print(fttext)
                            if len(re.findall('''\d{1,2} ' \d{1,2} "''', fttext))>0: 
                                text = re.findall(
                                    '''\d{1,2} ' \d{1,2} "''', fttext)[0].replace(
                                        " ", "") + ' WIDE'
                                df['Dimension'][i] = text

                    elif len(fttext) < 4:
                        #                         print('5')
                        #                         print(fttext)
                        text = re.findall('''\d{2,3}''', fttext)[0]
                        if int(text[0:2]) > 20:
                            text = f'''{text[0:1]}'{text[1:3]}" ''' + 'WIDE'
                            df['Dimension'][i] = text
                        else:
                            text = f'''{text[0:2]}'{text[2:4]}" ''' + 'WIDE'
                            df['Dimension'][i] = text
                        #                         print(text)

                    else:
                        #                         print('8')
                        text = re.findall('\d{4}', fttext[:index + 4])[0]
                        length = int(re.findall('\d{4}', text)[0])
                        length = length / 304.8
                        length = round(length, 2)
                        length = str(length).split('.')[0] + """'""" + str(
                            int(float(str(length).split('.')[1]) / 100 *
                                12)) + '''"'''
                        text = str(length) + ' WIDE'
                        #                 print('dim 2', text)
                        df['Dimension'][i] = text

                else:
                    if 'X' in text:
                        index = text.find('X')
                        text = text[:index] + "X" + text[index + 1:index + 6]
                        text = text.replace('  X  ', ' X ')
                        text = text[:text.find('X') + 6]
                        lst = text.replace(' ', '').replace(
                            'X30500', 'X3050').replace("'", '').replace(
                                'BED3350',
                                '').replace('43600.', '3600').replace(
                                    '22450', '2450').replace('32450',
                                                             '2450').split('X')
#                         print(lst)
                        text = vision_api.mm_to_ft(int(lst[0]), int(lst[1]))
                        #                 print('w/o W', text)
                        df['Dimension'][i] = text
                    else:
                        lst = text.replace(' ', '').split('X')
                        if len(lst) > 1:
                            text = vision_api.mm_to_ft(int(lst[0]),
                                                       int(lst[1]))
                            #                     print('orig MM text', text)
                            df['Dimension'][i] = text
                        else:
                            #                     print('else single', text)
                            df['Dimension'][i] = text

        elif len(df['Dimension'][int(len(df)/2)].split()[0]) >= 4 and '.' in df[
                'Dimension'][int(len(df)/2)] and "'" not in df['Dimension'][int(len(df)/2)].split(
                )[0] and '"' not in df['Dimension'][int(len(df)/2)].split()[0]:
            print('meter')
            if 'M' in df['Dimension'][0] or 'm' in df['Dimension'][0]:
                #         print("w/o M")
                for i in range(len(df['Dimension'])):
                    text = df['Dimension'][i].replace(',', '.')
                    if 'WIDE' in text:
                        index = text.find('WIDE')
                        text = text[:index + 4]
                        #                 print('dim 1', text)
                        df['Dimension'][i] = text
                    else:
                        index = text.find('X')
                        text = text[:index] + "X" + text[index + 1:index + 8]
                        text = text.replace('MX', 'M X').replace(
                            'X0', 'X 0').replace('M 5', 'M')
                        mindex = text.find('M')
                        if '.' not in text[:mindex]:
                            text = f'{text[0]}.{text[1]}{text[mindex:]}'
                            lst = text.replace(' ', '').replace('M',
                                                                '').split('X')
                            text = vision_api.meter_to_ft(
                                float(lst[0]), float(lst[1]))
                            #                             print('mindex', text)
                            df['Dimension'][i] = text
                            continue
                        else:
                            text = text.replace('  X  ', ' X ')
                            lst = text.replace('M', '').replace(
                                'm',
                                '').replace(' ', '').replace('T', '').replace(
                                    '*', '').replace('$',
                                                     '').replace('Ⓒ',
                                                                 '').split('X')
                            #                     print(lst[0], lst[1])
                            text = vision_api.meter_to_ft(
                                float(lst[0]), float(lst[1]))
                            #                     print("w/o mindex", text)
                            df['Dimension'][i] = text
            else:
                print("with M")
                for i in range(len(df['Dimension'])):
                    text = df['Dimension'][i]
                    if 'WIDE' in text:
                        index = text.find('WIDE')
                        text = text[:index + 4]
                        #                 print('with M wide', text)
                        df['Dimension'][i] = text
                    else:
                        index = text.find('X')
                        text = text[:index] + "X" + text[index + 1:index + 6]
                        lst = text.replace(',', '.').split('X')
                        if float(lst[1][:2]) > 6:
                            text = f'{lst[0]} X {lst[1][0]}.{lst[1][1]}'
                            lst = text.replace(' ', '').split('X')
                            text = vision_api.meter_to_ft(
                                float(lst[0]), float(lst[1]))
                            #                     print('re', text)
                            df['Dimension'][i] = text
                            continue
                        else:
                            text = text.replace('X', ' X ')
                            text = text.replace(',', '.').replace(
                                '.100', '1.100').replace('.8 9', '.8').replace(
                                    'M', '').replace('4W', "")
                            lst = text.replace(' ', '').split('X')
                            text = vision_api.meter_to_ft(
                                float(lst[0]), float(lst[1]))
                            #                     print('orig else', text)
                            df['Dimension'][i] = text

        else:
            print('feet')

            for i in range(len(df['Dimension'])):
                text = df['Dimension'][i].replace(' ', '')
                if "WIDE" in text:
                    #         print(text[:(text.find("WIDE")-1)])
                    dim = text[:(text.find("WIDE") - 1)]
                    dim = dim.replace("""'""", '')
                    #                 print(dim)
                    if len(dim) <= 2:
                        if int(dim) <= 20:
                            text = dim + """'""" + " " + "WIDE"
                            #                             print("1", text)
                            df['Dimension'][i] = text
                        else:
                            text = dim[0] + """'""" + \
                                dim[1] + '"' + " " + "WIDE"
                            #                             print("2", text)
                            df['Dimension'][i] = text
                    elif len(dim) > 2:
                        dim = dim.replace('''"''', '')
                        if int(dim[0:2]) < 20:
                            text = dim[0:2] + """'""" + \
                                dim[2:] + '"' + " " + "WIDE"
                            #                             print("3", text)
                            df['Dimension'][i] = text
                        else:
                            text = dim[0:1] + """'""" + \
                                dim[1:] + '"' + " " + "WIDE"
                            #                             print("4", text)
                            df['Dimension'][i] = text
                else:
                    lst = text.split("X")
                    if len(lst) > 1:
                        lst[0] = lst[0].replace(" ", "").replace(
                            """'""",
                            "").replace('''"''', "").replace("*", "").replace(
                                ".",
                                "").replace(".", "").replace('4W', "").replace(
                                    ')1', "").replace('/', "").replace(
                                        '100126', "100 126").replace('O', '')
                        lst[1] = lst[1].replace(" ", "").replace(
                            """'""",
                            "").replace('''"''', "").replace("*", "").replace(
                                ".",
                                "").replace(".", "").replace('4W', "").replace(
                                    ')1', "").replace('/', "").replace(
                                        '100126', "100 126").replace('O', '')
                        #                         print(lst[0], lst[1])
                        if len(lst[0]) >= 3 and len(lst[1]) >= 3:
                            if int(lst[0][0:2]) > 23 and int(lst[1][0:2]) > 23:
                                a = f'''{lst[0][0]}'{lst[0][1:2]}" X {lst[1][0]}'{lst[1][1:2]}" '''
                                #                                 print('dim1', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) > 23 and int(
                                    lst[1][0:2]) < 23:
                                a = f'''{lst[0][0]}'{lst[0][1:3]}" X {lst[1][0:2]}'{lst[1][2:4]}" '''
                                #                                 print('dim2', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) < 23 and int(
                                    lst[1][0:2]) > 23:
                                a = f'''{lst[0][0:1]}'{lst[0][1:3]}" X {lst[1][0]}'{lst[1][1:2]}" '''
                                #                                 print('dim3', a)
                                df['Dimension'][i] = a
                            else:
                                a = f'''{lst[0][0:2]}'{lst[0][2:4]}" X {lst[1][0:2]}'{lst[1][2:3]}"'''
                                #                                 print('dim4', a)
                                df['Dimension'][i] = a

                        if len(lst[0]) >= 3 and len(lst[1]) <= 2:
                            if int(lst[0][0:2]) > 20 and int(lst[1][0:2]) > 20:
                                a = f'''{lst[0][0]}'{lst[0][1:]}" X {lst[1][0]}'{lst[1][1:]}" '''
                                #                                 print('dim5', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) > 20 and int(
                                    lst[1][0:2]) < 20:
                                a = f'''{lst[0][0]}'{lst[0][1:]}" X {lst[1][0:2]}' '''
                                #                                 print('dim6', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) < 20 and int(
                                    lst[1][0:2]) > 20:
                                a = f'''{lst[0][0:2]}' X {lst[1][0]}'{lst[1][1:]}" '''
                                #                                 print('dim7', a)
                                df['Dimension'][i] = a
                            else:
                                a = f'''{lst[0][0:2]}'{lst[0][2:]}" X {lst[1][0:2]}' '''
                                #                                 print('dim16', a)
                                df['Dimension'][i] = a

                        if len(lst[0]) <= 2 and len(lst[1]) >= 3:
                            if int(lst[0][0:2]) > 20 and int(lst[1][0:2]) > 20:
                                a = f'''{lst[0][0]}'{lst[0][1:]}" X {lst[1][0]}'{lst[1][1:2]}" '''
                                #                                 print('dim8', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) > 20 and int(
                                    lst[1][0:2]) < 20:
                                a = f'''{lst[0][0]}'{lst[0][1:]}" X {lst[1][0:2]}'{lst[1][2:3]}" '''
                                #                                 print('dim9', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) < 20 and int(
                                    lst[1][0:2]) > 20:
                                a = f'''{lst[0][0:2]}' X {lst[1][0]}'{lst[1][1:2]}" '''
                                #                                 print('dim10', a)
                                df['Dimension'][i] = a
                            else:
                                a = f'''{lst[0][0:2]}' X {lst[1][0:2]}'{lst[1][2:]}"'''
                                #                                 print('dim11', a)
                                df['Dimension'][i] = a

                        if len(lst[0]) <= 2 and len(lst[1]) <= 2:
                            if int(lst[0][0:2]) > 20 and int(lst[1][0:2]) > 20:
                                a = f'''{lst[0][0]}'{lst[0][1:]}" X {lst[1][0]}'{lst[1][1:]}" '''
                                #                                 print('dim12', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) > 20 and int(
                                    lst[1][0:2]) < 20:
                                a = f'''{lst[0][0]}'{lst[0][1:]}" X {lst[1][0:2]}' '''
                                #                                 print('dim13', a)
                                df['Dimension'][i] = a
                            elif int(lst[0][0:2]) < 20 and int(
                                    lst[1][0:2]) > 20:
                                a = f'''{lst[0][0:2]}' X {lst[1][0]}'{lst[1][1:]}" '''
                                #                                 print('dim14', a)
                                df['Dimension'][i] = a
                            else:
                                a = f'''{lst[0][0:2]}' X {lst[1][0:2]}' '''
                                #                                 print('dim15', a)
                                df['Dimension'][i] = a

        dimension_df = df.copy()

        manual = []

        for i in range(len(dimension_df['Dimension'])):
            match1 = re.findall('''\d{1}'\d{1}" X \d{1}'\d{1}"''',
                                dimension_df['Dimension'][i])
            match2 = re.findall('''\d{2}'\d{1}" X \d{2}'\d{2}"''',
                                dimension_df['Dimension'][i])
            match3 = re.findall('''\d{2}'\d{1}" X \d{2}'\d{1}"''',
                                dimension_df['Dimension'][i])
            match4 = re.findall('''\d{1}'\d{1}" X \d{2}'\d{1}"''',
                                dimension_df['Dimension'][i])
            match5 = re.findall('''\d{1}'\d{2}" X \d{1}'\d{1}"''',
                                dimension_df['Dimension'][i])
            match6 = re.findall('''\d{2}'\d{2}" X \d{2}'\d{1}"''',
                                dimension_df['Dimension'][i])
            match7 = re.findall('''\d{1}'\d{2}" X \d{2}'\d{1}"''',
                                dimension_df['Dimension'][i])
            match8 = re.findall('''\d{2}' X \d{1}'\d{1}"''',
                                dimension_df['Dimension'][i])
            match9 = re.findall('''\d{1,2}'\d{1}" X \d{2}' ''',
                                dimension_df['Dimension'][i])
            match10 = re.findall('''\d{1}'\d{2}" X \d{1}'\d{2}"''',
                                 dimension_df['Dimension'][i])
            match11 = re.findall('''\d{1}'\d{1}" X \d{1}'\d{2}"''',
                                 dimension_df['Dimension'][i])
            match12 = re.findall('''\d{2}'\d{1}" X \d{1}'\d{2}"''',
                                 dimension_df['Dimension'][i])
            match13 = re.findall('''\d{2}'\d{2}" X \d{2}'\d{2}"''',
                                 dimension_df['Dimension'][i])
            match14 = re.findall('''\d{1}'\d{1}" X \d{2}'\d{2}"''',
                                 dimension_df['Dimension'][i])
            match15 = re.findall('''\d{2}' X \d{2}'\d{1}"''',
                                 dimension_df['Dimension'][i])
            match21 = re.findall('''\d{1}'\d{2}" WIDE''',
                                 dimension_df['Dimension'][i])
            match22 = re.findall('''\d{1}'\d{1}" WIDE''',
                                 dimension_df['Dimension'][i])
            match23 = re.findall('''\d{1}' WIDE''',
                                 dimension_df['Dimension'][i])
            match24 = re.findall('''\d{1,2}' X \d{1,2}' ''',
                                 dimension_df['Dimension'][i])
            match25 = re.findall('''\d{1}'\d{1}" X \d{1}' ''',
                                 dimension_df['Dimension'][i])
            match26 = re.findall('''\d{1}' X \d{1}'\d{1}" ''',
                                 dimension_df['Dimension'][i])
            match27 = re.findall('''\d{1,2}'\d{1,2}" X \d{1,2}'\d{1,2}" ''',
                                 dimension_df['Dimension'][i])

            if len(match1) > 0:
                #         print("1", i, match1[0])
                manual.append("No")
            elif len(match2) > 0:
                #         print("2", i, match2[0])
                manual.append("No")
            elif len(match3) > 0:
                #         print("3", i, match3[0])
                manual.append("No")
            elif len(match4) > 0:
                #         print("4", i, match4[0])
                manual.append("No")
            elif len(match5) > 0:
                #         print("5", i, match5[0])
                manual.append("No")
            elif len(match6) > 0:
                #         print("6", i, match6[0])
                manual.append("No")
            elif len(match7) > 0:
                #         print("7", i, match7[0])
                manual.append("No")
            elif len(match8) > 0:
                #         print("8", i, match8[0])
                manual.append("No")
            elif len(match9) > 0:
                #         print("9", i, match9[0])
                manual.append("No")
            elif len(match10) > 0:
                #         print("10", i, match10[0])
                manual.append("No")
            elif len(match11) > 0:
                #         print("11", i, match11[0])
                manual.append("No")
            elif len(match12) > 0:
                #         print("12", i, match12[0])
                manual.append("No")
            elif len(match13) > 0:
                #         print("13", i, match13[0])
                manual.append("No")
            elif len(match14) > 0:
                #         print("14", i, match14[0])
                manual.append("No")
            elif len(match15) > 0:
                #         print("15", i, match15[0])
                manual.append("No")
            elif len(match21) > 0:
                #         print("21", i, match21[0])
                manual.append("No")
            elif len(match22) > 0:
                #         print("22", i, match22[0])
                manual.append("No")
            elif len(match23) > 0:
                #         print("22", i, match22[0])
                manual.append("No")
            elif len(match24) > 0:
                #         print("22", i, match22[0])
                manual.append("No")
            elif len(match25) > 0:
                #         print("22", i, match22[0])
                manual.append("No")
            elif len(match26) > 0:
                #         print("22", i, match22[0])
                manual.append("No")
            elif len(match27) > 0:
                #         print("22", i, match22[0])
                manual.append("No")

            else:
                #         print('Manual', df['Dimension'][i])
                manual.append("Yes")

        dimension_df['Manual'] = manual
              
        area = []

        for i in range(len(dimension_df['Dimension'])):
            if "WIDE" in dimension_df['Dimension'][i] or dimension_df['Manual'][
                    i] == 'Yes':
                #         print(i, dimension_df['Dimension'][i])
                area.append("Manual")
            else:
                lst = dimension_df['Dimension'][i].replace(' ', '').split('X')
                if '''"''' in lst[0]:
                    length = float(lst[0][:lst[0].find("""'""")]) + \
                        float(float(lst[0][lst[0].find("""'""")+1:lst[0].find('''"''')])/12)
                else:
                    length = float(lst[0][:lst[0].find("""'""")])
                if '''"''' in lst[1]:
                    breath = float(lst[1][:lst[1].find("""'""")]) + \
                        float(float(lst[1][lst[1].find("""'""")+1:lst[1].find('''"''')])/12)
                else:
                    breath = float(lst[1][:lst[1].find("""'""")])
                a = int(length * breath)
                #         print(i, area)
                area.append(a)
                

        dimension_df['Area (Sq. Ft.)'] = area
        
        for i in range(len(dimension_df['Dimension'])):
            if len(re.findall(r"""\d{3}""", dimension_df['Dimension'][i]))>0:
                dimension_df['Area (Sq. Ft.)'][i]='Manual'
            if len(re.findall(r'''('\d{2}")''', dimension_df['Dimension'][i]))>0:
                lst = re.findall(r'''('\d{2}")''', dimension_df['Dimension'][i])
                for j in lst:
                    if int(j.replace("""'""", '').replace('''"''', ""))>11:
                        dimension_df['Area (Sq. Ft.)'][i]='Manual'
            if len(re.findall(r'''(\d{2}')''', dimension_df['Dimension'][i]))>0:
                lst = re.findall(r'''(\d{2}')''', dimension_df['Dimension'][i])
                for j in lst:
                    if int(j.replace("""'""", '').replace('''"''', ""))>25:
                        dimension_df['Area (Sq. Ft.)'][i]='Manual'
        
        dimension_df.drop(['Manual'], axis=1, inplace=True)
        
        dimension_df.drop_duplicates(subset=['Dimension'], keep='first', inplace=True)

        dimension_df = dimension_df.reset_index(drop=True)

        return dimension_df