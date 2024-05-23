import cv2
import numpy as np
# from glob import glob
# from PIL import Image,ImageOps
# from skimage.color import rgb2gray
# import os 
# from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
# from skimage import io
import imutils
# import matplotlib.pyplot as plt
from unet_feature_extraction_model import feature_extraction_model_withdropout
import pandas as pd
import math

class main():
    def __init__(self) -> None:
        self.cuda = torch.cuda.is_available() #check cuda
        weight_path_2class='./weights/March_8_base_model_t391_v0.015544576237076207_vfocal_0.5441184573703342_vdice_0.008580684661865234.pth.tar'
        self.net_2class = feature_extraction_model_withdropout(n_channels=3,output_ch=2,start_n=8)
        if self.cuda:
            #put model on gpu
            self.net_2class = self.net_2class.cuda()
            self.net_2class.load_state_dict(torch.load(weight_path_2class)['model_state_dict'])
        else:
            self.net_2class.load_state_dict(torch.load(weight_path_2class,map_location=torch.device('cpu'))['model_state_dict'])
        print('model loaded!!....')
        self.net_2class.eval()


    def rescale(self,thresh_size=512,img=None):
        h,w=img.shape
        if h>thresh_size or w>thresh_size:
            print(h,w)
            if h>=w:
                img1=imutils.resize(img, height=thresh_size)
            else:
                img1=imutils.resize(img, width=thresh_size)
            # h,w,c=img1.shape
        else: 
            img1=img
        return img1
            


    def erode_twice(self,image):
        kernel = np.ones((5, 5), np.uint8) 
        img_erosion = cv2.erode(image, kernel, iterations=2)
        return img_erosion 
    
    def dilation_twice(self,image):
        kernel = np.ones((5, 5), np.uint8) 
        img_dilation = cv2.dilate(image, kernel, iterations=2)
        return img_dilation 
    

    def pad_input(self,input_img,patch_size=[32,32]):
        h = input_img.shape[0]
        w = input_img.shape[1]
        hMod = patch_size[0]-(h % patch_size[0])
        wMod = patch_size[1]-(w % patch_size[1])
        print('hMod,wMod : ',hMod,wMod)
        if(hMod!=patch_size[0] or wMod!=patch_size[1]):
            output = np.pad(input_img, ((0, hMod),(0,wMod),(0,0)), 'constant')
        else:
            output,hMod,wMod=input_img,0,0
        print(output.shape)
        return(output,hMod,wMod)

    def dataloader(self,image):
        image=255-image
        image=self.rescale(img=image)
        image= np.expand_dims(image, axis=2)
        image=np.concatenate([image,image,image],axis=2)
        
        image,hMod,wMod = self.pad_input(image,[32,32])
        
        
        image = np.transpose(image,(2,0,1))
        image=image[None]
        image = torch.FloatTensor(image)
        return image/255.,hMod,wMod
    
    def draw_contour(self,image, c, i):
        # compute the center of the contour area and draw a circle
        # representing the center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the countour number on the image
        cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 0, 0), 2)
        # return the image with the contour number drawn on it
        return image
    def draw_center(self,image,c):
        # compute the center of the contour area and draw a circle
        # representing the center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the countour number on the image
        image = cv2.circle(image, (cX,cY), radius=1, color=(0, 0, 255), thickness=2)
        cv2.putText(image, "#{}".format('center'), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 0, 0), 2)
        # return the image with the contour number drawn on it
        return image,cX,cY
    def find_center(self,cnt,bin_image_copy):
        #cnt = cv2.convexHull(cnt)
        cnt = np.vstack(cnt)
        image,cX,cY=self.draw_center(bin_image_copy, cnt)
        cv2.imwrite('Output_Images/find_center_test.png',image)
        return cX,cY
    def rotate_point(self, point, angle=0):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """

        pi=22/7
        #degree = #float(input("Input degrees: "))
        radian = angle*(pi/180)
        print('radian : ',radian)


        print('point : ',point)
        ox, oy = self.img_origin
        px, py = int(point[0]),int(point[1])

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy
    
    def rotate_point_v2(self, x, y, angle=45, x_shift=0, y_shift=0, units="DEGREES"):
        """
        Rotates a point in the xy-plane counterclockwise through an angle about the origin
        https://en.wikipedia.org/wiki/Rotation_matrix
        :param x: x coordinate
        :param y: y coordinate
        :param x_shift: x-axis shift from origin (0, 0)
        :param y_shift: y-axis shift from origin (0, 0)
        :param angle: The rotation angle in degrees
        :param units: DEGREES (default) or RADIANS
        :return: Tuple of rotated x and y
        """

        # Shift to origin (0,0)
        x = x - x_shift
        y = y - y_shift

        # Convert degrees to radians
        if units == "DEGREES":
            angle = math.radians(angle)

        # Rotation matrix multiplication to get rotated x & y
        xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
        yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

        return xr, yr

    def rotate_point_v3(self,x,y, angle=45):
        """Only rotate a point around the origin (0, 0)."""

        pi=22/7
        #degree = #float(input("Input degrees: "))
        radians = angle*(pi/180)
        print('radians : ',radians)



        # x, y = xy
        x,y=int(x),int(y)
        # xx = x * math.cos(radians) + y * math.sin(radians)
        # yy = -x * math.sin(radians) + y * math.cos(radians)

        
        # return xx, yy

        c, s = np.cos(radians), np.sin(radians)
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y])

        return float(m.T[0]), float(m.T[1])

    
    def find_center_point_text(self,dataFrame):
        temp=[]
        dataFrame['center_x']=dataFrame['x1']+(abs(dataFrame['x1']-dataFrame['x2'])/2)
        dataFrame['center_y']=dataFrame['y1']+(abs(dataFrame['y1']-dataFrame['y2'])/2)

        for index,data in dataFrame.iterrows():
            updated_x,updated_y=self.rotate_point((data['center_x'],data['center_y'])) #v1
            # updated_x,updated_y=self.rotate_point(data['center_x'],data['center_y'])
            data['old_x'] ,data['old_y'] =data['center_x'],data['center_y']
            data['center_x'],data['center_y']=updated_x,updated_y
            print('data : ',data)
            temp.append(data)


        rotated_dataFrame=pd.DataFrame(temp)
        # rotated_dataFrame.to_csv('rotated_dataFrame.csv',index=False)
        # print(dataFrame.head(2))
        # dataFrame.to_csv('input.csv',index=False)
        return rotated_dataFrame
    def fill_conture(self,image,cnt,class_id,class_name):
        # compute the center of the contour area and draw a circle
        # representing the center
        #M = cv2.moments(c)
        #cX = int(M["m10"] / M["m00"])
        #cY = int(M["m01"] / M["m00"])
        # draw the countour number on the image
        
        cv2.fillPoly(image, pts =[cnt], color=(class_id,class_id,class_id))
        #cv2.putText(image, "#{}".format(class_name), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
        #    1.0, (255, 0, 0), 2)
        # return the image with the contour number drawn on it
        return image
    
    def map_class(self,image,cnts,dataFrame):
        index_conture_mapping_dict={}
        index_area_mapping_dict={}


        

        for (i, cnt) in enumerate(cnts):
            i+=100
            for index,data in dataFrame.iterrows():
                dist = cv2.pointPolygonTest(cnt,(data['center_x'],data['center_y']),True)
                if dist>0:
                    #positive ie point lies into polygon
                    
                    if i not in index_conture_mapping_dict.keys():
                        image=self.fill_conture(image,cnt,i,data['Features'])#data['id']
                        
                        index_conture_mapping_dict[i]=[{'Features':data['Features'],'Dimension':data['Dimension'],'Area (Sq. Ft.)':data['Area (Sq. Ft.)']}]
                        
                    else:
                        temp=list(index_conture_mapping_dict[i])
                        new_feature={'Features':data['Features'],'Dimension':data['Dimension'],'Area (Sq. Ft.)':data['Area (Sq. Ft.)']}
                        temp.append(new_feature)
                        index_conture_mapping_dict[i]=temp
                elif dist<0:
                    # negative ie point lies outside polygon
                    pass
                else:
                    # point lies on the polygon edge
                    pass
        
        unique, counts = np.unique(image, return_counts=True)
        # print('class image np.asarray((unique, counts)).T : \n',np.asarray((unique, counts)).T)
        for class_index,pixel_count in np.asarray((unique, counts)).T:
            if class_index!=0:
                if class_index not in index_area_mapping_dict.keys():
                    index_area_mapping_dict[class_index]=pixel_count
        # from class image removinng white color pixels
        image[image==255]=0
        cv2.imwrite('Output_Images/test_class.png',image)
        print('index_conture_mapping_dict : \n',index_conture_mapping_dict)
        print('index_area_mapping_dict : \n',index_area_mapping_dict)
        

        #{0: ['FOYER', 'LIVING DINING'], 1: ['M. BEDROOM'], 2: ['BEDROOM'], 3: ['BALCONY'], 
        # 4: ['KITCHEN'], 5: ['M. BATH'], 6: ['BATH'], 7: ['PASSAGE'], 8: ['BALCONY'], 9: ['PUJA']}

        return index_conture_mapping_dict,index_area_mapping_dict,image

    def find_split_precentages(self,unique_class_list,index_area_mapping_dict,Quadrant_image_list=[]):
        quadrant_wise_dict={}
        print('unique_class_list : ',unique_class_list)
        for index,quadrant_image in enumerate(Quadrant_image_list):
            index=index+1
            unique, counts = np.unique(quadrant_image, return_counts=True)
            # print('np.asarray((unique, counts)).T : \n',np.asarray((unique, counts)).T)
            for class_index,pixel_count in np.asarray((unique, counts)).T:
                # print('class_index,pixel_count : ',class_index,pixel_count)
                
                if class_index in unique_class_list and  class_index!=0:
                    if index not in quadrant_wise_dict.keys():
                        print('(pixel_count)*100,index_area_mapping_dict[class_index] : ',(pixel_count)*100,index_area_mapping_dict[class_index])
                        percrnt_value=(pixel_count)*100/index_area_mapping_dict[class_index]
                        if percrnt_value>0:
                            quadrant_wise_dict[index]=[{class_index:{'pixel_count':pixel_count,'Percent %':'%.2f'%percrnt_value}}]
                    else:
                        temp=list(quadrant_wise_dict[index])
                        percrnt_value=(pixel_count)*100/index_area_mapping_dict[class_index]
                        if percrnt_value>0:
                            temp.append({class_index:{'pixel_count':pixel_count,'Percent %':'%.2f'%percrnt_value}})
                            quadrant_wise_dict[index]=temp
                    #print('quadrant_wise_dict : ',quadrant_wise_dict)

            #break
        print('-------------------Quadrant Wise Values--------------------\n')
        print('quadrant_wise_dict : ',quadrant_wise_dict)
        return quadrant_wise_dict

    def rotate_img(self,image,org_image,angle):
        print('image.shape',image.shape)
        rotated_image = imutils.rotate_bound(image, angle)#imutils.rotate(image, -angle)#cv2.rotate(image,angle)#imutils.rotate_bound(image, angle)#
        rotated_org_image = imutils.rotate_bound(org_image, angle)
        cv2.imwrite('Output_Images/rotated_image.png', rotated_image)
        cv2.imwrite('Output_Images/rotated_org_image.png', rotated_org_image)
        
        return rotated_image,rotated_org_image

    def split_image(self,image=None,org_image=None):
         
        h,w=image.shape[0],image.shape[1]
        q1=image[0:int(h/2),int(w/2):]
        q2=image[0:int(h/2),:int(w/2)]
        q3=image[int(h/2):,:int(w/2)]
        q4=image[int(h/2):,int(w/2):]

        org_q1=org_image[0:int(h/2),int(w/2):]
        org_q2=org_image[0:int(h/2),:int(w/2)]
        org_q3=org_image[int(h/2):,:int(w/2)]
        org_q4=org_image[int(h/2):,int(w/2):]

        

        # cv2.imwrite('Output_Images/Quadrant/q1.png',q1)
        # cv2.imwrite('Output_Images/Quadrant/q2.png',q2)
        # cv2.imwrite('Output_Images/Quadrant/q3.png',q3)
        # cv2.imwrite('Output_Images/Quadrant/q4.png',q4)

        
        # cv2.imwrite('Output_Images/Quadrant/org_q1.png',org_q1)
        # cv2.imwrite('Output_Images/Quadrant/org_q2.png',org_q2)
        # cv2.imwrite('Output_Images/Quadrant/org_q3.png',org_q3)
        # cv2.imwrite('Output_Images/Quadrant/org_q4.png',org_q4)
        
        return [q1,q2,q3,q4],[org_q1,org_q2,org_q3,org_q4]

    '''
    index_conture_mapping_dict=
     {100: [{'Features': 'FOYER', 'Dimension': '7\'3" X 5\'7"', 'Area (Sq. Ft.)': '40'}, '{\'Features\': \'LIVING DINING\', \'Dimension\': \'11\\\'11" X 17\\\'0"\', \'Area (Sq. Ft.)\': \'202\'}'],
       101: [{'Features': 'M. BEDROOM', 'Dimension': '10\'11" X 16\'3"', 'Area (Sq. Ft.)': '177'}], 
       102: [{'Features': 'BEDROOM', 'Dimension': '10\'0" X 11\'11"', 'Area (Sq. Ft.)': '119'}], 
       103: [{'Features': 'BALCONY', 'Dimension': '14\'0" X 5\'10"', 'Area (Sq. Ft.)': '81'}], 
       104: [{'Features': 'KITCHEN', 'Dimension': '7\'3" X 9\'10"', 'Area (Sq. Ft.)': '71'}], 
       105: [{'Features': 'M. BATH', 'Dimension': '8\'0" X 6\'0"', 'Area (Sq. Ft.)': '48'}], 
       106: [{'Features': 'BATH', 'Dimension': '7\'10" X 6\'0"', 'Area (Sq. Ft.)': '47'}], 
       107: [{'Features': 'PASSAGE', 'Dimension': '4\'0" WIDE', 'Area (Sq. Ft.)': 'Manual'}], 
       108: [{'Features': 'BALCONY', 'Dimension': '4\'11" X 5\'10"', 'Area (Sq. Ft.)': '28'}], 
       109: [{'Features': 'PUJA', 'Dimension': '4\'2" X 4\'0"', 'Area (Sq. Ft.)': '16'}]}

-------------------Quadrant Wise Values--------------------

quadrant_wise_dict :  {
    1: [{101: {'pixel_count': 746237, 'Percent %': '23.63'}}, 
    {105: {'pixel_count': 785349, 'Percent %': '100.00'}}, 
    {106: {'pixel_count': 766226, 'Percent %': '100.00'}}, 
    {107: {'pixel_count': 352968, 'Percent %': '51.58'}}, 
    {108: {'pixel_count': 6080, 'Percent %': '1.47'}}], 
    2: [{100: {'pixel_count': 1017293, 'Percent %': '22.76'}}, 
    {104: {'pixel_count': 1046552, 'Percent %': '100.00'}}, 
    {107: {'pixel_count': 328792, 'Percent %': '48.05'}}, 
    {108: {'pixel_count': 406506, 'Percent %': '98.53'}}, 
    {109: {'pixel_count': 320987, 'Percent %': '100.00'}}, 
    {255: {'pixel_count': 1368, 'Percent %': '100.00'}}], 
    3: [{100: {'pixel_count': 3451448, 'Percent %': '77.24'}}, 
    {102: {'pixel_count': 981908, 'Percent %': '46.98'}}, 
    {103: {'pixel_count': 404710, 'Percent %': '27.67'}},
    {107: {'pixel_count': 1194, 'Percent %': '0.17'}}], 
    4: [{101: {'pixel_count': 2412415, 'Percent %': '76.37'}}, 
    {102: {'pixel_count': 1108261, 'Percent %': '53.02'}}, 
    {103: {'pixel_count': 1058164, 'Percent %': '72.33'}},
    {107: {'pixel_count': 1364, 'Percent %': '0.20'}}]}


    '''
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    def create_final_dataFrame(self,index_conture_mapping_dict,quadrant_wise_dict):
        data_frame = pd.DataFrame(columns=['index','Quadrant','Feature','Dimension','Total Area(Sq.Ft.)','Area(Sq.Ft.)_in_Quadrant','Area(Sq.Ft.)_in_Quadrant %'])
        final_dict=[]
        for quadrant,data_list in quadrant_wise_dict.items():
            
            for data_dict in data_list:
                temp={}
                # data_dict = {101: {'pixel_count': 746237, 'Percent %': '23.63'}
                print('data_dict : ',data_dict)
                index=list(data_dict.keys())[0]
                percent=data_dict[index]['Percent %']
                if self.isfloat(percent) and int(float(percent))==0:
                  continue  
                if index in index_conture_mapping_dict:
                    feature_info =index_conture_mapping_dict[index]
                    # temp['index'] = index
                    temp['Quadrant'] = quadrant
                    
                    if len(feature_info) == 1:
                        feature_info=feature_info[0] #{'Features': 'M. BEDROOM', 'Dimension': '10\'11" X 16\'3"', 'Area (Sq. Ft.)': '177'}            
                        temp['Feature'] = feature_info['Features']
                    else:
                        feature_temp=''
                        for feature_dict in feature_info:
                            print('feature_dict : ',feature_dict)
                            feature_temp+=(feature_dict['Features']+'_')
                            print('feature_temp : ',feature_temp)
                        temp['Feature'] =feature_temp[:-1] 
                        feature_info=feature_info[0]
                    # temp['Dimension'] = feature_info['Dimension']
                    temp['Total Area(Sq.Ft.)'] = feature_info['Area (Sq. Ft.)']
                    try:
                        temp['Area(Sq.Ft.)_in_Quadrant'] = int(float(feature_info['Area (Sq. Ft.)'])*float(percent)/100)
                    except Exception as e:
                        temp['Area(Sq.Ft.)_in_Quadrant'] = 'Manual'#int( float(feature_info['Area (Sq. Ft.)'])*float(percent))

                    temp['Area(Sq.Ft.)_in_Quadrant %'] = int(float(percent))

                    print('temp : ',temp)
                    
                    final_dict.append(temp)
        print('final_dict : ',final_dict)
        final_df=pd.DataFrame(final_dict)
        final_df.to_csv('final_output.csv',index=False)
        return final_df


    def draw_liners_text(self,image):
        h,w =image.shape[0],image.shape[1]
        print('(0,h),(h,w) : ',(0,h/2),(h/2,w))
        image=cv2.line(image, (0,int(h/2)),(w,int(h/2)) , (255,0,0), thickness=2)
        image=cv2.line(image, (int(w/2),0),(int(w/2),h) , (255,0,0), thickness=2)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,0,0)
        thickness              = 2
        lineType               = 2
    
        #q_dict={'Q1':(h-50,50),'Q2':(50,50),'Q3':(50,w-50),'Q4':(h-50,w-50)}
        q_dict={'Q1':(int(w*.9),int(h*.1)),'Q2':(int(w*.1),int(h*.1)),'Q3':(int(w*.1),int(h*.9)),'Q4':(int(w*.9),int(h*.9))}

        


        for q,point in q_dict.items():
            cv2.putText(image,q, 
            point, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)


        
        # cv2.imwrite('draw_liners.png',image) 
        return image

    def Post_Processing(self,org_image,bin_image,dataFrame,roation_angle):
        print('input bin_image',bin_image.shape)
        bin_image_copy=bin_image.copy()
        bin_image=bin_image.astype(np.uint8)
        # bin_image=self.rotate_img(bin_image,0)
        bin_image=cv2.cvtColor(bin_image,cv2.COLOR_BGR2GRAY )

        bin_image=self.erode_twice(bin_image)

        print('bin_image.shape',bin_image.shape)

        cnts = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] 
        
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x),reverse=True)[:]
        dataFrame=self.find_center_point_text(dataFrame)
        index_conture_mapping_dict,index_area_mapping_dict,class_image=self.map_class(bin_image,cntsSorted,dataFrame)

        cX,cY=self.find_center(cntsSorted.copy(),bin_image_copy)
        unique_class_list = np.unique(class_image)
        rotated_class_image,rotated_org_image=self.rotate_img(class_image,org_image,roation_angle)
        
        
        # print('class image np.asarray((unique, counts)).T : \n',np.asarray((unique, counts)).T)

        Quadrant_wise_image_list,Org_Quadrant_wise_image_list= self.split_image(rotated_class_image,rotated_org_image)
        quadrant_wise_info_dict=self.find_split_precentages(unique_class_list,index_area_mapping_dict,Quadrant_wise_image_list)
        final_df=self.create_final_dataFrame(index_conture_mapping_dict,quadrant_wise_info_dict)

        # for (i, c) in enumerate(cntsSorted):
        #     # print('i,c : ',i,c)
        #     bin_image_area_wise = self.draw_contour(bin_image_copy, c, i)

        # cv2.imwrite('bin_image_area_wise.png',bin_image_area_wise)
        # bin_image_area_wise=self.dilation_twice(bin_image_area_wise)

        return final_df,(cX,cY),Org_Quadrant_wise_image_list,rotated_org_image,rotated_class_image

    def Inferance(self,RGB_image,dataFrame,rotation_angle):
        
        org_image=RGB_image.copy()
        image=cv2.cvtColor(RGB_image,cv2.COLOR_BGR2GRAY)
        org_h,org_w=image.shape[0],image.shape[1]
        self.img_origin=int(org_h/2),int(org_w/2)
        processed_image,hMod,wMod=self.dataloader(image)
        print('processed_image.shape : ',processed_image.shape)
        if self.cuda:
            processed_image=processed_image.cuda()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output_2class= self.net_2class(processed_image)
                #output_2class= net_2class(processed_image)
                
                output_2class = F.softmax(output_2class,dim=1)
                if self.cuda:
                    output_2class = output_2class.detach().cpu().numpy() 
                else:
                    output_2class = output_2class.detach().numpy() 
                    
                print('output_2class.shape : ',output_2class.shape)
                


        pred_2class = np.array(output_2class)
        
        
        pred_2class = np.transpose(pred_2class,(0,2,3,1))[0]
        
        pred_2class = np.argmax(pred_2class,2).astype(np.uint8)
        
        print('pred_2class.shape : ',pred_2class.shape)
        
        pred_2class=pred_2class[:-hMod,:-wMod]
        print('pred_2class.shape : ',pred_2class.shape)
        
        final_rgb = np.empty((pred_2class.shape[0],pred_2class.shape[1],3)) 
        

        final_rgb[pred_2class==0] = [0,0,0] # Bg Black
        final_rgb[pred_2class==1] = [255,255,255] # Object White
        
        print('final_rgb.shape',final_rgb.shape) 
        final_rgb_mask_resized=cv2.resize(final_rgb,(org_w,org_h), 0, 0, interpolation = cv2.INTER_NEAREST)
        print('Resize Done.....',final_rgb_mask_resized.shape)
        final_df,center_point,Org_Quadrant_wise_image_list,rotated_org_image,rotated_class_image=self.Post_Processing(org_image,final_rgb_mask_resized,dataFrame,rotation_angle)
        rotated_org_image=self.draw_liners_text(rotated_org_image)

        return final_df,center_point,Org_Quadrant_wise_image_list,rotated_org_image,rotated_class_image
    
    def main(self,image,input_df,rotation_angle):

        final_df,center_point,Org_Quadrant_wise_image_list,rotated_org_image,rotated_class_image=self.Inferance(image,input_df,rotation_angle)
        
        return {
            'final_df':final_df,
            'center_point':center_point,
            'Org_Quadrant_wise_image_list':Org_Quadrant_wise_image_list,
            'rotated_org_image':rotated_org_image,
            'rotated_class_image':rotated_class_image
        }



# main_obj=main()
# data=glob('D:/BizMetric/Project/Godrej/Unet/results/scripts/data/ocr/*P2*')

# for csv_file_path in data:
#     image_path=csv_file_path.replace('_png.csv','.png').replace('ocr','images')
#     image_name=image_path.split('\\')[-1]
#     file_name=csv_file_path.split('\\')[-1]
#     dataFrame=pd.read_csv(csv_file_path)
#     print(image_path,file_name,file_name)
#     image=cv2.imread(image_path)
#     output=main_obj.main(image,dataFrame,rotation_angle=0)

#     cv2.imwrite('Output_Images/'+image_name,output['rotated_org_image'])
#     cv2.imwrite('Output_Images/rotated_class_image/'+image_name,output['rotated_class_image'])
    

#     cv2.imwrite('Output_Images/Quadrant/Q1_'+image_name,output['Org_Quadrant_wise_image_list'][0])
#     cv2.imwrite('Output_Images/Quadrant/Q2_'+image_name,output['Org_Quadrant_wise_image_list'][1])
#     cv2.imwrite('Output_Images/Quadrant/Q3_'+image_name,output['Org_Quadrant_wise_image_list'][2])
#     cv2.imwrite('Output_Images/Quadrant/Q4_'+image_name,output['Org_Quadrant_wise_image_list'][3])
    
#     dataFrame=output['final_df']
#     dataFrame.to_csv('Output_Images/csvs/'+file_name,index=False)
#     break


# from Image_Processing_Pipeline import main
# main_obj=main()
# output=main_obj.main(image,dataFrame,rotation_angle=45)
