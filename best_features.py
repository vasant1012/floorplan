import pandas as pd
import numpy as np
import re
import random
import warnings

warnings.filterwarnings('ignore')


class best_features:
    def __init__(self):
        pass

    def std_df(df):
        df['Features'] = df['Features'].apply(lambda x: x.upper())
        df['Features'] = [
            i.replace('BED ROOM',
                      'BEDROOM').replace('M. BEDROOM',
                                         'MASTER BEDROOM').replace('-', '')
            for i in df['Features']
        ]
        df = df.drop(columns=['Dimension'])
        
#         print(df)

        for i in range(len(df['Area (Sq. Ft.)'])):
            if df['Area (Sq. Ft.)'][i] == 'Manual':
                df['Area (Sq. Ft.)'][i] = random.randint(25, 60)


        df['Area (Sq. Ft.)'] = df['Area (Sq. Ft.)'].astype(int)

        for i in range(len(df['Features'])):
            text = df['Features'][i].strip()
            match1 = re.findall(r"BEDROOM-\d{2}", text)
            match2 = re.findall(r"BEDROOM \d{1,2}", text)
            match3 = re.findall(r"BATH ROOM \d{1,2}", text)
            match4 = re.findall(r"BATHROOM\d{2}", text)
            match5 = re.findall(r"BATHROOM \d{2}", text)
            match6 = re.findall(r"TOILET\d{1}", text)
            match7 = re.findall(r"BEDROOM\d{1,2}", text)
            match8 = re.findall(r"BATH ROOM\d{1,2}", text)
            if len(match1) > 0:
                df['Features'][i] = "BEDROOM"
            elif len(match2) > 0:
                df['Features'][i] = "BEDROOM"
            elif len(match3) > 0:
                df['Features'][i] = "BATHROOM"
            elif len(match4) > 0:
                df['Features'][i] = "BATHROOM"
            elif len(match5) > 0:
                df['Features'][i] = "BATHROOM"
            elif len(match6) > 0:
                df['Features'][i] = "BATHROOM"
            elif len(match7) > 0:
                df['Features'][i] = "BEDROOM"
            elif len(match8) > 0:
                df['Features'][i] = "BATHROOM"
            elif text == "MASTER BATH" or text == "BATH" or text == "M. BATH" or text == "WC":
                df['Features'][i] = "BATHROOM"
            elif text == "TOILET" or text == "MASTER TOILET" or text == "M. TOILET" or text == "M.TOILET":
                df['Features'][i] = "BATHROOM"
            elif text == "BED" or text == "M.BEDROOM":
                df['Features'][i] = "BEDROOM"
            elif text == "LIVING DINING":
                df['Features'][i] = "LIVING/DINING"

        df = df.sort_values(by=['Features'])
        df = df.reset_index(drop=True)
        return df

    def combine_Living_Dining(df_org):
        df = df_org.copy()
        unique_features = df.Features.unique()
        if ('LIVING/DINING' not in unique_features) and (
                'LIVING' in unique_features) and ('DINING' in unique_features):
            sum_of_area = df[(df['Features'] == 'LIVING') |
                             (df['Features'] == 'DINING')].sum().values[1]
            living_index = df.Features.to_list().index('LIVING')
            dining_index = df.Features.to_list().index('DINING')
            df.loc[int(living_index), 'Features'] = 'LIVING/DINING'
            df.loc[int(living_index), 'Area (Sq. Ft.)'] = sum_of_area
            df = df.drop(dining_index)
        elif ('LIVING' in unique_features) and ('DINING'
                                                not in unique_features):
            living_index = df.Features.to_list().index('LIVING')
            df.loc[int(living_index), 'Features'] = 'LIVING/DINING'
        elif ('LIVING' not in unique_features) and ('DINING'
                                                    in unique_features):
            dining_index = df.Features.to_list().index('DINING')
            df.loc[int(dining_index), 'Features'] = 'LIVING/DINING'
        return df

    def sort_data(df):
        updated_df = best_features.combine_Living_Dining(df)
        final_df = pd.DataFrame()
        list_unique_feature = updated_df.Features.unique()
        for feature in list_unique_feature:
            lst = []
            sub_df = updated_df.loc[updated_df['Features'] == feature]
            sub_df = sub_df.sort_values(by=['Area (Sq. Ft.)'], ascending=False)
            # print(feature,'\n-----------\n',sub_df)
            count = 1
            row, col = sub_df.shape
            #print('row,col : ',row,col)
            if row == 1:
                if feature == 'BEDROOM' and 'MASTER BEDROOM' not in list_unique_feature:
                    sub_df.iloc[0, 0] = 'MASTER BEDROOM'
                final_df = pd.concat([final_df, sub_df], axis=0)
            else:
                if feature == 'BEDROOM' and 'MASTER BEDROOM' not in list_unique_feature:
                    sub_df.iloc[0, 0] = 'MASTER BEDROOM'
                for index, data in sub_df.iterrows():
                    #print('data : ',data)
                    if data.Features == 'MASTER BEDROOM':
                        pass
                    else:
                        sub_df.loc[index, 'Features'] = data.Features + \
                            ' '+str(count)
                        count += 1
                        #print('updated data :',data)
                #print('final sub_df : ',sub_df)
                final_df = pd.concat([final_df, sub_df], axis=0)

        return final_df

    def final_summary(new_1, new_2):
        list_unique_feature_new_1 = new_1.Features.unique()
        list_unique_feature_new_2 = new_2.Features.unique()
        common = [
            value for value in list(list_unique_feature_new_1)
            if value in list(list_unique_feature_new_2)
        ]
        df3_dict = {'Features': [], 'floorplan1': [], 'floorplan2': []}


        for fet_name in common:
            A1=new_1.loc[new_1.Features==fet_name,'Area (Sq. Ft.)'].values[0]
            A2=new_2.loc[new_2.Features==fet_name,'Area (Sq. Ft.)'].values[0]
            df3_dict['Features'].append(fet_name)
            df3_dict['floorplan1'].append(A1)
            df3_dict['floorplan2'].append(A2)
        #     print(fet_name,A1,A2)

        df3 = pd.DataFrame(df3_dict, columns=df3_dict.keys())
        dif1 = np.setdiff1d(list_unique_feature_new_1,
                            list_unique_feature_new_2)
        dif2 = np.setdiff1d(list_unique_feature_new_2,
                            list_unique_feature_new_1)
        temp3 = np.concatenate((dif1, dif2))
        not_common_list = list(temp3)
        my_dict = {'Features': [], 'Area (Sq. Ft.)': [], 'Plan': []}
        for index, data in new_1.iterrows():
            if data.Features in not_common_list:
                my_dict['Features'].append(data.Features)
                my_dict['Area (Sq. Ft.)'].append(data['Area (Sq. Ft.)'])
                my_dict['Plan'].append("floorplan1")
        for index, data in new_2.iterrows():
            if data.Features in not_common_list:
                my_dict['Features'].append(data.Features)
                my_dict['Area (Sq. Ft.)'].append(data['Area (Sq. Ft.)'])
                my_dict['Plan'].append("floorplan2")

        df4 = pd.DataFrame(my_dict, columns=my_dict.keys())
        return df3, df4