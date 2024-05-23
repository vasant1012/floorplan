import pathlib
from Image_Processing_Pipeline import main
from visionocrv6 import vision_api
import base64
import warnings
from main import app
from best_features import best_features
import pandas as pd
import plotly.express as px
from dash import Input, Output, html, dcc, State
import cv2
import os
import dash_bootstrap_components as dbc
import numpy as np
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
main_obj = main()

warnings.filterwarnings('ignore')

PATH = pathlib.Path(__file__).parent

layout = dbc.Container([
    html.Br(),
    dcc.Upload(id='upload-image_p2',
               children=html.Div(
                   [html.A('Select or Upload the Second Floorplan')]),
               style={
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   "height": "60px"
               },
               multiple=True),
    html.Br(),
    html.Div(id='output-image-upload_p2'),
])


def parse_contents(contents, filename):
    # contents = str(contents[0])
    encoded_data = contents.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    fig = px.imshow(img)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    output = html.Div([dcc.Graph(figure=fig)])
    cv2.imwrite(filename, img)
    data = vision_api.api(filename)
    temp = vision_api.temp(data)
    temp_df = pd.DataFrame(temp, columns=['API Data', 'cords'])
    temp_df.drop(columns=['cords'], inplace=True)
    temp_df.to_csv("data_img_2.csv", index=False)
    table = dbc.Table.from_dataframe(temp_df, bordered=True)
    try:
        feature_df = vision_api.feature(temp)
        feature_df.to_csv("data_img_2.csv", index=False)
        table = dbc.Table.from_dataframe(
            feature_df.drop(columns=['x1', 'y1', 'x2', 'y2']), bordered=True)
        output = html.Div([dcc.Graph(figure=fig), table])
        try:
            dimension_df = vision_api.dimension(feature_df)
            dimension_df.to_csv("data_img_2.csv", index=False)
            dim_table = dbc.Table.from_dataframe(
                dimension_df.drop(columns=['x1', 'y1', 'x2', 'y2']),
                bordered=True)
            image = cv2.imread(filename)
            model_output = main_obj.main(image, dimension_df, rotation_angle=0)
            quad_df, quad_image = model_output['final_df'], model_output[
                'rotated_org_image']
            cv2.imwrite('quad2.png', quad_image)
            quad_img = cv2.imread('quad2.png')
            quad_fig = px.imshow(quad_img)
            quad_fig.update_yaxes(visible=False)
            quad_fig.update_xaxes(visible=False)
            quad_df.rename(columns={
                'Quadrant': 'Quad',
                'Feature': 'Feature',
                'Total Area(Sq.Ft.)': 'Total area',
                "Area(Sq.Ft.)_in_Quadrant": 'Area in Quadrant',
                "Area(Sq.Ft.)_in_Quadrant %": '% Area in Quadrant'
            },
                           inplace=True)
            quad_table = dbc.Table.from_dataframe(quad_df, bordered=True)
            from app import summary
            output = html.Div([
                dcc.Graph(figure=fig), dim_table,
                dcc.Graph(figure=quad_fig), quad_table, summary.layout
            ])
        except:
            output = html.Div([
                dcc.Graph(figure=fig), table,
                html.H3("Dimension has lots of exception to handle.")
            ])
    except:
        output = html.Div([
            dcc.Graph(figure=fig), table,
            html.H3("The ocr output is having trouble.")
        ])
    return html.Div([output])


@app.callback(Output('output-image-upload_p2', 'children'),
              Input('upload-image_p2', 'contents'),
              State('upload-image_p2', 'filename'))


def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n)
            for c, n in zip(list_of_contents, list_of_names)
        ]
        return children


summary_layout = html.Div([
    html.Br(),
    dbc.Button("Summary",
               id="button",
               external_link=True,
               style={
                   "background-color": "#292929",
                   "height": "40px"
               }),  # NOQA E501
    html.Br(),
    html.Div(id="summary")
])  # NOQA E501


@app.callback(
    Output("summary", "children"),
    Input("button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    df1 = pd.read_csv("data_img_1.csv")
    df2 = pd.read_csv("data_img_2.csv")
    df1.drop(columns=['x1', 'y1', 'x2', 'y2'], inplace=True)
    df2.drop(columns=['x1', 'y1', 'x2', 'y2'], inplace=True)
    # print("df1", df1.info())
    # print("df2", df2.info())
    print("dataframed are read")

    df1 = best_features.std_df(df1)
    print('df1.head() : \n', df1.head())
    print("df1.info \n -------------------------\n", df1.info())
    new_1 = best_features.sort_data(df1)
    df2 = best_features.std_df(df2)
    print("df2.info \n -------------------------\n", df2.info())
    new_2 = best_features.sort_data(df2)
    df3, df4 = best_features.final_summary(new_1, new_2)
    # print("df3", df3)
    # print("df4", df4)
    df3.to_csv('best_features.csv', index=False)
    df4.to_csv('uncommon_features.csv', index=False)
    df3 = pd.read_csv('best_features.csv')
    df4 = pd.read_csv('uncommon_features.csv')
    print('got best features')

    for i in range(len(df3['Features'])):
        if df3['floorplan1'][i] < df3['floorplan2'][i]:
            df3['floorplan1'][i] = 0
            df3['floorplan2'][i] = 1
        elif df3['floorplan1'][i] > df3['floorplan2'][i]:
            df3['floorplan1'][i] = 1
            df3['floorplan2'][i] = 0
        else:
            df3['floorplan1'][i] = 1
            df3['floorplan2'][i] = 1

    if df3["floorplan1"].sum() > df3["floorplan2"].sum():
        conclusion = "Floor plan 1 is better than Floor plan 2."
    if df3["floorplan1"].sum() < df3["floorplan2"].sum():
        conclusion = "Floor plan 2 is better than Floor plan 1."
    if df3["floorplan1"].sum() == df3["floorplan2"].sum():
        conclusion = "Floor plan 1 and Floor plan 2 both are same."

    df3['floorplan1'] = df3['floorplan1'].apply(lambda x: '❌' if x == 0 else
                                                ('✔️' if x == 1 else ''))
    df3['floorplan2'] = df3['floorplan2'].apply(lambda x: '❌' if x == 0 else
                                                ('✔️' if x == 1 else ''))

    try:
        ext = ('.png', '.jpg', '.PNG', '.JPG', '.jpeg', ".csv")
        for file in os.listdir():
            if file.endswith(ext):
                if file == 'google vision image analysis.csv':
                    pass
                else:
                    print("Removing ", file)
                    os.remove(file)
    except Exception as e:
        print(e)
    finally:
        print("No garbage available")

    return dbc.Container([
        html.H3("Plan Comparison",
                className="display-6",
                style={'textAlign': 'left'}),
        dbc.Table.from_dataframe(df3, bordered=True),
        dbc.Table.from_dataframe(df4, bordered=True),
        html.H3("Conclusion",
                className="display-6",
                style={'textAlign': 'left'}),
        dbc.Card(html.P(conclusion), body=True)
    ])