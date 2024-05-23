from Image_Processing_Pipeline import main
from visionocrv6 import vision_api
import pathlib
import plotly.express as px
from dash import Input, Output, html, dcc, State
import cv2
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from main import app
import warnings
import base64
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
main_obj = main()

warnings.filterwarnings('ignore')

PATH = pathlib.Path(__file__).parent

layout = dbc.Container([
    html.Br(),
    dcc.Upload(id='upload-image_p1',
               children=html.Div(
                   [html.A('Select or Upload the First Floorplan')]),
               style={
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   "height": "60px"
               },
               multiple=True),
    html.Br(),
    html.Div(id='output-image-upload_p1'),
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
    temp_df.to_csv("data_img_1.csv", index=False)
    table = dbc.Table.from_dataframe(temp_df, bordered=True)
    try:
        feature_df = vision_api.feature(temp)
        feature_df.to_csv("data_img_1.csv", index=False)
        table = dbc.Table.from_dataframe(
            feature_df.drop(columns=['x1', 'y1', 'x2', 'y2']), bordered=True)
        output = html.Div([dcc.Graph(figure=fig), table])
        try:
            dimension_df = vision_api.dimension(feature_df)
            dimension_df.to_csv("data_img_1.csv", index=False)
            dim_table = dbc.Table.from_dataframe(
                dimension_df.drop(columns=['x1', 'y1', 'x2', 'y2']),
                bordered=True)
            image = cv2.imread(filename)
            model_output = main_obj.main(image, dimension_df, rotation_angle=0)
            quad_df, quad_image = model_output['final_df'], model_output[
                'rotated_org_image']
            cv2.imwrite('quad1.png', quad_image)
            quad_img = cv2.imread('quad1.png')
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
            output = html.Div([
                dcc.Graph(figure=fig), dim_table,
                dcc.Graph(figure=quad_fig), quad_table
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


@app.callback(Output('output-image-upload_p1', 'children'),
              Input('upload-image_p1', 'contents'),
              State('upload-image_p1', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n)
            for c, n in zip(list_of_contents, list_of_names)
        ]
        return children