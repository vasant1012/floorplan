import time
from main import app
import plotly.express as px
from dash import Input, Output, State, html, dcc
import cv2
import json
from skimage import io
import dash_bootstrap_components as dbc
import numpy as np
from os import path
from ocr import ocr
import pathlib
import warnings
import base64

warnings.filterwarnings('ignore')

PATH = pathlib.Path(__file__).parent

layout = dbc.Container([
    html.Br(),
    dcc.Upload(id='upload-image_p1',
               children=html.Div(['Drag and Drop or ',
                                  html.A('Select Files')]),
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

def parse_contents(contents):
    contents = str(contents[0])
    encoded_data = contents.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("img_1.jpg", img)
    return html.Div([annot_layout])


@app.callback(Output('output-image-upload_p1', 'children'),
              Input('upload-image_p1', 'contents'))
def update_output(list_of_contents):
    if list_of_contents is not None:
        children = [parse_contents(c) for c in zip(list_of_contents)]
        return children

if path.exists("img_1.jpg"):
    img_1 = io.imread("img_1.jpg")
else:
    time.sleep(2)
    img_1 = np.full((500, 500, 3), 255, dtype = np.uint8)

fig_1 = px.imshow(img_1)
fig_1.update_layout(dragmode="drawrect")

# Build App
annot_layout = html.Div([
    dcc.Graph(id="fig_1", figure=fig_1),
    dbc.Container([
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.Br(),
                html.Label('Living Room : Click and Select'),
                html.Br(),
                dbc.Card(id="card_1_p1", body=True),
                dbc.Button("Click",
                           id="btn_1_p1",
                           outline=True,
                           color="primary",
                           className="me-1"),
                html.Br(),
                html.Br(),
                html.Label('Master Bedroom : Click and Select'),
                html.Br(),
                dbc.Card(id="card_2_p1", body=True),
                dbc.Button("Click",
                           id="btn_2_p1",
                           outline=True,
                           color="primary",
                           className="me-1"),
                html.Br(),
                html.Br(),
                html.Label('Toilet : Click and Select'),
                html.Br(),
                dbc.Card(id="card_3_p1", body=True),
                dbc.Button("Click",
                           id="btn_3_p1",
                           outline=True,
                           color="primary",
                           className="me-1"),
                html.Br(),
                html.Br(),
                html.Label('Bedroom : Click and Select'),
                html.Br(),
                dbc.Card(id="card_4_p1", body=True),
                dbc.Button("Click",
                           id="btn_4_p1",
                           outline=True,
                           color="primary",
                           className="me-1"),
                html.Br(),
                html.Br(),
                html.Label('Kitchen : Click and Select'),
                html.Br(),
                dbc.Card(id="card_5_p1", body=True),
                dbc.Button("Click",
                           id="btn_5_p1",
                           outline=True,
                           color="primary",
                           className="me-1"),
                html.Br(),
                html.Br(),
                html.Label('Dry Balcony : Click and Select'),
                html.Br(),
                dbc.Card(id="card_6_p1", body=True),
                dbc.Button("Click",
                           id="btn_6_p1",
                           outline=True,
                           color="primary",
                           className="me-1")
            ])
        ])
    ])
])


@app.callback(Output('card_1_p1', 'children'),
              Input('btn_1_p1', 'n_clicks'),
              State("fig_1", "relayoutData"),
              prevent_initial_call=True)
@app.callback(Output('card_2_p1', 'children'),
              Input('btn_2_p1', 'n_clicks'),
              State("fig_1", "relayoutData"),
              prevent_initial_call=True)
@app.callback(Output('card_3_p1', 'children'),
              Input('btn_3_p1', 'n_clicks'),
              State("fig_1", "relayoutData"),
              prevent_initial_call=True)
@app.callback(Output('card_4_p1', 'children'),
              Input('btn_4_p1', 'n_clicks'),
              State("fig_1", "relayoutData"),
              prevent_initial_call=True)
@app.callback(Output('card_5_p1', 'children'),
              Input('btn_5_p1', 'n_clicks'),
              State("fig_1", "relayoutData"),
              prevent_initial_call=True)
@app.callback(Output('card_6_p1', 'children'),
              Input('btn_6_p1', 'n_clicks'),
              State("fig_1", "relayoutData"),
              prevent_initial_call=True)
def button(btn, relayout_data):
    if btn > 0 and relayout_data == {'autosize': True}:
        return "Please draw a rectangle on image and click."
    elif btn > 0 and relayout_data:
        for key in relayout_data:
            if "shapes" in key:
                data = json.dumps(f'{relayout_data[key]}')
                data = json.loads(data)
                data = eval(data)
                data = data[0:]
                for i in data:
                    keys, values = list(i.keys()), list(i.values())
                    cord = dict(zip(keys[-4:], values[-4:]))
                cropped_image = img_1[int(cord.get("y0")):int(cord.get("y1")),
                                    int(cord.get("x0")):int(cord.get("x1"))]
                lst = ocr.img_to_str(cropped_image)
                final = ocr.summary(lst)
                return html.Div([
                    html.H5(f"Type of Room - {final[2]}",
                            className="card-title"),
                    html.P([
                        f"Dimensions of Room - Length - {final[0]} ft & Breath - {final[1]} ft",
                        html.Br(), f"Area of the Room  - {final[3]} sq.ft."
                    ],
                        className="card-text")
                ])