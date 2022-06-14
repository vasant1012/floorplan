from dash import html
import dash_bootstrap_components as dbc
import pathlib
import warnings
from os import path, remove
from app import plan1, plan2
warnings.filterwarnings('ignore')


PATH = pathlib.Path(__file__).parent
if path.exists("img_2.jpg"):
    remove("img_2.jpg")
elif path.exists("img_1.jpg"):
    remove("img_1.jpg")

layout = html.Div([html.H3('2 BHK', style={'textAlign': 'center'}),
            dbc.Row([
                dbc.Col(html.Div(plan1.layout), width=6),
                dbc.Col(html.Div(plan2.layout), width=6)]),
    ])
