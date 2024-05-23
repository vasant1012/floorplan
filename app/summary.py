import pandas as pd
from dash import html
from main import app
import os
from dash import Input, Output
import dash_bootstrap_components as dbc
from best_features import best_features
import warnings
warnings.filterwarnings('ignore')
import pathlib

PATH = pathlib.Path(__file__).parent

layout = html.Div([
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