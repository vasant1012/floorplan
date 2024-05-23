from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from main import app
import base64
from app import onebhk, threebhk, twobhk, fourbhk


godrej_logo = './assets/godrej_logo.png'
encoded_image_gl = base64.b64encode(open(godrej_logo, 'rb').read())
bizlogo = './assets/bizlogo.jpg'
encoded_image_bl = base64.b64encode(open(bizlogo, 'rb').read())


try:
    ext = ('.png', '.jpg', '.csv', '.PNG', '.JPG', '.jpeg')
    for file in os.listdir():
        if file.endswith(ext):
            if file=='google vision image analysis.csv':
                pass
            else:
                print("Removing ", file)
                os.remove(file)
except Exception as e: print(e)
finally:
    print("No garbage available")
    

app.layout = html.Div([html.Br(), dbc.Row([
                # html.Img(src='./assets/godrej_logo.png',  # NOQA E501
                # style={"max-width": "250px", "height": "auto", "margin-left": "20px"}),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image_gl.decode()),
                style={"max-width": "250px", "height": "auto", "margin-left": "20px"}),
                dbc.Col(html.H1('Floor Layout Comparison', style={'textAlign': 'center'})),
            #     html.Img(src='./assets/bizogo.jpg',  # NOQA E501
            #  style={"max-width": "100px", "height": "auto", "margin-right": "20px"}),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image_bl.decode()),
            style={"max-width": "100px", "height": "auto", "margin-right": "20px"})
            ]),
        html.Br(),
     dcc.Location(id='url', refresh=True),
     html.Div(id='page-content')])

app.title = "Floor Layout Comparison"

index_page = dbc.Container([
    dbc.DropdownMenu([
        dbc.DropdownMenuItem("1 BHK", href='/app/onebhk',style={"width": "1100px"}),
        dbc.DropdownMenuItem("2 BHK", href='/app/twobhk',style={"width": "1100px"}),
        dbc.DropdownMenuItem("3 BHK", href='/app/threebhk',style={"width": "1100px"}),
        dbc.DropdownMenuItem("4 BHK", href='/app/fourbhk',style={"width": "1100px"}),
    ], label="Select Flat", toggle_style={"width": "1100px"})])


# Update the index
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname=index_page):
    if pathname == '/app/onebhk':
        return html.Div([index_page, html.Br(), onebhk.layout])
    elif pathname == '/app/twobhk':
        return html.Div([index_page, html.Br(), twobhk.layout])
    elif pathname == '/app/threebhk':
        return html.Div([index_page, html.Br(), threebhk.layout])
    elif pathname == '/app/fourbhk':
        return html.Div([index_page, html.Br(), fourbhk.layout])
    else:
        return index_page

if __name__ == '__main__':
    # app.run_server(debug=False, use_reloader=False, port=7920)
    app.run_server(port=8030, debug=False, use_reloader=False)