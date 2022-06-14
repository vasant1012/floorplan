from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from main import app
from app import twobhk, threebhk

app.layout = html.Div(
    [dcc.Location(id='url', refresh=False),
     html.Div(id='page-content')])

index_page = dbc.Container([html.H1('Floor Plan Usecase', style={'textAlign': 'center'}),html.Br(),
                           dbc.DropdownMenu([
        dbc.DropdownMenuItem("2 BHK", href='/app/twobhk',style={"width": "1100px"}),
        dbc.DropdownMenuItem("3 BHK", href='/app/threebhk',style={"width": "1100px"}),
    ], label="Select Flat", toggle_style={"width": "1100px"})])


# Update the index
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname=index_page):
    if pathname == '/app/twobhk':
        return html.Div([index_page, html.Br(), twobhk.layout])
    elif pathname == '/app/threebhk':
        return html.Div([index_page, html.Br(), threebhk.layout])
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=8091)