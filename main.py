import dash
import dash_bootstrap_components as dbc

# meta_tags are required for the app layout to be mobile responsive
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True,  # NOQA E501
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, "background-color": "#D1CBC9", initial-scale=1.0'}]  # NOQA E501
                )
app.title = 'Floor Plan'
server = app.server
