import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
# 1. Import App, Server, and Cache from our new initialization file
from app_init import app, server, cache

app.layout = dbc.Container([    
    dbc.Row([
        dbc.Row([dbc.Col([
                        dbc.Button("Menu", id="menu-button", color="primary", size="lg", className="m-2")], width=12),
                        ]),
        
        dbc.Col(dbc.Nav(
            [dbc.NavLink(page['name'], href=page['path'], active="exact") for page in dash.page_registry.values()],
            pills=True, justified=False, vertical=True
        ), width=3, style={"display":"none"}, id="menu-col"),
    ]),
    
    # This dynamically swaps content from your pages_dir folder
    dash.page_container 
], fluid=True)

# 3. The Callback to toggle visibility
@app.callback(
    Output("menu-col", "style"),
    Input("menu-button", "n_clicks"),
    State("menu-col", "style"),
    prevent_initial_call=True
)
def toggle_navbar(n_clicks, current_style):
    if current_style.get("display") == "none":
        return {"display": "block"}
    return {"display": "none"}

if __name__ == "__main__":
    app.run(debug=True)

