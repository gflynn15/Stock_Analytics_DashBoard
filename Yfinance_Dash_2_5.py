import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from flask_caching import Cache

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX], use_pages=True, pages_folder="pages")
# --- SETUP FILESYSTEM CACHE ---
cache = Cache(app.server, config={
    # Use the file system to store cache data
    'CACHE_TYPE': 'FileSystemCache',
    
    # The folder where cache files will be saved
    'CACHE_DIR': 'cache_directory',
    
    # Default timeout (5 minutes). 
    # Data older than this is automatically deleted.
    'CACHE_DEFAULT_TIMEOUT': 300,
    
    # Maximum number of items in the cache (prevents filling up the disk)
    'CACHE_THRESHOLD': 500
})
server=app.server
load_figure_template('simplex')

# hand-shake for the cache
cache.init_app(app.server, config={
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'cache-directory',
})

# ONLY shared layout elements stay here
app.layout = dbc.Container([
    dbc.Row([
        dbc.Row([dbc.Col([html.H1("FIN HUB", style={"textAlign":"center","fontWeight":"bold"}),
                          dbc.Button("Menu", id="menu-button", color="primary", className="m-2")], width=12),
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

