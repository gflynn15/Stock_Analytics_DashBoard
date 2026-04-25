import dash
from flask import Flask
from flask_caching import Cache
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# Optional safety measure for interactive environments (Jupyter/VS Code Interactive)
# This clears the global registry so running the script multiple times doesn't cause Duplicate Callback errors
import dash._callback
dash._callback.GLOBAL_CALLBACK_LIST = []
dash._callback.GLOBAL_CALLBACK_MAP = {}
dash.page_registry.clear()

# 1. Initialize Flask Server FIRST
server = Flask(__name__)

# 2. Initialize Cache using the Flask server
cache = Cache(server, config={
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'cache_file',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_THRESHOLD': 500
})

# 3. Initialize Dash App
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.CYBORG, 
        "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css",
        "https://unpkg.com/aos@next/dist/aos.css" # Added AOS stylesheet
    ], 
    external_scripts=[
        "https://unpkg.com/aos@next/dist/aos.js" # Added AOS script
    ],
    use_pages=True, 
    pages_folder="pages", 
    server=server
)
load_figure_template('cyborg')
