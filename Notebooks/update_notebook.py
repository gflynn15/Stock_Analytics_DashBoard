import json

def update_notebook():
    with open('Stock_Review.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # --- Update Layout (Cell 7) ---
    # Find the end of the container and insert the accordion
    layout_source = nb['cells'][7]['source']
    new_layout = []
    for line in layout_source:
        if '], fluid=True)' in line:
            # Add the accordion before closing the container
            new_layout.append('    ###===================================================COMPANY SUMMARY===================================================###\n')
            new_layout.append('    dbc.Row([\n')
            new_layout.append('        dbc.Col([\n')
            new_layout.append('            dbc.Accordion([\n')
            new_layout.append('                dbc.AccordionItem([\n')
            new_layout.append('                    html.P(\n')
            new_layout.append('                        id="company-summary",\n')
            new_layout.append('                        className="animate__animated animate__fadeIn",\n')
            new_layout.append('                        style={"fontSize": 22, "textAlign": "justify", "color": "white"}\n')
            new_layout.append('                    )\n')
            new_layout.append('                ], title=html.Span("🏢 Click to Review Company Summary", style={"fontSize": 30}))\n')
            new_layout.append('            ], start_collapsed=True)\n')
            new_layout.append('        ], width=12)\n')
            new_layout.append('    ], className="mb-4"),\n')
            new_layout.append('\n')
        new_layout.append(line)
    nb['cells'][7]['source'] = new_layout

    # --- Update Callback (Cell 8) ---
    callback_source = nb['cells'][8]['source']
    new_callback = []
    skip = False
    for line in callback_source:
        # Add the new output
        if 'Output("MACD", "figure"),' in line:
            new_callback.append(line)
            new_callback.append('    Output("company-summary", "children"),\n')
            continue
        
        # Start of the breakdown section that needs fixing
        if '###=============================================FUNDAMENTALS BREAKDOWN==============================================###' in line:
            new_callback.append('###=============================================FUNDAMENTALS BREAKDOWN==============================================###\n')
            new_callback.append('    with engine.connect() as conn:\n')
            new_callback.append('        ticker_clean = ticker.split("-")[0]\n')
            new_callback.append('        # Query the summary\n')
            new_callback.append('        summary_df = pd.read_sql(text(f"SELECT longBusinessSummary FROM FUNDAMENTAL_DATA WHERE \\"index\\" = \'{ticker_clean}\'"), con=conn)\n')
            new_callback.append('        summary_text = summary_df["longBusinessSummary"].iloc[0] if not summary_df.empty else "No summary available for this company."\n')
            new_callback.append('        \n')
            new_callback.append('        # Fetching other fundamentals if needed\n')
            new_callback.append('        fundamentals = pd.read_sql(text(f"""SELECT * FROM FUNDAMENTAL_DATA WHERE \\"index\\" = \'{ticker_clean}\'"""), con=conn).rename(columns={\\"index\\":\\"Company\\"})\n')
            new_callback.append('        fundamentals.columns = fundamentals.columns.str.upper()\n')
            new_callback.append('        \n')
            new_callback.append('    return news_table, trend_fig, rsi_fig, macd_fig, summary_text\n')
            skip = True # Skip the rest of the original cell as we just replaced the return and broken logic
            break
        
        if not skip:
            new_callback.append(line)
            
    nb['cells'][8]['source'] = new_callback

    # Save the notebook
    with open('Stock_Review.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Notebook updated successfully!")

if __name__ == "__main__":
    update_notebook()
