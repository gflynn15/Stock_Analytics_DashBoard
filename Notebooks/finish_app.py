import json

def complete_fundamentals():
    with open('Stock_Review.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Insert dash_table_create function cell (at index 5)
    table_func_code = [
        "def dash_table_create(df):\n",
        "    if df is None or df.empty:\n",
        "        return html.Div(\"No data available.\")\n",
        "\n",
        "    num_cols = len(df.columns)\n",
        "    style_cell_conditional = [\n",
        "        {'if': {'column_id': df.columns[0]}, 'width': '70%', 'minWidth': '70%', 'maxWidth': '70%'}\n",
        "    ]\n",
        "    if num_cols > 1:\n",
        "        rem = f\"{30 / (num_cols - 1)}%\"\n",
        "        for c in df.columns[1:]: \n",
        "            style_cell_conditional.append({'if': {'column_id': c}, 'width': rem, 'minWidth': rem, 'maxWidth': rem})\n",
        "\n",
        "    return dash_table.DataTable(\n",
        "        columns=[{\"name\": i, \"id\": i} for i in df.columns],\n",
        "        data=df.to_dict(\"records\"),\n",
        "        style_cell_conditional=style_cell_conditional,\n",
        "        style_table={\"overflowX\": \"auto\", \"border\": \"1px solid #ccc\"},\n",
        "        style_data={\"padding\": \"5px\", \"height\": \"auto\", \"textAlign\": \"left\", \"color\":\"white\", \"backgroundColor\": \"#585859\", \"fontSize\":22, \"fontFamily\":\"Inter\"},\n",
        "        style_header={\"padding\": \"5px\", \"height\": \"auto\", \"backgroundColor\": \"#eb4904d4\", \"color\": \"white\", \"fontWeight\": \"bold\", \"textAlign\": \"center\", \"fontSize\":25},\n",
        "    )\n"
    ]
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": table_func_code
    }
    nb['cells'].insert(5, new_cell)

    # 2. Update Layout (now at index 8 after insert)
    layout_source = nb['cells'][8]['source']
    new_layout = []
    for line in layout_source:
        if '], fluid=True)' in line:
            new_layout.append('    ###===================================================FUNDAMENTALS TABLES==============================================###\n')
            new_layout.append('    dbc.Row([\n')
            new_layout.append('        dbc.Col([html.H3("📊 Valuation"), html.Div(id="val-table")], width=4),\n')
            new_layout.append('        dbc.Col([html.H3("💰 Financials"), html.Div(id="fin-table")], width=4),\n')
            new_layout.append('        dbc.Col([html.H3("⚠️ Risk & Shares"), html.Div(id="risk-table")], width=4),\n')
            new_layout.append('    ], className="mb-4"),\n')
            new_layout.append('\n')
        new_layout.append(line)
    nb['cells'][8]['source'] = new_layout

    # 3. Update Callback (now at index 9)
    cb_source = nb['cells'][9]['source']
    new_cb = []
    skip = False
    for line in cb_source:
        if 'Output("company-summary", "children"),' in line:
            new_cb.append(line)
            new_cb.append('    Output("val-table", "children"),\n')
            new_cb.append('    Output("fin-table", "children"),\n')
            new_cb.append('    Output("risk-table", "children"),\n')
            continue
        
        if '###=============================================FUNDAMENTALS BREAKDOWN==============================================###' in line:
            new_cb.append('    ###=============================================FUNDAMENTALS BREAKDOWN==============================================###\n')
            new_cb.append('    with engine.connect() as conn:\n')
            new_cb.append('        ticker_clean = ticker.split("-")[0]\n')
            new_cb.append('        full_df = pd.read_sql(text(f"SELECT * FROM FUNDAMENTAL_DATA WHERE \\\"index\\\" = \'{ticker_clean}\'"), con=conn)\n', )
            new_cb.append('        full_df.columns = full_df.columns.str.upper()\n')
            new_cb.append('        \n')
            new_cb.append('        # 1. Company Summary\n')
            new_cb.append('        summary_text = full_df["LONGBUSINESSSUMMARY"].iloc[0] if not full_df.empty else \"No summary available.\"\n')
            new_cb.append('        \n')
            new_cb.append('        # Helper to create vertical tables\n')
            new_cb.append('        def get_vertical_table(df, cols):\n')
            new_cb.append('            if df.empty: return dash_table_create(pd.DataFrame())\n')
            new_cb.append('            subset = df[cols].T.reset_index()\n')
            new_cb.append('            subset.columns = ["METRIC", "VALUE"]\n')
            new_cb.append('            return dash_table_create(subset)\n')
            new_cb.append('\n')
            new_cb.append('        # 2. Valuation Table\n')
            new_cb.append('        val_cols = ["MARKETCAP", "ENTERPRISEVALUE", "TRAILINGPE", "FORWARDPE", "PRICETOBOOK", "TRAILINGPEGRATIO"]\n')
            new_cb.append('        val_table = get_vertical_table(full_df, [c for c in val_cols if c in full_df.columns])\n')
            new_cb.append('\n')
            new_cb.append('        # 3. Financials Table\n')
            new_cb.append('        fin_cols = ["TOTALREVENUE", "GROSSPROFITS", "PROFITMARGINS", "FREECASHFLOW", "TOTALDEBT", "RETURNONEQUITY"]\n')
            new_cb.append('        fin_table = get_vertical_table(full_df, [c for c in fin_cols if c in full_df.columns])\n')
            new_cb.append('\n')
            new_cb.append('        # 4. Risk & Shares Table\n')
            new_cb.append('        risk_cols = ["OVERALLRISK", "BETA", "SHARESOUTSTANDING", "SHORTPERCENTOFFLOAT", "SHORTRATIO"]\n')
            new_cb.append('        risk_table = get_vertical_table(full_df, [c for c in risk_cols if c in full_df.columns])\n')
            new_cb.append('\n')
            new_cb.append('    return news_table, trend_fig, rsi_fig, macd_fig, summary_text, val_table, fin_table, risk_table\n')
            skip = True
            break
        
        if not skip:
            new_cb.append(line)
    nb['cells'][9]['source'] = new_cb

    with open('Stock_Review.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    complete_fundamentals()
