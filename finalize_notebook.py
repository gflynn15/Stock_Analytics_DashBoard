import json

def finalize_notebook():
    with open('Notebooks/Stock_Review.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Update Callback (Cell 9)
    cb_source = nb['cells'][9]['source']
    new_cb = []
    skip = False
    
    for line in cb_source:
        # Add the missing Outputs
        if 'Output("company-risk","children"),' in line:
            new_cb.append(line)
            new_cb.append('    Output("company-financials", "children"),\n')
            new_cb.append('    Output("company-performance", "children"),\n')
            continue
            
        # Replace the entire fundamentals logic at the end
        if '###=============================================FUNDAMENTALS BREAKDOWN==============================================###' in line:
            new_cb.append('###=============================================FUNDAMENTALS BREAKDOWN==============================================###\n')
            new_cb.append('    company = ticker.split("-")[0]\n')
            new_cb.append('    with engine.connect() as conn:\n')
            new_cb.append('        fundamentals = pd.read_sql(text(f"SELECT * FROM FUNDAMENTAL_DATA WHERE \\"index\\\" = \'{company}\'"), con=conn).rename(columns={\\"index\\":\\"Company\\"})\n')
            new_cb.append('        fundamentals.columns = fundamentals.columns.str.upper()\n')
            new_cb.append('        \n')
            new_cb.append('        # Industry average for risk/beta context\n')
            new_cb.append('        sector = fundamentals["SECTOR"].iloc[0] if not fundamentals.empty else \"Unknown\"\n')
            new_cb.append('        sector_fundamentals = pd.read_sql(text(f"SELECT * FROM FUNDAMENTAL_DATA WHERE sector = \'{sector}\'"), con=conn)\n')
            new_cb.append('        sector_fundamentals.columns = sector_fundamentals.columns.str.upper()\n')
            new_cb.append('\n')
            new_cb.append('    # 1. Company Summary\n')
            new_cb.append('    summary_text = fundamentals["LONGBUSINESSSUMMARY"].iloc[0] if not fundamentals.empty else \"Summary unavailable.\"\n')
            new_cb.append('\n')
            new_cb.append('    # Helper for vertical tables\n')
            new_cb.append('    def create_v_table(df, cols, label_name=\"METRIC\"):\n')
            new_cb.append('        if df.empty: return dash_table_create(pd.DataFrame())\n')
            new_cb.append('        sub = df[cols].T.reset_index()\n')
            new_cb.append('        sub.columns = [label_name, \"VALUE\"]\n')
            new_cb.append('        return dash_table_create(sub)\n')
            new_cb.append('\n')
            new_cb.append('    # 2. Risk Table (with Industry Average)\n')
            new_cb.append('    risk_cols = [c for c in fundamentals.columns if \"RISK\" in c or \"BETA\" in c]\n')
            new_cb.append('    risk_df = fundamentals[risk_cols].T.reset_index()\n')
            new_cb.append('    risk_df.columns = [\"METRIC\", \"VALUE\"]\n')
            new_cb.append('    # Add sector avg\n')
            new_cb.append('    sector_avg = sector_fundamentals[risk_cols].mean().reset_index()\n')
            new_cb.append('    sector_avg.columns = [\"METRIC\", \"INDUSTRY AVG\"]\n')
            new_cb.append('    risk_final = risk_df.merge(sector_avg, on=\"METRIC\").round(2)\n')
            new_cb.append('    risk_table = dash_table_create(risk_final)\n')
            new_cb.append('\n')
            new_cb.append('    # 3. Financials Table\n')
            new_cb.append('    fin_cols = ["TOTALREVENUE", "GROSSPROFITS", "PROFITMARGINS", "EBITDA", "TOTALDEBT", "DEBTTOEQUITY"]\n')
            new_cb.append('    fin_table = create_v_table(fundamentals, [c for c in fin_cols if c in fundamentals.columns])\n')
            new_cb.append('\n')
            new_cb.append('    # 4. Performance & Valuation Table\n')
            new_cb.append('    perf_cols = ["MARKETCAP", "TRAILINGPE", "FORWARDPE", "PRICETOBOOK", "REVENUEGROWTH", "EARNINGSGROWTH"]\n')
            new_cb.append('    perf_table = create_v_table(fundamentals, [c for c in perf_cols if c in fundamentals.columns])\n')
            new_cb.append('\n')
            new_cb.append('    return news_table, trend_fig, rsi_fig, macd_fig, summary_text, risk_table, fin_table, perf_table\n')
            skip = True
            break
        
        if not skip:
            new_cb.append(line)
            
    nb['cells'][9]['source'] = new_cb
    
    with open('Notebooks/Stock_Review.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Notebook finalized successfully!")

if __name__ == "__main__":
    finalize_notebook()
