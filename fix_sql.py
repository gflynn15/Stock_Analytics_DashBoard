import os
import re

files_to_fix = [
    "pages/MARKET_PULSE_1_1.py",
    "pages/MACRO_HEALTH_1_1.py",
    "pages/Stock_Review_1_1.py",
    "app_functions.py"
]

for file in files_to_fix:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace pd.read_sql(text(...), con=conn)
    # Be careful with the closing parenthesis of text()
    # We can use regex: text(  (something)  )
    
    # We want to remove text( and the closing parenthesis before , con=conn
    # pd.read_sql(text('SELECT ...'), con=conn)
    new_content = re.sub(r'text\((.*?)\)(,\s*con=conn)', r'\1\2', content, flags=re.DOTALL)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(new_content)
