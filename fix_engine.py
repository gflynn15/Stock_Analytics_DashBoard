import os
import re

files_to_fix = [
    "pages/MARKET_PULSE_1_1.py",
    "pages/MACRO_HEALTH_1_1.py",
    "pages/Stock_Review_1_1.py",
    "app_functions.py"
]

for file_path in files_to_fix:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Replace 'with engine.connect() as conn:\n    df = pd.read_sql(..., con=conn)'
    # with 'df = pd.read_sql(..., con=engine)'
    # This needs to be careful with indentation.
    
    # Pattern to match:
    # with engine.connect() as conn:
    #     variable = pd.read_sql(query, con=conn)
    
    # Simple regex for the most common pattern:
    new_content = re.sub(
        r'with engine\.connect\(\) as conn:\n\s+(.*? = pd\.read_sql\(.*?), con=conn\)',
        r'\1, con=engine)',
        content,
        flags=re.MULTILINE
    )
    
    # Also handle multiple lines if needed
    # (e.g. Stock_Review_1_1.py line 150)
    
    # Another pattern:
    # with engine.connect() as conn:
    #     df = pd.read_sql(
    #         query,
    #         con=conn
    #     )
    
    # Let's just do a simpler approach:
    # Replace all con=conn with con=engine if they are inside pd.read_sql
    # and then remove the with engine.connect() lines.
    
    # Step 1: Replace con=conn with con=engine inside pd.read_sql calls
    new_content = re.sub(r'(pd\.read_sql\(.*?), con=conn\)', r'\1, con=engine)', new_content)
    
    # Step 2: Remove the 'with engine.connect() as conn:' line and fix indentation
    # This is tricky with regex. Let's try a line-by-line approach for safety.
    
    lines = new_content.split('\n')
    new_lines = []
    skip_next = False
    indent_to_remove = 0
    
    for i in range(len(lines)):
        line = lines[i]
        if 'with engine.connect() as conn:' in line:
            # Check if next line is a pd.read_sql
            if i + 1 < len(lines) and 'pd.read_sql' in lines[i+1]:
                indent_to_remove = len(line) - len(line.lstrip()) + 4
                continue # Skip this line
        
        if indent_to_remove > 0 and line.startswith(' ' * indent_to_remove):
            new_lines.append(line[4:]) # Remove 4 spaces
            # If next line doesn't have the same or more indent, reset
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line.strip() and not next_line.startswith(' ' * indent_to_remove):
                    indent_to_remove = 0
            continue
        
        new_lines.append(line)
        indent_to_remove = 0
            
    final_content = '\n'.join(new_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
