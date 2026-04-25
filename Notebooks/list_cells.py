import json

def list_cells():
    with open('Stock_Review.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        first_line = cell['source'][0] if cell['source'] else "EMPTY"
        print(f"Cell {i}: {first_line.strip()[:50]}")

if __name__ == "__main__":
    list_cells()
