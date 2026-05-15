import pandas as pd
import os
import re

def prepare_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', 'DATASET2.xlsx')
    output_file = os.path.join(base_dir, 'data', 'processed', 'processed_dataset.csv')
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    xl = pd.ExcelFile(file_path)
    all_data = []
    
    target_sheets = [s for s in xl.sheet_names if '%' in s and 'Plot' not in s]
    print(f"Targeting data sheets: {target_sheets}")
    
    for sheet_name in target_sheets:
        print(f"Processing sheet: {sheet_name}...")
        df = pd.read_excel(xl, sheet_name=sheet_name)
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        
        # Calculate precise KM from sheet name (Mandate: 5% -> 0.25km on 5.0km line)
        km_val = 0.0
        pct_match = re.search(r'(\d+)%', sheet_name)
        if pct_match:
            pct = float(pct_match.group(1))
            km_val = (pct / 100.0) * 5.0

        # 1. Standardize Detection
        if 'Detection' in df.columns:
            df['Detection'] = df['Detection'].apply(lambda x: str(x).strip().capitalize())
        else:
            df['Detection'] = 'Fault' if km_val > 0 else 'Normal'

        # 2. Clean Classification
        if 'Classification' in df.columns:
            df['Classification'] = df['Classification'].apply(lambda x: str(x).strip())
        else:
            # Infer classification from sheet name (e.g., "A-G 5%" -> "A-G Fault")
            base_class = sheet_name.split(' ')[0]
            df['Classification'] = f"{base_class} Fault" if km_val > 0 else 'Health Condition'
            
        # Hardening: If Detection is Normal, Classification MUST be Health Condition
        df.loc[df['Detection'] == 'Normal', 'Classification'] = 'Health Condition'

        # 3. Handle Location (Precise calculation for Fault rows, 0.0 for Normal)
        df['Fault Location (km)'] = df['Detection'].apply(lambda x: km_val if x == 'Fault' else 0.0)

        # 4. Select Features + Detection/Classification in EXACT Excel Order
        # We use the columns directly from Excel as they contain dynamic values (Health Condition vs Fault)
        cols_to_keep = [
            't (s)', 'V(a) p.u', 'V(b) p.u', 'V(c) p.u', 
            'I(a) p.u', 'I(b) p.u', 'I(c) p.u', 
            'Detection', 'Fault Location (km)', 'Classification'
        ]
        
        # Ensure columns exist
        for col in cols_to_keep:
            if col not in df.columns:
                df[col] = 0.0

        all_data.append(df[cols_to_keep])

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved with 7 features (including time) to {output_file}")

if __name__ == "__main__":
    prepare_data()
