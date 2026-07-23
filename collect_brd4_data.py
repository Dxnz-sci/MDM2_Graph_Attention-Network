"""
BRD4 Data Collection
=====================
Fetches BRD4 (Bromodomain-containing protein 4, CHEMBL1163125) bioactivity
data from ChEMBL, following the same cleaning convention as mdm2_cleaned.csv
so the existing GAT pipeline (data_preparation.py, model.py, train.py) can
be reused unchanged on a second target.

Target: BRD4 (CHEMBL1163125) - epigenetic reader domain, cancer/inflammation
"""

import os
import pandas as pd
from chembl_webresource_client.new_client import new_client

SAVE_PATH = 'data'
os.makedirs(SAVE_PATH, exist_ok=True)


def collect_data():
    print("Collecting BRD4 data from ChEMBL...")

    activity = new_client.activity
    brd4_activity = activity.filter(
        target_chembl_id='CHEMBL1163125',
        standard_type='IC50'
    )

    df = pd.DataFrame.from_records(brd4_activity)
    print(f"Raw compounds retrieved: {len(df)}")

    df = df[['molecule_chembl_id', 'canonical_smiles',
              'standard_value', 'standard_units', 'pchembl_value']]

    df = df[df['standard_units'] == 'nM']
    df = df.dropna(subset=['canonical_smiles', 'standard_value', 'pchembl_value'])
    df = df.drop_duplicates(subset=['molecule_chembl_id'])
    df = df.reset_index(drop=True)
    df['pchembl_value'] = df['pchembl_value'].astype(float)

    print(f"Clean compounds: {len(df)}")
    save_file = os.path.join(SAVE_PATH, 'brd4_cleaned.csv')
    df.to_csv(save_file, index=False)
    print(f"Saved to {save_file}")

    return df


if __name__ == "__main__":
    collect_data()
