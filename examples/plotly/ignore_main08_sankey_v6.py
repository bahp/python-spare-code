import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.cm as cm
import re

# --- 1. Simulate Patient Data (Using your latest request) ---
antibiotics = [
    "amoxicillin", "co-trimoxazole", "meropenem", "metronidazole", "amikacin", "cefuroxime",
    "ciprofloxacin", "piperacillin-tazobactam", "vancomycin", "gentamicin", "flucloxacillin",
    "teicoplanin", "ceftriaxone", "temocillin", "fosfomycin", "clarithromycin", "aztreonam",
    "linezolid", "erythromycin", "cefepime", "ceftazidime", "pivmecillinam", "chloramphenicol",
    "tazocin", "co-amoxiclav", "doxycycline"
]
therapies = antibiotics + ['No Therapy']
np.random.seed(42)



def create_synthetic_data():
    """

    .. note:: Simulate different lengths of stay.

    """
    patient_ids = [np.random.randint(1000, 6000) for _ in range(50)]
    days = [1, 2, 3, 4]
    records = []
    base_date = datetime(2025, 9, 1)
    admission_dates = {pid: base_date + timedelta(days=np.random.randint(0, 30)) for pid in patient_ids}

    # Day 1
    for pid in patient_ids:
        admission_date = admission_dates[pid]
        therapy_name = np.random.choice(antibiotics)
        records.append({
            'SUBJECT': pid, 'ADMISSION_DATE': admission_date.strftime('%Y-%m-%d'),
            'NEWS2_SCORE': np.random.randint(0, 20), 'DAY_NUM': 1,
            'MEDICATION_NAME': therapy_name,
            'ORDER_DT_TM': admission_date + timedelta(days=1, hours=np.random.randint(0, 24)),
            'drug_norm': therapy_name
        })

    # Days 2, 3, 4
    for day_idx in range(1, len(days)):
        current_day = days[day_idx]
        previous_day = days[day_idx - 1]
        previous_day_df = pd.DataFrame(records)[pd.DataFrame(records)['DAY_NUM'] == previous_day]
        for _, row in previous_day_df.iterrows():
            pid, prev_therapy, adm_date_str = row['SUBJECT'], row['drug_norm'], row['ADMISSION_DATE']
            admission_date = datetime.strptime(adm_date_str, '%Y-%m-%d')
            if prev_therapy == 'No Therapy':
                new_therapy = np.random.choice(['No Therapy'] + antibiotics, p=[0.8] + [0.2 / len(antibiotics)] * len(antibiotics))
            else:
                other_abs = [ab for ab in antibiotics if ab != prev_therapy]
                prob_switch = 0.3 / len(other_abs) if other_abs else 0
                new_therapy = np.random.choice([prev_therapy, 'No Therapy'] + other_abs, p=[0.6, 0.1] + [prob_switch] * len(other_abs))
            records.append({
                'SUBJECT': pid, 'ADMISSION_DATE': adm_date_str, 'NEWS2_SCORE': np.random.randint(0, 20),
                'DAY_NUM': current_day, 'MEDICATION_NAME': new_therapy,
                'ORDER_DT_TM': admission_date + timedelta(days=current_day, hours=np.random.randint(0, 24)),
                'drug_norm': new_therapy
            })

    return pd.DataFrame(records)




# --- 2. Define Helper Functions ---

def limit_therapies_to_top_n(df: pd.DataFrame, therapy_col: str, n: int) -> pd.DataFrame:
    """Keeps the top N most frequent therapies and groups the rest into 'Other'."""
    # Calculate frequency and identify top N
    therapy_counts = df[therapy_col].value_counts()
    top_n_therapies = therapy_counts.head(n).index.tolist()

    # Rewrite
    df[therapy_col] = np.where(df[therapy_col].isin(top_n_therapies), df[therapy_col], 'Other')
    return df

def create_flow_from_patient_data(df: pd.DataFrame, subject_col: str, day_col: str, therapy_col: str) -> pd.DataFrame:
    df = df.drop_duplicates(subset=[subject_col, day_col, therapy_col])
    df = df.sort_values(by=[subject_col, day_col])
    df['item_from'] = df.groupby(subject_col)[therapy_col].shift(1)
    df['day_from'] = df.groupby(subject_col)[day_col].shift(1)
    df.rename(columns={day_col: 'day_to', therapy_col: 'item_to'}, inplace=True)
    flow_df = df.dropna(subset=['item_from', 'day_from'])
    flow_df['day_from'] = flow_df['day_from'].astype(int)
    flow_counts = flow_df.groupby(['day_from', 'item_from', 'day_to', 'item_to']).size().reset_index(name='count')
    return flow_counts

def create_comprehensive_color_map(df: pd.DataFrame) -> dict:
    """Generates a consistent color map using 'item_from' and 'item_to'."""
    therapies = sorted(list(set(np.concatenate([df['item_from'].unique(), df['item_to'].unique()]))))
    color_map = {}
    if 'No Therapy' in therapies: color_map['No Therapy'] = '#A3A3A3'; therapies.remove('No Therapy')
    if not therapies: return color_map
    colormap = cm.get_cmap('tab20b', len(therapies))
    hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, a in colormap(np.linspace(0, 1, len(therapies)))]
    for therapy, color in zip(therapies, hex_colors): color_map[therapy] = color
    return color_map



# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

# Lookup table to map antibiotic to access class
ACCESS_LIST = [
    "Amikacin", "Amoxicillin", "Amoxicillin (contains penicillin)",
    "Amoxicillin/clavulanic-acid", "Amoxiclavulanic-acid", "Benzylpenicillin sodium", "Co-amoxiclav",
    "Co-amoxiclav (contains penicillin)", "Co-amoxiclav (contains penicillin) (anes)",
    "Co-amoxiclav 400mg/57mg in 5ml oral suspension (contains penicillin)",
    "co-amoxiclav 250mg/62mg in 5ml oral suspension (contains penicillin)",
    "Cefalexin", "Cefazolin", "Chloramphenicol", "Clindamycin", "Doxycycline",
    "Flucloxacillin (contains penicillin)", "Gentamicin", "Gentamicin (anes)", "Mecillinam",
    "Metronidazole", "Metronidazole (anes)", "Metronidazole_IV", "Metronidazole_oral",
    "Nitrofurantoin", "Co-amoxiclav in suspension", "Pivmecillinam",
    "Pivmecillinam (contains penicillin)", "Sulfamethoxazole/trimethoprim", "Trimethoprim",
    "co-trimoxazole"
]
WATCH_LIST = [
    "Azithromycin", "Cefepime", "Ceftazidime", "Ceftriaxone", "Cefuroxime", "Cefuroxime (anes)",
    "Ciprofloxacin", "Clarithromycin", "Ertapenem", "Erythromycin", "Fosfomycin", "Fosfomycin_oral",
    "Levofloxacin", "Meropenem", "Moxifloxacin", "Piperacillin/tazobactam",
    "piperacillin-tazobactam (contains penicillin)", "piperacillin + tazobactam (contains penicillin)",
    "Teicoplanin", "Teicoplanin (anes)", "Temocillin", "Temocillin (contains penicillin)", "Vancomycin",
    "Vancomycin (anes)", "Vancomycin (anes) 1g", "Vancomycin_IV", "Vancomycin_oral"
]
RESERVE_LIST = [
    "Aztreonam", "Cefiderocol", "Ceftazidime/avibactam", "Ceftolozane/tazobactam", "Colistin",
    "Colistin_IV", "Dalbavancin", "Daptomycin", "Fosfomycin_IV", "Iclaprim", "Linezolid", "Tigecycline"
]

# This is more efficient for lookup than searching through lists every time.
# We convert all drug names to lowercase to ensure case-insensitive matching.
DRUG_TO_AWARE_CLASS = {}
for drug in ACCESS_LIST:
    DRUG_TO_AWARE_CLASS[drug.lower()] = "Access"
for drug in WATCH_LIST:
    DRUG_TO_AWARE_CLASS[drug.lower()] = "Watch"
for drug in RESERVE_LIST:
    DRUG_TO_AWARE_CLASS[drug.lower()] = "Reserve"

# Lookup table to map drug name to its broader therapeutic class.
DRUG_TO_CLASS = {
    "benzylpenicillin": "Penicillin",
    "flucloxacillin": "Penicillin",
    'temocillin': "Penicillin",
    "amoxicillin": "Penicillin",
    "piperacillin tazobactam": "Penicillin and beta-lactamase inhibitor",
    "pivmecillinam": "Penicillin",
    "co-amoxiclav": "Penicillin and beta-lactamase inhibitor",
    "cefalexin": "1st Generation Cephalosporin",
    "cefazolin": "1st Generation Cephalosporin",
    "cefuroxime": "2nd Generation Cephalosporin",
    "ceftriaxone": "3rd Generation Cephalosporin",
    "ceftazidime": "3rd Generation Cephalosporin",
    "cefepime": "4th Generation Cephalosporin",
    'cefiderocol': "5th Generation Cephalosporin",
    "clindamycin": "Lincomycin",
    "meropenem": "Carbapenem",
    "ertapenem": "Carbapenem",
    "aztreonam": "Monobactam",
    "gentamicin": "Aminoglycoside",
    "amikacin": "Aminoglycoside",
    "ciprofloxacin": "Fluoroquinolone",
    "levofloxacin": "Fluoroquinolone",
    "moxifloxacin": "Fluoroquinolone",
    "clarithromycin": "Macrolide",
    "erythromycin": "Macrolide",
    "azithromycin": "Macrolide",
    "vancomycin": "Glycopeptide",
    "teicoplanin": "Glycopeptide",
    "linezolid": "Oxazolidinone",
    "tigecycline": "Glycylcycline",
    "daptomycin": "Lipopeptide",
    "metronidazole": "Nitroimidazole",
    "doxycycline": "Tetracycline",
    "nitrofurantoin": "Nitrofuran",
    "fosfomycin": "Phosphonic acid",
    "co trimoxazole": "Sulfonamides and trimethoprim",
    "trimethoprim": "Sulfonamides and trimethoprim",
    "chloramphenicol": "Amphenicol"
}


# Configure
TOP_N_THERAPIES = 5
ANALYSIS_LEVEL = 'class'
COLUMN_FOR_ANALYSIS = {
    'class': 'drug_class',
    'aware': 'aware_class',
    'drug': 'drug_norm'
}

# Select the column to use based on the chosen level
therapy_col = COLUMN_FOR_ANALYSIS[ANALYSIS_LEVEL]

# Create synthetic data
patient_df = create_synthetic_data()
print("\n\n ---> Patient data:\n %s" % patient_df)

# Group by antibiotic class
if ANALYSIS_LEVEL == 'class':
    patient_df[therapy_col] = patient_df['drug_norm'] \
        .map(DRUG_TO_CLASS).fillna('Unknown')
elif ANALYSIS_LEVEL == 'aware':
    patient_df[therapy_col] = patient_df['drug_norm'].str.lower()\
        .map(DRUG_TO_AWARE_CLASS).fillna('Unknown')

# Filter top N therapies
if TOP_N_THERAPIES is not None:
    patient_df = limit_therapies_to_top_n(df=patient_df,
        therapy_col=therapy_col, n=TOP_N_THERAPIES)

# Create the flow data.
flow_df = create_flow_from_patient_data(patient_df.copy(),
    subject_col='SUBJECT', day_col='DAY_NUM', therapy_col=therapy_col)

# Create the color map NOW, while 'item_from' and 'item_to' still exist.
therapy_colors = create_comprehensive_color_map(flow_df)

# Now, prepare the DataFrame for plotting by adding/renaming columns.
flow_df['Source'] = flow_df['item_from'] + ' (Day ' + flow_df['day_from'].astype(str) + ')'
flow_df['Target'] = flow_df['item_to'] + ' (Day ' + flow_df['day_to'].astype(str) + ')'
flow_df.rename(columns={'count': 'Value', 'item_from': 'Source_Therapy'}, inplace=True)
final_flow_df = flow_df

# Create the sorted node list and map for correct plotting order
all_nodes_unsorted = pd.concat([final_flow_df['Source'], final_flow_df['Target']]).unique()
get_day = lambda label: int(re.search(r'\(Day (\d+)\)', label).group(1))
all_nodes = sorted(all_nodes_unsorted, key=lambda x: (get_day(x), x))
node_map = {node: i for i, node in enumerate(all_nodes)}
final_flow_df['Source_ID'] = final_flow_df['Source'].map(node_map)
final_flow_df['Target_ID'] = final_flow_df['Target'].map(node_map)

# Generate node positions and link colors
unique_days = sorted(np.unique(np.concatenate([flow_df['day_from'], flow_df['day_to']])))
day_x_map = {day: i / (len(unique_days) - 1) if len(unique_days) > 1 else 0.5 for i, day in enumerate(unique_days)}
node_x = [day_x_map.get(get_day(label), 0) for label in all_nodes]
node_colors = [therapy_colors.get(next((t for t in therapy_colors if t in n), None), '#A3A3A3') for n in all_nodes]
link_colors = [f"rgba({int(therapy_colors.get(r['Source_Therapy'], '#A3A3A3')[1:3], 16)}," \
               f"{int(therapy_colors.get(r['Source_Therapy'], '#A3A3A3')[3:5], 16)}," \
               f"{int(therapy_colors.get(r['Source_Therapy'], '#A3A3A3')[5:7], 16)},0.4)"
               for _, r in final_flow_df.iterrows()]

# F. Create the Sankey Diagram
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(pad=20, thickness=25,
              line=dict(color="black", width=0.5),
              label=all_nodes, color=node_colors, x=node_x),
    link=dict(source=final_flow_df['Source_ID'],
              target=final_flow_df['Target_ID'],
              value=final_flow_df['Value'],
              color=link_colors)
)])
fig.update_layout(title_text="<b>Patient Antimicrobial Therapy Transitions</b>", font=dict(size=14))
fig.show()
