import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.cm as cm


# -----------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------

def create_synthetic_sample1():
    """
    Creates a small, specific dataset designed to reproduce the backward link error.
    - Patient 1 is a control with a simple forward flow.
    - Patient 2 is the problem case, transitioning to a rare drug on Day 3.
    """
    records = [
        # Patient 1 (Control)
        {'SUBJECT': 1, 'DAY_NUM': 1, 'drug_norm': 'Amoxicillin'},
        {'SUBJECT': 1, 'DAY_NUM': 2, 'drug_norm': 'Ciprofloxacin'},
        {'SUBJECT': 1, 'DAY_NUM': 3, 'drug_norm': 'Vancomycin'},

        # Patient 2 (Problem Case)
        {'SUBJECT': 2, 'DAY_NUM': 1, 'drug_norm': 'Amoxicillin'},
        {'SUBJECT': 2, 'DAY_NUM': 2, 'drug_norm': 'Penicillin'},
        {'SUBJECT': 2, 'DAY_NUM': 3, 'drug_norm': 'Doxycycline'},  # This will become 'Other'

        # More data to establish top therapies
        {'SUBJECT': 3, 'DAY_NUM': 1, 'drug_norm': 'Ciprofloxacin'},
        {'SUBJECT': 3, 'DAY_NUM': 2, 'drug_norm': 'Penicillin'},
        {'SUBJECT': 4, 'DAY_NUM': 1, 'drug_norm': 'Vancomycin'},
    ]
    return pd.DataFrame(records)


def create_synthetic_sample2():
    pass

def create_synthetic_data(num_patients=50, max_los=7):
    """
    Creates a DataFrame of synthetic patient antibiotic data with variable lengths of stay.

    Args:
        num_patients: The number of patients to simulate.
        max_los: The maximum possible length of stay for any patient.
    """
    antibiotics = [
        "amoxicillin", "co-trimoxazole", "meropenem", "metronidazole", "amikacin", "cefuroxime",
        "ciprofloxacin", "piperacillin-tazobactam", "vancomycin", "gentamicin", "flucloxacillin",
        "teicoplanin", "ceftriaxone", "temocillin", "fosfomycin", "clarithromycin", "aztreonam",
        "linezolid", "erythromycin", "cefepime", "ceftazidime", "pivmecillinam", "chloramphenicol",
        "tazocin", "co-amoxiclav", "doxycycline"
    ]
    patient_ids = [i for i in range(num_patients)]
    records = []

    # Assign a random length of stay (LoS) to each patient
    # Ensure a minimum LoS of 2 days to allow for at least one transition
    patient_lengths_of_stay = {pid: np.random.randint(2, max_los + 1) for pid in patient_ids}

    # Simulate records patient by patient
    for pid in patient_ids:
        los = patient_lengths_of_stay[pid]

        # Day 1: Assign initial therapy
        current_therapy = np.random.choice(antibiotics)
        records.append({'SUBJECT': pid, 'DAY_NUM': 1, 'drug_norm': current_therapy})

        # Days 2 through LoS: Simulate transitions
        for day in range(2, los + 1):
            if current_therapy == 'No Therapy':
                # High chance of staying off therapy
                new_therapy = np.random.choice(
                    ['No Therapy'] + antibiotics,
                    p=[0.8] + [0.2 / len(antibiotics)] * len(antibiotics)
                )
            else:
                # Logic for patients on an existing therapy
                other_abs = [ab for ab in antibiotics if ab != current_therapy]
                prob_switch = 0.3 / len(other_abs) if other_abs else 0
                new_therapy = np.random.choice(
                    [current_therapy, 'No Therapy'] + other_abs,
                    p=[0.6, 0.1] + [prob_switch] * len(other_abs)
                )

            records.append({'SUBJECT': pid, 'DAY_NUM': day, 'drug_norm': new_therapy})
            current_therapy = new_therapy  # Update for the next day's transition

    # Add some combined therapies for specific patients to test other logic
    # This will overwrite their original Day 2 record
    for pid in patient_ids[:5]:  # Add for the first 5 patients
        # Ensure these patients have at least 2 days of data before adding a combination
        if patient_lengths_of_stay[pid] >= 2:
            records.append({'SUBJECT': pid, 'DAY_NUM': 2, 'drug_norm': 'amoxicillin'})
            records.append({'SUBJECT': pid, 'DAY_NUM': 2, 'drug_norm': 'linezolid'})

    return pd.DataFrame(records)


def handle_aware_combinations(df: pd.DataFrame,
                              subject_col: str,
                              day_col: str,
                              aware_col: str) -> pd.DataFrame:
    """Resolves combined therapies by choosing the most restrictive AWaRe class."""
    aware_ordinal_map = {'Access': 0, 'Watch': 1, 'Reserve': 2, 'Unknown': -1}
    df_copy = df.copy()
    df_copy['aware_ordinal'] = df_copy[aware_col].map(aware_ordinal_map)
    df_copy['max_aware_ordinal'] = df_copy.groupby([subject_col, day_col])['aware_ordinal'].transform('max')
    df_copy = df_copy.drop_duplicates(subset=[subject_col, day_col])
    ordinal_to_aware_map = {v: k for k, v in aware_ordinal_map.items()}
    df_copy[aware_col] = df_copy['max_aware_ordinal'].map(ordinal_to_aware_map)
    return df_copy.drop(columns=['aware_ordinal', 'max_aware_ordinal'])


def aggregate_drug_combinations(df: pd.DataFrame,
                                subject_col: str,
                                day_col: str,
                                therapy_col: str) -> pd.DataFrame:
    """
    Resolves combined therapies by creating a sorted, comma-separated string of drug names.
    """
    # First, remove any exact duplicate rows for a patient on the same day
    df_unique = df.drop_duplicates(subset=[subject_col, day_col, therapy_col])

    # Group by patient and day, then aggregate the unique drug names
    aggregated_df = df_unique.groupby([subject_col, day_col])[therapy_col].apply(
        lambda drugs: ', '.join(sorted(drugs.unique()))
    ).reset_index()

    return aggregated_df

def limit_therapies_to_top_n(df: pd.DataFrame, therapy_col: str, n: int) -> pd.DataFrame:
    """Keeps the top N most frequent therapies and groups the rest into 'Other'."""
    if n is None:
        return df
    df_copy = df.copy()
    therapy_counts = df_copy[therapy_col].value_counts()
    top_n_therapies = therapy_counts.head(n).index.tolist()
    df_copy[therapy_col] = np.where(df_copy[therapy_col].isin(top_n_therapies), df_copy[therapy_col], 'Other')
    return df_copy


def create_flow_from_patient_data(df: pd.DataFrame,
                                  subject_col: str,
                                  day_col: str,
                                  therapy_col: str) -> pd.DataFrame:
    """Transforms raw patient data into a Sankey-ready flow DataFrame."""
    """
    df = df.drop_duplicates(subset=[subject_col, day_col, therapy_col]).sort_values(by=[subject_col, day_col])
    df['item_from'] = df.groupby(subject_col)[therapy_col].shift(1)
    df['day_from'] = df.groupby(subject_col)[day_col].shift(1)
    df.rename(columns={day_col: 'day_to', therapy_col: 'item_to'}, inplace=True)
    flow_df = df.dropna(subset=['item_from', 'day_from'])
    flow_df['day_from'] = flow_df['day_from'].astype(int)
    return flow_df.groupby(['day_from', 'item_from', 'day_to', 'item_to']).size().reset_index(name='count')
    """
    # This function is not needed if you start with flow data, but is included for completeness.
    # It assumes columns: 'SUBJECT', 'DAY_NUM', 'drug_norm'
    df = df.sort_values(by=['SUBJECT', 'DAY_NUM'])
    df['item_from'] = df.groupby('SUBJECT')['drug_norm'].shift(1)
    df['day_from'] = df.groupby('SUBJECT')['DAY_NUM'].shift(1)
    df.rename(columns={'DAY_NUM': 'day_to', 'drug_norm': 'item_to'}, inplace=True)
    flow_df = df.dropna(subset=['item_from', 'day_from'])
    flow_df['day_from'] = flow_df['day_from'].astype(int)
    return flow_df.groupby(['day_from', 'item_from', 'day_to', 'item_to']).size().reset_index(name='count')


def create_comprehensive_color_map(df: pd.DataFrame) -> dict:
    """Generates a consistent color map, including special colors for specified categories."""
    therapies = sorted(list(set(np.concatenate([df['item_from'].unique(), df['item_to'].unique()]))))
    color_map = {}
    special_colors = {'No Therapy': '#A3A3A3', 'Other': '#D3D3D3', 'Unknown': '#E5E5E5'}
    for category, color in special_colors.items():
        if category in therapies:
            color_map[category] = color
            therapies.remove(category)
    if not therapies: return color_map
    colormap = cm.get_cmap('tab20b', len(therapies))
    hex_colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
                  for r, g, b, a in colormap(np.linspace(0, 1, len(therapies)))]
    for therapy, color in zip(therapies, hex_colors): color_map[therapy] = color
    return color_map


def plot_sankey(flow_df: pd.DataFrame, therapy_colors: dict, title: str):
    """
    Generates and displays a Sankey diagram from a flow DataFrame.
    This function uses a structural approach and does not rely on string parsing.
    """
    # 1. Create a structured DataFrame of all unique nodes
    source_nodes = flow_df[['day_from', 'item_from']].rename(columns={'day_from': 'day', 'item_from': 'label'})
    target_nodes = flow_df[['day_to', 'item_to']].rename(columns={'day_to': 'day', 'item_to': 'label'})
    nodes_df = pd.concat([source_nodes, target_nodes]).drop_duplicates().sort_values(['day', 'label']).reset_index(
        drop=True)
    nodes_df['id'] = nodes_df.index

    # 2. Map flow data to the unique node IDs
    node_map = pd.Series(nodes_df.id.values, index=nodes_df['label'] + '_day_' + nodes_df['day'].astype(str)).to_dict()
    flow_df['source_id'] = (flow_df['item_from'] + '_day_' + flow_df['day_from'].astype(str)).map(node_map)
    flow_df['target_id'] = (flow_df['item_to'] + '_day_' + flow_df['day_to'].astype(str)).map(node_map)

    # 3. Generate plot properties from the structured nodes_df
    unique_days = sorted(nodes_df['day'].unique())
    day_x_map = {day: i / (len(unique_days) - 1) if len(unique_days) > 1 else 0.5 for i, day in enumerate(unique_days)}

    node_labels = nodes_df['label'].tolist()
    node_x = nodes_df['day'].map(day_x_map).tolist()
    node_colors = nodes_df['label'].map(therapy_colors).fillna('#A3A3A3').tolist()
    link_colors = flow_df['item_from'].map(therapy_colors).fillna('#A3A3A3').apply(
        lambda hex: f"rgba({int(hex[1:3], 16)},{int(hex[3:5], 16)},{int(hex[5:7], 16)},0.4)"
    ).tolist()

    # 4. Create the figure
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=25, line=dict(color="black", width=0.5),
                  label=node_labels, color=node_colors, x=node_x),
        link=dict(source=flow_df['source_id'], target=flow_df['target_id'],
                  value=flow_df['count'], color=link_colors)
    )])

    # Add x-axis and other tweaks
    annotations = []
    for day, x_pos in day_x_map.items():
        annotations.append(
            dict(
                x=x_pos,  # Position on the x-axis (0 to 1)
                y=-0.07,  # Position on the y-axis (below the plot)
                text=f"<b>Day {day}</b>",  # The label text
                showarrow=False,  # No arrow
                font=dict(size=16, color="black"),
                xref="paper",  # Use 'paper' coordinates for positioning
                yref="paper",
                xanchor="center",  # Center the text on the x-position
                yanchor="top"
            )
        )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        font=dict(size=12),
        plot_bgcolor='white',
        annotations=annotations,  # Add the created annotations to the layout
        margin=dict(b=100)  # Increase bottom margin to make space for labels
    )


    fig.show()


def plot_sankey_robust(flow_df: pd.DataFrame, therapy_colors: dict, title: str):
    """
    Generates a Sankey diagram using a structural method that prevents backward links.
    """

    # 4. Create a structured DataFrame of all unique nodes from the flow data
    source_nodes = flow_df[['day_from', 'item_from']].rename(columns={'day_from': 'day', 'item_from': 'label'})
    target_nodes = flow_df[['day_to', 'item_to']].rename(columns={'day_to': 'day', 'item_to': 'label'})

    # 5. Combine, drop duplicates, and sort STRUCTURALLY (by day, then by label)
    nodes_df = pd.concat([source_nodes, target_nodes]) \
        .drop_duplicates() \
        .sort_values(['day', 'label']) \
        .reset_index(drop=True)

    # 6. The index of this sorted DataFrame is now the unique, guaranteed-correct ID
    nodes_df['id'] = nodes_df.index

    # 7. Map the flows to these new, reliable IDs
    node_map = pd.Series(nodes_df.id.values, index=nodes_df['label'] + '_day_' + nodes_df['day'].astype(str)).to_dict()
    flow_df['source_id'] = (flow_df['item_from'] + '_day_' + flow_df['day_from'].astype(str)).map(node_map)
    flow_df['target_id'] = (flow_df['item_to'] + '_day_' + flow_df['day_to'].astype(str)).map(node_map)

    # 8. Generate all properties needed for the plot from the structured nodes_df
    unique_days = sorted(nodes_df['day'].unique())
    day_x_map = {day: i / (len(unique_days) - 1) if len(unique_days) > 1 else 0.5 for i, day in enumerate(unique_days)}

    node_labels = nodes_df['label'].tolist()
    node_x = nodes_df['day'].map(day_x_map).tolist()
    node_colors = nodes_df['label'].map(therapy_colors).fillna('#A3A3A3').tolist()
    link_colors = flow_df['item_from'].map(therapy_colors).fillna('#A3A3A3').apply(
        lambda hex_val: f"rgba({int(hex_val[1:3], 16)},{int(hex_val[3:5], 16)},{int(hex_val[5:7], 16)},0.4)"
    ).tolist()

    # 9. Create the Figure with the corrected data
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=25, line=dict(color="black", width=0.5), label=node_labels, color=node_colors,
                  x=node_x),
        link=dict(source=flow_df['source_id'], target=flow_df['target_id'], value=flow_df['count'], color=link_colors)
    )])

    # Add day labels as annotations
    annotations = [dict(x=x_pos, y=-0.07, text=f"<b>Day {day}</b>", showarrow=False, font=dict(size=16), xref="paper",
                        yref="paper", xanchor="center") for day, x_pos in day_x_map.items()]
    fig.update_layout(
        title_text=f"<b>Patient Antimicrobial Therapy Transitions ({ANALYSIS_LEVEL.title()} View)</b>",
        font=dict(size=12),
        annotations=annotations,
        margin=dict(b=100)
    )
    fig.show()

# --- Main Execution ---

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

# Define lookup table as this is more efficient for lookup than searching
# through lists every time. We convert all drug names to lowercase to
# ensure case-insensitive matching.
DRUG_TO_AWARE_CLASS = {drug.lower(): "Access" for drug in ACCESS_LIST}
DRUG_TO_AWARE_CLASS.update({drug.lower(): "Watch" for drug in WATCH_LIST})
DRUG_TO_AWARE_CLASS.update({drug.lower(): "Reserve" for drug in RESERVE_LIST})

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







# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
# Configuration
N_PATIENTS = 50           # Number of synthetic patients
TOP_N_THERAPIES = 5      # Keep top N therapies + others
ANALYSIS_LEVEL = 'drug'  # Options: 'class', 'aware', 'drug'

# .. note:: We rely on the dictionary access below to validate the
#           analysis_level. An invalid key will correctly cause a
#           KeyError, which is the intended behavior
COLUMN_FOR_ANALYSIS = {
    'class': 'drug_class',
    'aware': 'aware_class',
    'drug': 'drug_norm'
}

therapy_col = COLUMN_FOR_ANALYSIS[ANALYSIS_LEVEL]
subject_col = 'SUBJECT'
day_col = 'DAY_NUM'


# Load data
#patient_df = create_synthetic_data(num_patients=N_PATIENTS)
#patient_df = create_synthetic_sample1()

#patient_df.to_csv('./objects/plot_main08_sankey/data1.csv')
patient_df = pd.read_csv('./objects/plot_main08_sankey/data1.csv')

# Apply classification based on analysis level
if ANALYSIS_LEVEL == 'class':
    patient_df[therapy_col] = patient_df['drug_norm'] \
        .str.lower().map(DRUG_TO_CLASS).fillna('Unknown')

elif ANALYSIS_LEVEL == 'aware':
    patient_df[therapy_col] = patient_df['drug_norm'] \
        .str.lower().map(DRUG_TO_AWARE_CLASS).fillna('Unknown')
    patient_df = handle_aware_combinations(df=patient_df,
        subject_col=subject_col,
        day_col=day_col,
        aware_col=therapy_col)

elif ANALYSIS_LEVEL == 'drug':
    patient_df = aggregate_drug_combinations(df=patient_df,
        subject_col=subject_col,
        day_col=day_col,
        therapy_col=therapy_col)
else:
    print("--> Analysis type <{ANALYSIS_LEVEL}> not supported.") # never runs


# .. note:: To ensure the logic is correct, you can add a test case. Create
#           and filter synthetic patient with a specific therapy progression
#           over several days. Then, run the script and confirm that the final
#           plot accurately reflects this known pathway.

#patient_df = patient_df[patient_df['SUBJECT'] == 1]

# Apply "Top N" filter
#patient_df_processed = limit_therapies_to_top_n(
#    df=patient_df, therapy_col=therapy_col, n=TOP_N_THERAPIES
#)
patient_df_processed = patient_df

# .. note:: Create a custom color scheme to make information easier to understand.
#           For example, you could use green, yellow, and red for statuses like
#           'access,' 'watch,' and 'reserve.' You can also assign a unique color
#           to each drug class and then use different shades of that color for
#           the individual drugs within the class. For help with this, you can
#           ask Gemini to generate the color palettes for you

# Create Flow and Color Map
flow_df = create_flow_from_patient_data(patient_df_processed,
    subject_col=subject_col, day_col=day_col, therapy_col=therapy_col)
flow_df.to_csv('./objects/plot_main08_sankey/flow1.csv')
therapy_colors = create_comprehensive_color_map(flow_df)

print(flow_df[flow_df.day_from.isin([4, 5])])

# 5. Generate the Plot
plot_title = f"Patient antimicrobial therapy transitions ({ANALYSIS_LEVEL.title()} View)"
plot_sankey_robust(flow_df, therapy_colors, title=plot_title)
