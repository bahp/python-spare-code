import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm

# =============================================================================
# LOOKUP TABLES & CONSTANTS
# =============================================================================

def drug_to_aware_default():
    """Create default drug to aware classes."""
    return {
        "amikacin": "Access", "amoxicillin": "Access", "co-amoxiclav": "Access",
        "benzylpenicillin sodium": "Access", "cefalexin": "Access", "cefazolin": "Access",
        "chloramphenicol": "Access", "clindamycin": "Access", "doxycycline": "Access",
        "flucloxacillin": "Access", "gentamicin": "Access", "mecillinam": "Access",
        "metronidazole": "Access", "nitrofurantoin": "Access", "pivmecillinam": "Access",
        "co-trimoxazole": "Access", "trimethoprim": "Access", "azithromycin": "Watch",
        "cefepime": "Watch", "ceftazidime": "Watch", "ceftriaxone": "Watch",
        "cefuroxime": "Watch", "ciprofloxacin": "Watch", "clarithromycin": "Watch",
        "ertapenem": "Watch", "erythromycin": "Watch", "fosfomycin": "Watch",
        "levofloxacin": "Watch", "meropenem": "Watch", "moxifloxacin": "Watch",
        "piperacillin-tazobactam": "Watch", "teicoplanin": "Watch", "temocillin": "Watch",
        "vancomycin": "Watch", "aztreonam": "Reserve", "cefiderocol": "Reserve",
        "ceftazidime/avibactam": "Reserve", "ceftolozane/tazobactam": "Reserve",
        "colistin": "Reserve", "daptomycin": "Reserve", "linezolid": "Reserve",
        "tazocin": "Watch"
    }

def drug_to_aware_class_winnie():
    """Create drug to aware class as defined by winnie"""
    ACCESS_LIST = [
        "Amikacin", "Amoxicillin", "Amoxicillin (contains penicillin)",
        "Amoxicillin/clavulanic-acid", "Amoxiclavulanic-acid",
        "Benzylpenicillin sodium", "Co-amoxiclav", "Co-amoxiclav (contains penicillin)",
        "Co-amoxiclav (contains penicillin) (anes)",
        "Co-amoxiclav 400mg/57mg in 5ml oral suspension (contains penicillin)",
        "co-amoxiclav 250mg/62mg in 5ml oral suspension (contains penicillin)",
        "Cefalexin", "Cefazolin", "Chloramphenicol", "Clindamycin", "Doxycycline",
        "Flucloxacillin (contains penicillin)", "Gentamicin", "Gentamicin (anes)",
        "Mecillinam", "Metronidazole", "Metronidazole (anes)", "Metronidazole_IV",
        "Metronidazole_oral", "Nitrofurantoin", "Co-amoxiclav in suspension",
        "Pivmecillinam", "Pivmecillinam (contains penicillin)",
        "Sulfamethoxazole/trimethoprim", "Trimethoprim", "co-trimoxazole"
    ]
    WATCH_LIST = [
        "Azithromycin", "Cefepime", "Ceftazidime", "Ceftriaxone", "Cefuroxime",
        "Cefuroxime (anes)", "Ciprofloxacin", "Clarithromycin", "Ertapenem",
        "Erythromycin", "Fosfomycin", "Fosfomycin_oral", "Levofloxacin", "Meropenem",
        "Moxifloxacin", "Piperacillin/tazobactam", "piperacillin-tazobactam (contains penicillin)",
        "piperacillin + tazobactam (contains penicillin)", "Teicoplanin", "Teicoplanin (anes)",
        "Temocillin", "Temocillin (contains penicillin)", "Vancomycin",
        "Vancomycin (anes)", "Vancomycin (anes) 1g", "Vancomycin_IV", "Vancomycin_oral"
    ]
    RESERVE_LIST = [
        "Aztreonam", "Cefiderocol", "Ceftazidime/avibactam", "Ceftolozane/tazobactam",
        "Colistin", "Colistin_IV", "Dalbavancin", "Daptomycin", "Fosfomycin_IV",
        "Iclaprim", "Linezolid", "Tigecycline"
    ]

    # Define lookup table as this is more efficient for lookup than searching
    # through lists every time. We convert all drug names to lowercase to
    # ensure case-insensitive matching.
    d = {drug.lower(): "Access" for drug in ACCESS_LIST}
    d.update({drug.lower(): "Watch" for drug in WATCH_LIST})
    d.update({drug.lower(): "Reserve" for drug in RESERVE_LIST})
    return d

DRUG_TO_AWARE_CLASS = drug_to_aware_default()

DRUG_TO_CLASS = {
    "amoxicillin": "Penicillin", "flucloxacillin": "Penicillin", "pivmecillinam": "Penicillin",
    "co-amoxiclav": "Penicillin and beta-lactamase inhibitor", "tazocin": "Penicillin and beta-lactamase inhibitor",
    "piperacillin-tazobactam": "Penicillin and beta-lactamase inhibitor", "cefuroxime": "2nd Gen Cephalosporin",
    "ceftriaxone": "3rd Gen Cephalosporin", "ceftazidime": "3rd Gen Cephalosporin",
    "cefepime": "4th Gen Cephalosporin", "meropenem": "Carbapenem", "aztreonam": "Monobactam",
    "gentamicin": "Aminoglycoside", "amikacin": "Aminoglycoside", "ciprofloxacin": "Fluoroquinolone",
    "clarithromycin": "Macrolide", "erythromycin": "Macrolide", "vancomycin": "Glycopeptide",
    "teicoplanin": "Glycopeptide", "linezolid": "Oxazolidinone", "metronidazole": "Nitroimidazole",
    "doxycycline": "Tetracycline", "fosfomycin": "Phosphonic acid", "co-trimoxazole": "Folate pathway inhibitor",
    "chloramphenicol": "Amphenicol", "temocillin": "Penicillin"
}

COLUMN_FOR_ANALYSIS = {
    'drug': 'drug_norm',
    'class': 'drug_class',
    'aware': 'aware_class'
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_validation_data():
    """
    Creates a small, specific dataset with two patients to visually validate Sankey logic.
    - Tests single therapy -> combination therapy transition.
    - Tests merging flows from different sources into one combination node.
    - Tests combination therapy -> single therapy transition.
    - Tests transitions to 'No Therapy' and terminating journeys.
    """
    records = [
        # --- Patient 101: Longer journey with a combo ---
        {'SUBJECT': 101, 'DAY_NUM': 1, 'drug_norm': 'amoxicillin'},
        # Day 2 is a combination therapy
        {'SUBJECT': 101, 'DAY_NUM': 2, 'drug_norm': 'amoxicillin'},
        {'SUBJECT': 101, 'DAY_NUM': 2, 'drug_norm': 'linezolid'},
        # Day 3 transitions out of the combo
        {'SUBJECT': 101, 'DAY_NUM': 3, 'drug_norm': 'vancomycin'},
        {'SUBJECT': 101, 'DAY_NUM': 4, 'drug_norm': 'No Therapy'},
        {'SUBJECT': 101, 'DAY_NUM': 5, 'drug_norm': 'No Therapy'},

        # --- Patient 102: Shorter journey, merges into the combo on Day 2 ---
        {'SUBJECT': 102, 'DAY_NUM': 1, 'drug_norm': 'ciprofloxacin'},
        # Day 2 is the same combination therapy as Patient 101
        {'SUBJECT': 102, 'DAY_NUM': 2, 'drug_norm': 'amoxicillin'},
        {'SUBJECT': 102, 'DAY_NUM': 2, 'drug_norm': 'linezolid'},
        # Day 3 transitions to the same drug, but journey ends here
        {'SUBJECT': 102, 'DAY_NUM': 3, 'drug_norm': 'vancomycin'},
    ]
    return pd.DataFrame(records)

def create_synthetic_data(num_patients=50, max_los=7):
    """
    Creates synthetic patient antibiotic data with variable lengths of stay.

    Args:
        num_patients: The number of patients to simulate.
        max_los: The maximum possible length of stay for any patient.
    """
    antibiotics = list(DRUG_TO_AWARE_CLASS.keys()) + ['No Therapy']
    records = []
    for pid in range(num_patients):
        los = np.random.randint(2, max_los + 1)
        current_therapy = np.random.choice(antibiotics)
        records.append({'SUBJECT': pid, 'DAY_NUM': 1, 'drug_norm': current_therapy})
        for day in range(2, los + 1):
            if current_therapy == 'No Therapy':
                new_therapy = np.random.choice(antibiotics,
                    p=[0.8] + [0.2 / (len(antibiotics)-1)] * (len(antibiotics)-1))
            else:
                other_abs = [ab for ab in antibiotics if ab != current_therapy]
                new_therapy = np.random.choice([current_therapy, 'No Therapy'] + other_abs,
                    p=[0.6, 0.1] + [0.3/len(other_abs)]*len(other_abs))
            records.append({'SUBJECT': pid, 'DAY_NUM': day, 'drug_norm': new_therapy})
            current_therapy = new_therapy
    for pid in range(5): # Add combination therapies for testing
        records.append({'SUBJECT': pid, 'DAY_NUM': 2, 'drug_norm': 'amoxicillin'})
        records.append({'SUBJECT': pid, 'DAY_NUM': 2, 'drug_norm': 'linezolid'})
    return pd.DataFrame(records)


def process_patient_data(df: pd.DataFrame,
                         level: str,
                         therapy_col: str,
                         subject_col: str,
                         day_col: str) -> pd.DataFrame:
    """
    Applies classifications and correctly aggregates drug combinations for all
    analysis levels to make the data "pivot-ready".
    """
    df_copy = df.copy()

    if level == 'class':
        df_copy[therapy_col] = df_copy['drug_norm'].str.lower() \
            .map(DRUG_TO_CLASS).fillna('Unknown')
        # Aggregate classes for combination therapies into a single sorted string
        return df_copy.groupby([subject_col, day_col])[therapy_col] \
            .apply(lambda x: ' + '.join(sorted(x.unique()))) \
            .reset_index()

    elif level == 'aware':
        df_copy[therapy_col] = df_copy['drug_norm'].str.lower() \
            .map(DRUG_TO_AWARE_CLASS).fillna('Unknown')
        # For combinations, resolve by picking the highest-order class
        aware_order = {'Access': 0, 'Watch': 1, 'Reserve': 2, 'Unknown': -1}
        df_copy['aware_ordinal'] = df_copy[therapy_col].map(aware_order)
        idx = df_copy.groupby([subject_col, day_col])['aware_ordinal'].idxmax()
        return df_copy.loc[idx].drop(columns='aware_ordinal')

    elif level == 'drug':
        # Aggregates combination drugs into a single sorted string
        return df_copy.groupby([subject_col, day_col])[therapy_col]\
            .apply(lambda x: ' + '.join(sorted(x.unique()))) \
            .reset_index()

    return df_copy


def limit_therapies_to_top_n(df: pd.DataFrame, therapy_col: str, n: int) -> pd.DataFrame:
    """Keeps the top N most frequent therapies and groups the rest into 'Other'."""
    if n is None: return df
    top_n = df[therapy_col].value_counts().nlargest(n).index
    df[therapy_col] = df[therapy_col].where(df[therapy_col].isin(top_n), 'Other')
    return df


def create_flow_from_patient_data_pivot(df: pd.DataFrame,
                                        therapy_col: str,
                                        subject_col: str,
                                        day_col: str) -> pd.DataFrame:
    """Transforms raw patient data into a Sankey-ready flow DataFrame."""
    patient_journeys = df.pivot(index=subject_col, columns=day_col, values=therapy_col)
    all_links = []
    for i in range(1, patient_journeys.columns.max()):
        day_from, day_to = i, i + 1
        if day_from in patient_journeys.columns and day_to in patient_journeys.columns:
            transition_df = patient_journeys[[day_from, day_to]].dropna()
            links = transition_df.value_counts().reset_index(name='count')
            links.rename(columns={day_from: 'item_from', day_to: 'item_to'}, inplace=True)
            links['day_from'], links['day_to'] = day_from, day_to
            all_links.append(links)
    if not all_links:
        return pd.DataFrame(columns=['day_from', 'item_from', 'day_to', 'item_to', 'count'])
    return pd.concat(all_links, ignore_index=True)


def create_comprehensive_color_map(df: pd.DataFrame) -> dict:
    """Generates a consistent color map, including special colors for specified categories."""
    therapies = sorted(list(set(np.concatenate([df['item_from'].unique(), df['item_to'].unique()]))))
    color_map = {}
    special_colors = {'No Therapy': '#aaaaaa', 'Other': '#d3d3d3', 'Unknown': '#e5e5e5'}
    for category, color in special_colors.items():
        if category in therapies:
            color_map[category] = color
            therapies.remove(category)
    if not therapies: return color_map
    colormap = cm.get_cmap('tab20b', len(therapies))
    hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, a in colormap(np.linspace(0, 1, len(therapies)))]
    for therapy, color in zip(therapies, hex_colors): color_map[therapy] = color
    return color_map


def plot_sankey_robust(flow_df: pd.DataFrame, therapy_colors: dict, title: str):
    """Generates a Sankey diagram using a structural method that prevents backward links."""
    source_nodes = flow_df[['day_from', 'item_from']].rename(columns={'day_from': 'day', 'item_from': 'label'})
    target_nodes = flow_df[['day_to', 'item_to']].rename(columns={'day_to': 'day', 'item_to': 'label'})
    nodes_df = pd.concat([source_nodes, target_nodes]).drop_duplicates().sort_values(['day', 'label']).reset_index(drop=True)
    nodes_df['id'] = nodes_df.index
    node_map = pd.Series(nodes_df.id.values, index=nodes_df['label'] + '_day_' + nodes_df['day'].astype(str))
    flow_df['source_id'] = (flow_df['item_from'] + '_day_' + flow_df['day_from'].astype(str)).map(node_map)
    flow_df['target_id'] = (flow_df['item_to'] + '_day_' + flow_df['day_to'].astype(str)).map(node_map)
    unique_days = sorted(nodes_df['day'].unique())
    day_x_map = {day: i / (len(unique_days) - 1) if len(unique_days) > 1 else 0.5 for i, day in enumerate(unique_days)}
    node_x = nodes_df['day'].map(day_x_map)
    node_colors = nodes_df['label'].map(therapy_colors).fillna('#CCCCCC')
    link_colors = flow_df['item_from'].map(therapy_colors).fillna('#CCCCCC').apply(lambda h: f"rgba({int(h[1:3],16)},{int(h[3:5],16)},{int(h[5:7],16)},0.4)")
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=25, line=dict(color="black", width=0.5),
                  label=nodes_df['label'], color=node_colors, x=node_x),
        link=dict(source=flow_df['source_id'], target=flow_df['target_id'],
                  value=flow_df['count'], color=link_colors)
    )])
    annotations = [
        dict(x=x, y=-0.07, text=f"<b>Day {d}</b>", showarrow=False,
             font=dict(size=14), xref="paper", yref="paper", xanchor="center")
            for d, x in day_x_map.items()]
    fig.update_layout(title_text=f"<b>{title}</b>",
        font=dict(size=12),
        annotations=annotations,
        margin=dict(b=100))
    fig.show()





# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':

    # .. note:: To ensure the logic is correct, you can add a test case. Create
    #           and filter synthetic patient with a specific therapy progression
    #           over several days. Then, run the script and confirm that the final
    #           plot accurately reflects this known pathway.

    # Constants
    N_PATIENTS = 200          # Number of patients
    MAX_LOS = 6               # Maximum length of stay.
    TOP_N_THERAPIES = 8       # Filter by top most common therapies
    ANALYSIS_LEVEL = 'drug'   # Options ('drug', 'class' and 'aware')

    # Define variables
    therapy_col = COLUMN_FOR_ANALYSIS[ANALYSIS_LEVEL]
    subject_col = 'SUBJECT'
    day_col = 'DAY_NUM'

    # Generate synthetic data.
    patient_df = create_synthetic_data(num_patients=N_PATIENTS, max_los=MAX_LOS)
    #patient_df = create_validation_data()

    # Pre-process patient data
    processed_df = process_patient_data(df=patient_df,
                                        level=ANALYSIS_LEVEL,
                                        therapy_col=therapy_col,
                                        subject_col=subject_col,
                                        day_col=day_col)

    # Filter top n therapies
    filtered_df = limit_therapies_to_top_n(df=processed_df,
                                           therapy_col=therapy_col,
                                           n=TOP_N_THERAPIES)

    # Generate flow data
    flow_data = create_flow_from_patient_data_pivot(df=filtered_df,
                                                    therapy_col=therapy_col,
                                                    subject_col=subject_col,
                                                    day_col=day_col)

    # Display
    if not flow_data.empty:

        # .. note:: Create a custom color scheme to make information easier to understand.
        #           For example, you could use green, yellow, and red for statuses like
        #           'access,' 'watch,' and 'reserve.' You can also assign a unique color
        #           to each drug class and then use different shades of that color for
        #           the individual drugs within the class. For help with this, you can
        #           ask Gemini to generate the color palettes for you

        colors = create_comprehensive_color_map(flow_data)
        plot_title = f"Antimicrobial Therapy Transitions ({ANALYSIS_LEVEL.title()} View)"
        if TOP_N_THERAPIES: plot_title += f" - Top {TOP_N_THERAPIES}"
        plot_sankey_robust(flow_data, colors, title=plot_title)
    else:
        print("No flow data to plot. The dataset might be too small or filtered.")
