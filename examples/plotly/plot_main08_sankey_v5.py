import pandas as pd
from io import StringIO
import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm
import re  # Import the regular expressions module



# -------------------------------------------------------------------
# Methods (Your functions are correct, no changes needed here)
# -------------------------------------------------------------------
def prepare_sankey_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Source'] = df['item_from'] + ' (Day ' + df['day_from'].astype(str) + ')'
    df['Target'] = df['item_to'] + ' (Day ' + df['day_to'].astype(str) + ')'
    df.rename(columns={'count': 'Value'}, inplace=True)

    # **FIX**: We will create the node map here based on a properly sorted list later
    # For now, just prepare the main dataframe

    df.rename(columns={'item_from': 'Source_Therapy'}, inplace=True)
    df.rename(columns={'item_to': 'Target_Therapy'}, inplace=True)  # Also keep target therapy for coloring

    return df[['Source', 'Target', 'Value', 'Source_Therapy', 'Target_Therapy']]


def create_comprehensive_color_map(df: pd.DataFrame, source_col: str, target_col: str) -> dict:
    source_therapies = df[source_col].unique()
    target_therapies = df[target_col].unique()
    all_unique_therapies = sorted(list(set(np.concatenate([source_therapies, target_therapies]))))

    no_therapy_color = '#A3A3A3'
    color_map = {}
    if 'No Therapy' in all_unique_therapies:
        color_map['No Therapy'] = no_therapy_color
        all_unique_therapies.remove('No Therapy')

    num_therapies = len(all_unique_therapies)
    if num_therapies == 0:
        return color_map

    colormap = cm.get_cmap('tab20b', num_therapies)
    hex_colors = [
        '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, a in colormap(np.linspace(0, 1, num_therapies))
    ]

    for therapy, color in zip(all_unique_therapies, hex_colors):
        color_map[therapy] = color

    return color_map


# --- Main Script ---

# 1. Load Data
try:
    # Use your local file
    input_df = pd.read_csv('./objects/plot_main08_sankey/flow.csv')
except FileNotFoundError:
    # Fallback to sample data if file not found
    print("Warning: local file not found. Using sample data.")
    csv_data = """,day_from,item_from,day_to,item_to,count
    0,0,2nd Generation Cephalosporin,1,3rd Generation Cephalosporin,1
    1,0,2nd Generation Cephalosporin,1,Aminoglycoside,1
    2,0,2nd Generation Cephalosporin,1,Fluoroquinolone,2
    3,0,3rd Generation Cephalosporin,1,2nd Generation Cephalosporin,3
    4,0,3rd Generation Cephalosporin,1,4th Generation Cephalosporin,2"""
    data_io = StringIO(csv_data)
    input_df = pd.read_csv(data_io, index_col=0)

# 2. Prepare the flow DataFrame
sankey_ready_df = prepare_sankey_data(input_df.copy())
final_flow_df = sankey_ready_df

# --- FIX 1: Sort nodes correctly and create the node map ---
# Get all unique node labels
all_nodes_unsorted = pd.concat([final_flow_df['Source'], final_flow_df['Target']]).unique()


# Define a function to extract the day number from a node label
def get_day_from_label(label):
    match = re.search(r'\(Day (\d+)\)', label)
    return int(match.group(1)) if match else -1


# Sort the nodes first by day, then alphabetically
all_nodes_sorted = sorted(all_nodes_unsorted, key=lambda x: (get_day_from_label(x), x))
all_nodes = all_nodes_sorted  # Use this sorted list from now on

# Now, create the node map from the correctly sorted list
node_map = {node: i for i, node in enumerate(all_nodes)}

# Apply the mapping to the DataFrame
final_flow_df['Source_ID'] = final_flow_df['Source'].map(node_map)
final_flow_df['Target_ID'] = final_flow_df['Target'].map(node_map)

# --- FIX 2: Generate colors correctly ---
# Generate the color map from the original, unmodified data for accuracy
therapy_colors = create_comprehensive_color_map(input_df, 'item_from', 'item_to')

# Create node colors list using the base therapy name
node_colors = []
for label in all_nodes:
    # Find which therapy name is in the full label
    base_therapy = next((therapy for therapy in therapy_colors if therapy in label), None)
    node_colors.append(therapy_colors.get(base_therapy, '#A3A3A3'))  # Default to grey

# Create link colors based on the 'Source_Therapy' column
link_colors = []
for _, row in final_flow_df.iterrows():
    base_color_hex = therapy_colors.get(row['Source_Therapy'], '#A3A3A3')
    r, g, b = int(base_color_hex[1:3], 16), int(base_color_hex[3:5], 16), int(base_color_hex[5:7], 16)
    link_colors.append(f'rgba({r},{g},{b},0.4)')

# --- FIX 3: Dynamically generate node x-positions ---
unique_days = sorted(input_df['day_from'].unique())
day_x_mapping = {day: i / (len(unique_days)) for i, day in enumerate(unique_days)}
# Also map the final day
final_day = input_df['day_to'].max()
if final_day not in day_x_mapping:
    day_x_mapping[final_day] = 1.0

node_x = [day_x_mapping[get_day_from_label(label)] for label in all_nodes]

# --- 4. Create the Sankey Diagram ---
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",  # Aligns nodes vertically
    node=dict(
        pad=20,
        thickness=25,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=node_colors,
        x=node_x,  # Use the dynamically generated x positions
    ),
    link=dict(
        source=final_flow_df['Source_ID'],
        target=final_flow_df['Target_ID'],
        value=final_flow_df['Value'],
        color=link_colors,
    ))])

fig.update_layout(
    title_text="<b>Patient Antimicrobial Therapy Transitions Over Days</b>",
    font=dict(size=14, family="Arial", color="black"),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.show()
