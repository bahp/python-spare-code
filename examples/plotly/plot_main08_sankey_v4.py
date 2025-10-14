import pandas as pd
from io import StringIO
import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm

# -------------------------------------------------------------------
# Methods
# -------------------------------------------------------------------
def prepare_sankey_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame of daily transitions into a format suitable for a Plotly Sankey diagram.

    Args:
        df: A DataFrame with columns ['day_from', 'item_from', 'day_to', 'item_to', 'count'].

    Returns:
        A DataFrame with columns ['Source', 'Target', 'Value', 'Source_Therapy',
                                  'Source_ID', 'Target_ID'] ready for plotting.
    """
    # 1. Create the 'Source' and 'Target' node labels
    df['Source'] = df['item_from'] + ' (Day ' + df['day_from'].astype(str) + ')'
    df['Target'] = df['item_to'] + ' (Day ' + df['day_to'].astype(str) + ')'

    # 2. Rename the count column to 'Value' for the Sankey diagram
    df.rename(columns={'count': 'Value'}, inplace=True)

    # 3. Create a unique list of all nodes (both sources and targets)
    all_nodes = pd.concat([df['Source'], df['Target']]).unique()

    # 4. Create a mapping from the node label to a unique integer ID
    node_map = {node: i for i, node in enumerate(all_nodes)}

    # 5. Map the Source and Target labels to their integer IDs
    df['Source_ID'] = df['Source'].map(node_map)
    df['Target_ID'] = df['Target'].map(node_map)

    # 6. Keep the original 'item_from' for potential coloring later, renaming it for clarity
    df.rename(columns={'item_from': 'Source_Therapy'}, inplace=True)

    # 7. Return the final formatted DataFrame with selected columns
    return df[['Source', 'Target', 'Value', 'Source_Therapy', 'Source_ID', 'Target_ID']]


def create_comprehensive_color_map(df: pd.DataFrame, source_col: str, target_col: str) -> dict:
    """
    Generates a color map for all unique therapies found in both source and target columns.

    A special, consistent grey color is assigned to 'No Therapy' if it exists.

    Args:
        df: The pandas DataFrame containing the therapy flow data.
        source_col: The column name for source therapies (e.g., 'item_from').
        target_col: The column name for target therapies (e.g., 'item_to').

    Returns:
        A dictionary where keys are therapy names and values are color hex codes.
    """
    # 1. Get unique therapies from both source and target columns
    source_therapies = df[source_col].unique()
    target_therapies = df[target_col].unique()

    # 2. Combine them into a single set to get the overall unique list
    all_unique_therapies = sorted(list(set(np.concatenate([source_therapies, target_therapies]))))

    # 3. Handle the 'No Therapy' case separately
    no_therapy_color = '#A3A3A3'  # Consistent grey
    color_map = {}

    if 'No Therapy' in all_unique_therapies:
        color_map['No Therapy'] = no_therapy_color
        all_unique_therapies.remove('No Therapy')

    # 4. Use a Matplotlib colormap to generate distinct colors for the rest
    num_therapies = len(all_unique_therapies)
    if num_therapies == 0:
        return color_map  # Return early if only 'No Therapy' was present

    # Use 'tab20b' or another qualitative colormap for good distinction
    colormap = cm.get_cmap('tab20b', num_therapies)

    hex_colors = [
        '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, a in colormap(np.linspace(0, 1, num_therapies))
    ]

    # 5. Create the final color map dictionary
    for therapy, color in zip(all_unique_therapies, hex_colors):
        color_map[therapy] = color

    return color_map



# --- Example Usage ---

# 1. Create a sample DataFrame from your provided data string
csv_data = """,day_from,item_from,day_to,item_to,count
0,0,2nd Generation Cephalosporin,1,3rd Generation Cephalosporin,1
1,0,2nd Generation Cephalosporin,1,Aminoglycoside,1
2,0,2nd Generation Cephalosporin,1,Fluoroquinolone,2
3,0,2nd Generation Cephalosporin,1,Oxazolidinone,2
4,0,2nd Generation Cephalosporin,1,Tetracycline,1
5,0,3rd Generation Cephalosporin,1,2nd Generation Cephalosporin,3
6,0,3rd Generation Cephalosporin,1,3rd Generation Cephalosporin,1
7,0,3rd Generation Cephalosporin,1,4th Generation Cephalosporin,2
8,0,3rd Generation Cephalosporin,1,Aminoglycoside,2
9,0,3rd Generation Cephalosporin,1,Amphenicol,2
10,0,3rd Generation Cephalosporin,1,Fluoroquinolone,2"""



# Use StringIO to read the string data into a pandas DataFrame
#data_io = StringIO(csv_data)
#input_df = pd.read_csv(data_io, index_col=0)
input_df = pd.read_csv('./objects/plot_main08_sankey/flow.csv')

# 2. Convert the data using the function
sankey_ready_df = prepare_sankey_data(input_df)

# 3. Print the result to verify the output format
print("--- Converted DataFrame for Sankey Diagram ---")
print(sankey_ready_df)

final_flow_df = sankey_ready_df

all_nodes = pd.concat([final_flow_df['Source'], final_flow_df['Target']]).unique()
node_map = {node: i for i, node in enumerate(all_nodes)}


# --- 3. Define a Nicer Color Palette ---
# You can customize these hex codes or use Plotly's built-in color scales.
# Example: https://plotly.com/python/builtin-colorscales/
therapy_colors = {
    'Therapy A': '#8A4F7D',  # Darker Purple
    'Therapy B': '#E07C24',  # Orange
    'Therapy C': '#55A8A2',  # Teal
    'No Therapy': '#A3A3A3'  # Grey for no therapy
}

therapy_colors = create_comprehensive_color_map(final_flow_df, 'Source', 'Target')

# Create node colors list
node_colors = []
for label in all_nodes:
    assigned_color = 'rgba(0,0,0,0.8)'  # Default if not found
    for therapy, color_hex in therapy_colors.items():
        if therapy in label:
            assigned_color = color_hex
            break
    node_colors.append(assigned_color)

# Create link colors list (softer version of source node color)
link_colors = []
for i in range(len(final_flow_df)):
    source_therapy_name = final_flow_df['Source'].iloc[i]
    base_color_hex = therapy_colors.get(source_therapy_name, '#A3A3A3')  # Default to grey

    # Convert hex to rgba with alpha for transparency
    # Remove '#' and convert to int base 16
    r = int(base_color_hex[1:3], 16)
    g = int(base_color_hex[3:5], 16)
    b = int(base_color_hex[5:7], 16)
    link_colors.append(f'rgba({r},{g},{b},0.3)')  # 0.3 for light transparency

# --- 4. Create the Sankey Diagram with enhanced styling ---

fig = go.Figure(data=[go.Sankey(
    # Node Styling
    node=dict(
        pad=20,  # Increased padding between nodes and links
        thickness=25,  # Slightly thicker nodes
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=node_colors,  # Custom, nicer node colors
        x=[
              0, 0, 0, 0,  # Day 1 nodes at x=0
              0.5, 0.5, 0.5, 0.5,  # Day 2 nodes at x=0.5 (midway)
              1, 1, 1, 1  # Day 3 nodes at x=1
          ][:len(all_nodes)]  # Ensure x-coords match number of nodes
    ),
    # Link Styling (the flows)
    link=dict(
        source=final_flow_df['Source_ID'],
        target=final_flow_df['Target_ID'],
        value=final_flow_df['Value'],
        color=link_colors,  # Softer, custom link colors
        # line=dict(color='lightgrey', width=0.5) # Optional: outline for links
    ))])

fig.update_layout(
    title_text="<b>Patient Antimicrobial Therapy Transitions Over Days</b>",  # Bold title
    font=dict(size=14, family="Arial", color="black"),  # Nicer font
    hovermode="x unified",  # Tooltips show all relevant info on hover
    # plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background (optional)
    # paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background (optional)
    margin=dict(l=40, r=40, t=80, b=40)  # Adjust margins for better fit
)

fig.show()

