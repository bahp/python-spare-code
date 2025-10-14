import pandas as pd
from io import StringIO
import plotly.graph_objects as go
import matplotlib.cm as cm


def create_comprehensive_color_map(df: pd.DataFrame) -> dict:
    """Generates a consistent color map for all therapies."""
    # Consolidate all unique therapy names
    therapies = sorted(list(set(pd.concat([df['item_from'], df['item_to']]))))

    color_map = {}
    special_colors = {'No Therapy': '#A3A3A3', 'Other': '#D3D3D3', 'Unknown': '#E5E5E5'}

    # Assign special colors first
    for category, color in special_colors.items():
        if category in therapies:
            color_map[category] = color
            therapies.remove(category)

    if not therapies:
        return color_map

    # Assign colors from a colormap to the remaining therapies
    colormap = cm.get_cmap('tab20c')  # A good colormap with 20 distinct colors
    # Correct: Expects three values (r, g, b)
    hex_colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colormap.colors]

    for i, therapy in enumerate(therapies):
        # Cycle through the colors if there are more therapies than colors
        color_map[therapy] = hex_colors[i % len(hex_colors)]

    return color_map

def plot_sankey_robust(flow_df: pd.DataFrame, therapy_colors: dict, title: str):
    """
    Generates a Sankey diagram using a structural method that prevents backward links.
    """
    if flow_df.empty:
        print("Warning: Flow DataFrame is empty. Cannot generate plot.")
        return

    # 1. Create a structured DataFrame of all unique nodes
    source_nodes = flow_df[['day_from', 'item_from']].rename(columns={'day_from': 'day', 'item_from': 'label'})
    target_nodes = flow_df[['day_to', 'item_to']].rename(columns={'day_to': 'day', 'item_to': 'label'})

    # 2. Combine, drop duplicates, and sort STRUCTURALLY (by day, then by label)
    # This is the crucial step that guarantees correct temporal ordering.
    nodes_df = pd.concat([source_nodes, target_nodes]) \
                 .drop_duplicates() \
                 .sort_values(['day', 'label']) \
                 .reset_index(drop=True)

    # 3. The index of this sorted DataFrame is now the unique, guaranteed-correct ID
    nodes_df['id'] = nodes_df.index

    # 4. Map the flows to these new, reliable IDs
    node_map = pd.Series(nodes_df.id.values, index=nodes_df['label'] + '_day_' + nodes_df['day'].astype(str)).to_dict()
    flow_df['source_id'] = (flow_df['item_from'] + '_day_' + flow_df['day_from'].astype(str)).map(node_map)
    flow_df['target_id'] = (flow_df['item_to'] + '_day_' + flow_df['day_to'].astype(str)).map(node_map)

    # 5. Generate plot properties from the structured nodes_df
    unique_days = sorted(nodes_df['day'].unique())
    day_x_map = {day: i / (len(unique_days) - 1) if len(unique_days) > 1 else 0.5 for i, day in enumerate(unique_days)}

    node_labels = nodes_df['label'].tolist()
    node_x = nodes_df['day'].map(day_x_map).tolist()
    node_colors = nodes_df['label'].map(therapy_colors).fillna('#A3A3A3').tolist()
    link_colors = flow_df['item_from'].map(therapy_colors).fillna('#A3A3A3').apply(
        lambda hex_val: f"rgba({int(hex_val[1:3], 16)},{int(hex_val[3:5], 16)},{int(hex_val[5:7], 16)},0.4)"
    ).tolist()

    # 6. Create the figure
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=25, line=dict(color="black", width=0.5), label=node_labels, color=node_colors, x=node_x),
        link=dict(source=flow_df['source_id'], target=flow_df['target_id'], value=flow_df['count'], color=link_colors)
    )])

    # Add day labels as annotations
    annotations = [dict(x=x_pos, y=-0.07, text=f"<b>Day {day}</b>", showarrow=False, font=dict(size=16), xref="paper", yref="paper", xanchor="center") for day, x_pos in day_x_map.items()]
    fig.update_layout(title_text=f"<b>{title}</b>", font=dict(size=12), annotations=annotations, margin=dict(b=100))
    fig.show()

def plot_sankey(flow_df: pd.DataFrame, therapy_colors: dict, title: str):
    """
    Generates and displays a Sankey diagram, adding day labels as annotations
    and custom hover data to show node IDs.
    """
    if flow_df.empty:
        print("Warning: The flow DataFrame is empty. Cannot generate a plot.")
        return

    # --- THE FIX: Filter for consecutive day transitions ---
    # This line ensures that we only keep rows where the 'to' day is
    # exactly one day after the 'from' day.
    flow_df = flow_df[flow_df['day_to'] == flow_df['day_from'] + 1].copy()

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
        lambda hex_val: f"rgba({int(hex_val[1:3], 16)},{int(hex_val[3:5], 16)},{int(hex_val[5:7], 16)},0.4)"
    ).tolist()

    # --- THE FIX: Prepare custom hover data and template ---
    # Create an array of custom data. Here, we're passing the node ID for each node.
    node_customdata = nodes_df['id'].values  # <-- NEW

    # Define the HTML-like template for the hover tooltip.
    # %{label} is the node's name.
    # %{customdata[0]} refers to the first (and only) item we passed in customdata.
    # <extra></extra> is a Plotly trick to hide the secondary tooltip box.
    node_hovertemplate = "<b>%{label}</b><br>Node ID: %{customdata}<extra></extra>"  # <-- NEW

    # 4. Create the figure
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
            x=node_x,
            customdata=node_customdata,  # <-- NEW: Pass the custom data here
            hovertemplate=node_hovertemplate  # <-- NEW: Apply the custom template
        ),
        link=dict(
            source=flow_df['source_id'],
            target=flow_df['target_id'],
            value=flow_df['count'],
            color=link_colors
        )
    )])

    # 5. Add day labels as annotations
    annotations = [dict(x=x_pos, y=-0.07, text=f"<b>Day {day}</b>", showarrow=False, font=dict(size=16), xref="paper",
                        yref="paper", xanchor="center") for day, x_pos in day_x_map.items()]
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        font=dict(size=12),
        annotations=annotations,
        margin=dict(b=100)
    )

    fig.show()

# --- Main Execution ---

# 1. Load the exact data that was causing the error
csv_data = """,day_from,item_from,day_to,item_to,count
0,1,3rd Generation Cephalosporin,2,3rd Generation Cephalosporin,4
1,1,3rd Generation Cephalosporin,2,Unknown,1
2,1,Carbapenem,2,3rd Generation Cephalosporin,1
3,1,Carbapenem,2,Carbapenem,3
4,1,Other,2,3rd Generation Cephalosporin,1
5,1,Other,2,Other,19
6,1,Other,2,Penicillin,3
7,1,Other,2,Penicillin and beta-lactamase inhibitor,1
8,1,Other,2,Unknown,1
9,1,Penicillin,2,Carbapenem,1
10,1,Penicillin,2,Other,2
11,1,Penicillin,2,Penicillin,4
12,1,Penicillin,2,Unknown,2
13,1,Penicillin and beta-lactamase inhibitor,2,Penicillin,1
14,1,Penicillin and beta-lactamase inhibitor,2,Penicillin and beta-lactamase inhibitor,3
15,1,Unknown,2,Other,1
16,1,Unknown,2,Unknown,2
17,2,3rd Generation Cephalosporin,3,3rd Generation Cephalosporin,4
18,2,Carbapenem,3,Carbapenem,2
19,2,Carbapenem,3,Other,2
20,2,Other,2,Penicillin,2
21,2,Other,3,Other,14
22,2,Other,3,Penicillin,1
23,2,Other,3,Penicillin and beta-lactamase inhibitor,2
24,2,Other,3,Unknown,3
25,2,Penicillin,2,Other,3
26,2,Penicillin,3,Other,2
27,2,Penicillin,3,Penicillin,3
28,2,Penicillin,3,Unknown,1
29,2,Penicillin and beta-lactamase inhibitor,2,Penicillin,1
30,2,Penicillin and beta-lactamase inhibitor,3,Penicillin and beta-lactamase inhibitor,3
31,2,Unknown,2,Penicillin,1
32,2,Unknown,3,Carbapenem,1
33,2,Unknown,3,Unknown,3
34,3,3rd Generation Cephalosporin,4,3rd Generation Cephalosporin,3
35,3,Carbapenem,4,Carbapenem,2
36,3,Other,4,Other,13
37,3,Other,4,Penicillin,1
38,3,Other,4,Unknown,3
39,3,Penicillin,4,Other,2
40,3,Penicillin,4,Penicillin,2
41,3,Penicillin and beta-lactamase inhibitor,4,3rd Generation Cephalosporin,1
42,3,Penicillin and beta-lactamase inhibitor,4,Other,1
43,3,Penicillin and beta-lactamase inhibitor,4,Penicillin and beta-lactamase inhibitor,3
44,3,Unknown,4,Unknown,6
45,4,3rd Generation Cephalosporin,5,3rd Generation Cephalosporin,2
46,4,3rd Generation Cephalosporin,5,Carbapenem,1
47,4,3rd Generation Cephalosporin,5,Penicillin,1
48,4,Carbapenem,5,Carbapenem,1
49,4,Carbapenem,5,Unknown,1
50,4,Other,5,Other,6
51,4,Other,5,Penicillin,1
52,4,Other,5,Unknown,2
53,4,Penicillin,5,Penicillin,3
54,4,Penicillin and beta-lactamase inhibitor,5,3rd Generation Cephalosporin,1
55,4,Penicillin and beta-lactamase inhibitor,5,Penicillin and beta-lactamase inhibitor,1
56,4,Unknown,5,Other,1
57,4,Unknown,5,Penicillin,1
58,4,Unknown,5,Unknown,5
59,5,Carbapenem,6,Carbapenem,1
60,5,Carbapenem,6,Unknown,1
61,5,Other,6,Other,3
62,5,Penicillin,6,Penicillin,2
63,5,Penicillin,6,Unknown,1
64,5,Unknown,6,Carbapenem,1
65,5,Unknown,6,Unknown,5
66,6,Carbapenem,7,Carbapenem,2
67,6,Other,7,Other,1
68,6,Penicillin,7,Penicillin,1
69,6,Unknown,7,Other,2
70,6,Unknown,7,Unknown,3"""
flow_df = pd.read_csv(StringIO(csv_data), index_col=0)

# 2. Generate the color map from the complete flow data
therapy_colors = create_comprehensive_color_map(flow_df)

#plot_sankey_robust(flow_df, therapy_colors, 'e')

# --- THE DEFINITIVE FIX: Structural Node and Link Creation ---

# 3. Create a structured DataFrame of all unique nodes
#    This is the key to solving the problem. We gather nodes from BOTH sources and targets.
print("--- Step 1: Gathering all source and target nodes ---")

# Gather all nodes that appear as a SOURCE
source_nodes = flow_df[['day_from', 'item_from']].rename(columns={'day_from': 'day', 'item_from': 'label'})
print("Source nodes found:\n", source_nodes)

# Gather all nodes that appear as a TARGET
target_nodes = flow_df[['day_to', 'item_to']].rename(columns={'day_to': 'day', 'item_to': 'label'})
print("\nTarget nodes found:\n", target_nodes)

# 4. Combine source and target nodes and find the unique set
#    This guarantees that nodes like 'Y' on Day 5 are included.
nodes_df = pd.concat([source_nodes, target_nodes]) \
             .drop_duplicates() \
             .sort_values(['day', 'label']) \
             .reset_index(drop=True)

print("\n--- Step 2: Final unique & sorted node list ---")
print("Note that 'Y' on day 5 is correctly included.")
print(nodes_df)


# 5. The index of this sorted DataFrame is now the unique, guaranteed-correct ID
nodes_df['id'] = nodes_df.index

# 6. Map the flows to these new, reliable IDs
node_map = pd.Series(nodes_df.id.values, index=nodes_df['label'] + '_day_' + nodes_df['day'].astype(str)).to_dict()
flow_df['source_id'] = (flow_df['item_from'] + '_day_' + flow_df['day_from'].astype(str)).map(node_map)
flow_df['target_id'] = (flow_df['item_to'] + '_day_' + flow_df['day_to'].astype(str)).map(node_map)

print(flow_df)
a = flow_df[flow_df.item_from.isin(['Other', 'Penicillin']) | flow_df.item_to.isin(['Other', 'Penicillin'])]
print(a)

# 7. Generate all plot properties from the structured nodes_df
unique_days = sorted(nodes_df['day'].unique())
day_x_map = {day: i / (len(unique_days) - 1) if len(unique_days) > 1 else 0.5 for i, day in enumerate(unique_days)}

node_labels = nodes_df['label'].tolist()
node_x = nodes_df['day'].map(day_x_map).tolist()
node_colors = nodes_df['label'].map(therapy_colors).fillna('#A3A3A3').tolist()
link_colors = flow_df['item_from'].map(therapy_colors).fillna('#A3A3A3').apply(
    lambda hex_val: f"rgba({int(hex_val[1:3], 16)},{int(hex_val[3:5], 16)},{int(hex_val[5:7], 16)},0.4)"
).tolist()

# 8. Create the Figure
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(pad=20, thickness=25, line=dict(color="black", width=0.5), label=node_labels, color=node_colors, x=node_x),
    link=dict(source=flow_df['source_id'], target=flow_df['target_id'], value=flow_df['count'], color=link_colors)
)])

# Add day labels as annotations
annotations = [dict(x=x_pos, y=-0.07, text=f"<b>Day {day}</b>", showarrow=False, font=dict(size=16), xref="paper", yref="paper", xanchor="center") for day, x_pos in day_x_map.items()]
fig.update_layout(title_text="<b>Therapy Transitions (With Terminal Nodes)</b>", font=dict(size=12), annotations=annotations, margin=dict(b=100))
fig.show()





import sys
sys.exit()
# --- THE DEFINITIVE FIX: Structural Node and Link Creation ---

# 3. Create a structured DataFrame of all unique nodes
source_nodes = flow_df[['day_from', 'item_from']].rename(columns={'day_from': 'day', 'item_from': 'label'})
target_nodes = flow_df[['day_to', 'item_to']].rename(columns={'day_to': 'day', 'item_to': 'label'})

# 4. Combine, drop duplicates, and sort STRUCTURALLY (by day, then by label)
# This is the crucial step that guarantees correct temporal ordering.
nodes_df = pd.concat([source_nodes, target_nodes]) \
    .drop_duplicates() \
    .sort_values(['day', 'label']) \
    .reset_index(drop=True)

# 5. The index of this sorted DataFrame is now the unique, guaranteed-correct ID
nodes_df['id'] = nodes_df.index

print(nodes_df)

# 6. Map the flows to these new, reliable IDs
node_map = pd.Series(nodes_df.id.values, index=nodes_df['label'] + '_day_' + nodes_df['day'].astype(str)).to_dict()
flow_df['source_id'] = (flow_df['item_from'] + '_day_' + flow_df['day_from'].astype(str)).map(node_map)
flow_df['target_id'] = (flow_df['item_to'] + '_day_' + flow_df['day_to'].astype(str)).map(node_map)

# 7. Generate all plot properties from the structured nodes_df
unique_days = sorted(nodes_df['day'].unique())
day_x_map = {day: i / (len(unique_days) - 1) if len(unique_days) > 1 else 0.5 for i, day in enumerate(unique_days)}

node_labels = nodes_df['label'].tolist()
node_x = nodes_df['day'].map(day_x_map).tolist()
node_colors = nodes_df['label'].map(therapy_colors).fillna('#A3A3A3').tolist()
link_colors = flow_df['item_from'].map(therapy_colors).fillna('#A3A3A3').apply(
    lambda hex_val: f"rgba({int(hex_val[1:3], 16)},{int(hex_val[3:5], 16)},{int(hex_val[5:7], 16)},0.4)"
).tolist()

# 8. Create the Figure
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(pad=20, thickness=25, line=dict(color="black", width=0.5), label=node_labels, color=node_colors,
              x=node_x),
    link=dict(source=flow_df['source_id'], target=flow_df['target_id'], value=flow_df['count'], color=link_colors)
)])

# Add day labels as annotations
annotations = [
    dict(x=x_pos, y=-0.07, text=f"<b>Day {day}</b>", showarrow=False, font=dict(size=16), xref="paper", yref="paper",
         xanchor="center") for day, x_pos in day_x_map.items()]
fig.update_layout(
    title_text="<b>Patient Therapy Transitions (Corrected)</b>",
    font=dict(size=12),
    annotations=annotations,
    margin=dict(b=100)  # Increase bottom margin for labels
)
fig.show()
