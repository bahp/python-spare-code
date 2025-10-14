import pandas as pd
from io import StringIO
import plotly.graph_objects as go
import matplotlib.cm as cm


def create_comprehensive_color_map(df: pd.DataFrame) -> dict:
    """Generates a consistent color map for all therapies."""
    # Consolidate all unique therapy names from the data
    therapies = sorted(list(set(pd.concat([df['item_from'], df['item_to']]))))

    color_map = {}
    # Define fixed colors for special, consistent categories
    special_colors = {'No Therapy': '#A3A3A3', 'Other': '#D3D3D3', 'Unknown': '#E5E5E5'}

    # Assign special colors first and remove them from the main list
    for category, color in special_colors.items():
        if category in therapies:
            color_map[category] = color
            therapies.remove(category)

    if not therapies:
        return color_map

    # Assign dynamic colors from a colormap to the remaining therapies
    colormap = cm.get_cmap('tab20c')  # A good colormap with 20 distinct colors
    hex_colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colormap.colors]

    for i, therapy in enumerate(therapies):
        # Cycle through the available colors if there are more therapies than colors
        color_map[therapy] = hex_colors[i % len(hex_colors)]

    return color_map


# --- Main Execution ---

# 1. Load the exact data that was causing the error
csv_data = """day_from,item_from,day_to,item_to,count
4,No Therapy,5,No Therapy,5
4,No Therapy,5,Other,2
4,Other,5,No Therapy,2
4,Other,5,Other,9
4,Other,5,meropenem,1
4,ceftriaxone,5,ceftriaxone,1
4,ceftriaxone,5,pivmecillinam,1
4,co-amoxiclav,5,ceftriaxone,1
4,co-amoxiclav,5,co-amoxiclav,1
4,meropenem,5,Other,1
4,meropenem,5,meropenem,1
4,pivmecillinam,5,pivmecillinam,2"""
flow_df = pd.read_csv(StringIO(csv_data))

# Add a hypothetical jump to Day 7 to demonstrate the fix
flow_df.loc[len(flow_df)] = [4, 'co-amoxiclav', 7, 'Other', 1]

# 2. Generate the color map from the complete flow data
therapy_colors = create_comprehensive_color_map(flow_df)

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
    margin=dict(b=100)
)
fig.show()
