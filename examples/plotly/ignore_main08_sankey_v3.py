import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. Simulate Patient Data (Same as before for consistency) ---
np.random.seed(42)  # for reproducibility

patient_ids = [f'P{i}' for i in range(1, 21)]  # 20 patients
days = [1, 2, 3]
therapies = ['Therapy A', 'Therapy B', 'Therapy C', 'No Therapy']

records = []
for pid in patient_ids:
    records.append({'Patient_ID': pid, 'Day': 1, 'Therapy': np.random.choice(therapies[:-1], p=[0.4, 0.4, 0.2])})

for day_idx in range(1, len(days)):
    current_day = days[day_idx]
    previous_day = days[day_idx - 1]

    previous_day_df = pd.DataFrame(records)[pd.DataFrame(records)['Day'] == previous_day]

    for _, row in previous_day_df.iterrows():
        pid = row['Patient_ID']
        prev_therapy = row['Therapy']

        if prev_therapy == 'Therapy A':
            new_therapy = np.random.choice(['Therapy A', 'Therapy B', 'No Therapy'], p=[0.6, 0.3, 0.1])
        elif prev_therapy == 'Therapy B':
            new_therapy = np.random.choice(['Therapy B', 'Therapy C', 'Therapy A', 'No Therapy'],
                                           p=[0.5, 0.3, 0.1, 0.1])
        elif prev_therapy == 'Therapy C':
            new_therapy = np.random.choice(['Therapy C', 'Therapy A', 'No Therapy'], p=[0.6, 0.2, 0.2])
        elif prev_therapy == 'No Therapy':
            new_therapy = np.random.choice(['No Therapy', 'Therapy A', 'Therapy B'], p=[0.7, 0.15, 0.15])

        records.append({'Patient_ID': pid, 'Day': current_day, 'Therapy': new_therapy})

patient_df = pd.DataFrame(records)

# --- 2. Prepare Data for Sankey Diagram (Same logic, slightly refined output) ---
sankey_data = []
for i in range(len(days) - 1):
    current_day = days[i]
    next_day = days[i + 1]

    merged_df = pd.merge(
        patient_df[patient_df['Day'] == current_day],
        patient_df[patient_df['Day'] == next_day],
        on='Patient_ID',
        suffixes=(f'_Day{current_day}', f'_Day{next_day}')
    )

    flow_df = merged_df.groupby([f'Therapy_Day{current_day}', f'Therapy_Day{next_day}']).size().reset_index(
        name='Patients')
    flow_df.columns = ['Source_Therapy', 'Target_Therapy', 'Patients']

    flow_df['Source'] = flow_df['Source_Therapy'] + f' (Day {current_day})'
    flow_df['Target'] = flow_df['Target_Therapy'] + f' (Day {next_day})'
    flow_df['Value'] = flow_df['Patients']

    sankey_data.append(flow_df[['Source', 'Target', 'Value']])

final_flow_df = pd.concat(sankey_data)

print(final_flow_df)

all_nodes = pd.concat([final_flow_df['Source'], final_flow_df['Target']]).unique()
node_map = {node: i for i, node in enumerate(all_nodes)}

final_flow_df['Source_ID'] = final_flow_df['Source'].map(node_map)
final_flow_df['Target_ID'] = final_flow_df['Target'].map(node_map)

print(final_flow_df)

# --- 3. Define a Nicer Color Palette ---
# You can customize these hex codes or use Plotly's built-in color scales.
# Example: https://plotly.com/python/builtin-colorscales/
therapy_colors = {
    'Therapy A': '#8A4F7D',  # Darker Purple
    'Therapy B': '#E07C24',  # Orange
    'Therapy C': '#55A8A2',  # Teal
    'No Therapy': '#A3A3A3'  # Grey for no therapy
}

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
