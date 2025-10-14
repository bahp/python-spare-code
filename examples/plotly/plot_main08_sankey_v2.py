import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- 1. Simulate Patient Data ---
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

# --- 2. Prepare Data for Sankey Diagram (Corrected) ---
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

    # This renames the two grouped columns to make them easier to work with
    flow_df.columns = ['Source_Therapy', 'Target_Therapy', 'Patients']

    # Create the full node labels needed for the plot
    flow_df['Source'] = flow_df['Source_Therapy'] + f' (Day {current_day})'
    flow_df['Target'] = flow_df['Target_Therapy'] + f' (Day {next_day})'
    flow_df['Value'] = flow_df['Patients']

    # ** THE FIX IS HERE **
    # We now keep 'Source_Therapy' in the data we append.
    sankey_data.append(flow_df[['Source', 'Target', 'Value', 'Source_Therapy']])

# Combine all flow dataframes
final_flow_df = pd.concat(sankey_data, ignore_index=True)

all_nodes = pd.concat([final_flow_df['Source'], final_flow_df['Target']]).unique()
node_map = {node: i for i, node in enumerate(all_nodes)}

final_flow_df['Source_ID'] = final_flow_df['Source'].map(node_map)
final_flow_df['Target_ID'] = final_flow_df['Target'].map(node_map)

# --- 3. Define the Color Palette ---
therapy_colors = {
    'Therapy A': '#8A4F7D',  # Darker Purple
    'Therapy B': '#E07C24',  # Orange
    'Therapy C': '#55A8A2',  # Teal
    'No Therapy': '#A3A3A3'  # Grey
}

# Create node colors list
node_colors = []
for label in all_nodes:
    for therapy, color_hex in therapy_colors.items():
        if therapy in label:
            node_colors.append(color_hex)
            break

# Create link colors list (this will now work correctly)
link_colors = []
for _, row in final_flow_df.iterrows():
    source_therapy_name = row['Source_Therapy']
    base_color_hex = therapy_colors.get(source_therapy_name, '#A3A3A3')
    r, g, b = int(base_color_hex[1:3], 16), int(base_color_hex[3:5], 16), int(base_color_hex[5:7], 16)
    link_colors.append(f'rgba({r},{g},{b},0.3)')

# --- 4. Create the Enhanced Sankey Diagram ---
fig = go.Figure(data=[go.Sankey(
    arrangement='snap',  # Snap nodes to vertical columns
    node=dict(
        pad=20,
        thickness=25,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=node_colors,
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
