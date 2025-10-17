"""
UKVI Travel History Visualizer (plotly)
=======================================

Improving the visualisation of previous example using plotly.

"""

import pandas as pd
import plotly.express as px
from pathlib import Path

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

def display_flights_plotly_final(df):
    """
    Generates an interactive flight schedule plot ensuring each trip is on its
    own unique row. Includes custom text labels, month lines, and alternating
    background shading.
    """
    # --- 1. Data Pre-processing ---
    df = df.copy()

    # Force columns to datetime, turning any errors into NaT (Not a Time)
    df['Outbound Date'] = pd.to_datetime(df['Outbound Date'], errors='coerce')
    df['Inbound Date'] = pd.to_datetime(df['Inbound Date'], errors='coerce')

    # Strip the time component to prevent rendering glitches
    df['Outbound Date'] = df['Outbound Date'].dt.normalize()
    df['Inbound Date'] = df['Inbound Date'].dt.normalize()

    # Compute extra information
    df['Days Difference'] = (df['Inbound Date'] - df['Outbound Date']).dt.days
    df['Departure Airport'] = df['Outbound Ports'].apply(lambda x: x.split('-')[0])

    # Sort by date and reset the index. This index (0, 1, 2...) will be the
    # unique y-axis position for each trip, guaranteeing one row per trip.
    df = df.sort_values('Outbound Date', ascending=True).reset_index(drop=True)

    # Hack: create y_axes_str to control order.
    # Calculate the required width (e.g., 2 for up to 99 items, 3 for up to 999)
    pad_width = len(str(len(df)))
    df['y_axis_str'] = df.index.astype(str).str.zfill(pad_width)

    # Create the label for annotations and hovering
    df['Voyage Label'] = df.apply(
        lambda row: f"{row['Outbound Ports']} ({row['Outbound Date'].strftime('%d %b')}) → "
                    f"{row['Inbound Ports']} ({row['Inbound Date'].strftime('%d %b')}) | "
                    f"{row['Days Difference']} Days",
        axis=1
    )

    # Show DataFrame to plot
    print("\nRoundtrips:")
    print(df)

    # --- 2. Generate Background Shapes ---
    shapes = []
    min_date = df['Outbound Date'].min().normalize()
    max_date = df['Inbound Date'].max().normalize() + pd.DateOffset(months=1)
    month_starts = pd.date_range(start=min_date, end=max_date, freq='MS')
    for i, month_start in enumerate(month_starts):
        shapes.append({
            'type': 'line', 'xref': 'x', 'yref': 'paper',
            'x0': month_start, 'y0': 0, 'x1': month_start, 'y1': 1,
            'line': {'color': 'Gainsboro', 'width': 1, 'dash': 'dot'},
            'layer': 'below'
        })
        if i % 2 == 0:
            shapes.append({
                'type': 'rect', 'xref': 'x', 'yref': 'paper',
                'x0': month_start, 'y0': 0,
                'x1': month_start + pd.DateOffset(months=1), 'y1': 1,
                'fillcolor': 'LightGray', 'opacity': 0.1,
                'line': {'width': 0},
                'layer': 'below'
            })

    # --- 3. Generate Annotations for Text Labels ---
    annotations = []
    for index, row in df.iterrows():
        annotations.append(
            dict(
                x=row['Inbound Date'],
                y=row['y_axis_str'], #index,  # Use the unique numerical index for the y-position
                text=f"  {row['Voyage Label']}",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                align='left'
            )
        )

    # --- 4. Core Plotting ---
    fig = px.timeline(
        df,
        x_start="Outbound Date",
        x_end="Inbound Date",
        y='y_axis_str', #df.index,  # KEY FIX: Use the unique index for the y-axis
        color="Departure Airport",
        hover_name="Voyage Label",
        hover_data={'Days Difference': True},
        title=f"Voyage Durations (Total Abroad: {df['Days Difference'].sum()} days)",
        opacity=1.0,
        category_orders={"y_axis_str": df.y_axis_str.tolist()[::-1]},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # --- 5. Final Layout Updates ---
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title="Date",
        legend_title="Departure Airport",
        shapes=shapes,
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Hide the meaningless y-axis numbers and title
    fig.update_yaxes(
        autorange="reversed",
        showticklabels=False,
        title_text=""
    )

    if not TERMINAL:
        fig.update_yaxes(automargin=False)
        fig.update_layout(
            title=dict(
                x=0.5,  # Center the title
                y=0.95,  # Move it up slightly
                xanchor='center',
                yanchor='top'
            ),
            margin=dict(l=0, r=20, t=110, b=20)
        )
        # We could adjust x-axis to get use more space.

    fig.show()



    # Show
    from plotly.io import show
    show(fig)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
# Libraries
from pathlib import Path

# Configuration
id = '1085721'
out_path = Path(f'./outputs/{id}')

try:
    flight_df = pd.read_json(out_path / 'roundtrips.json')
    flight_df['Outbound Date'] = pd.to_datetime(flight_df['Outbound Date'], unit='ms')
    flight_df['Inbound Date'] = pd.to_datetime(flight_df['Inbound Date'], unit='ms')
except FileNotFoundError:
    print(f"Error: File 'roundtrips.json' not found. Displaying sample data.")
    sample_data = [
        {"Outbound Date": "2024-01-15",
         "Inbound Date": "2024-02-23",
         "Outbound Ports": "LHR-JFK",
         "Inbound Ports": "JFK-LHR",
         "Voyage Code": "VS003"},
        {"Outbound Date": "2024-03-05",
         "Inbound Date": "2024-03-20",
         "Outbound Ports": "LGW-BCN",
         "Inbound Ports": "BCN-LGW",
         "Voyage Code": "BA2712"},
        {"Outbound Date": "2024-04-20",
         "Inbound Date": "2024-05-15",
         "Outbound Ports": "STN-FCO",
         "Inbound Ports": "FCO-STN",
         "Voyage Code": "FR123"},
        {"Outbound Date": "2024-06-10",
         "Inbound Date": "2024-06-25",
         "Outbound Ports": "LHR-BOS",
         "Inbound Ports": "BOS-LHR",
         "Voyage Code": "VS011"}
    ]
    flight_df = pd.DataFrame(sample_data)
    flight_df['Outbound Date'] = pd.to_datetime(flight_df['Outbound Date'])
    flight_df['Inbound Date'] = pd.to_datetime(flight_df['Inbound Date'])

display_flights_plotly_final(flight_df)