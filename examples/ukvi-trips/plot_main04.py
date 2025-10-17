"""
UKVI Travel History Visualizer (plotly)
=======================================


"""

import pandas as pd
import plotly.express as px
from pathlib import Path


def display_flights_plotly_final(df):
    """
    Generates an interactive flight schedule plot with month lines and
    alternating background shading, sorted with the oldest flights
    at the bottom of the y-axis.
    """
    # --- 1. Data Pre-processing ---
    df = df.copy()
    # CORRECTED: Calculate difference between Inbound and Outbound dates
    df['Days Difference'] = (df['Inbound Date'] - df['Outbound Date']).dt.days
    df['Departure Airport'] = df['Outbound Ports'].apply(lambda x: x.split('-')[0])

    # Sort by date to prepare the y-axis order
    df = df.sort_values('Outbound Date', ascending=True).reset_index(drop=True)
    df['Voyage Label'] = df.apply(
        lambda row: f"{row['Voyage Code']}: {row['Outbound Ports']} → {row['Inbound Ports']}",
        axis=1
    )

    # --- 2. Generate Background Shapes ---
    shapes = []
    # Ensure date range covers the full extent of the data
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

    # --- 3. Core Plotting ---
    # The title will now correctly calculate the sum
    fig = px.timeline(
        df,
        x_start="Outbound Date",
        x_end="Inbound Date",
        y="Voyage Label",
        color="Departure Airport",
        category_orders={"Voyage Label": df['Voyage Label'].tolist()},  # This preserves the sort order
        hover_name="Voyage Label",
        hover_data={'Days Difference': True},
        title=f"Voyage Durations (Total Abroad: {df['Days Difference'].sum()} days)"
    )

    # --- 4. Final Layout Updates ---
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title="Date",
        yaxis_title="Voyage",
        legend_title="Departure Airport",
        shapes=shapes,
        legend=dict(
            orientation="h",  # "h" for horizontal
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # ADDED: Reverse the y-axis to show oldest flights at the bottom
    fig.update_yaxes(autorange="reversed")

    fig.show()
    from plotly.io import show
    show(fig)




# -------------------------------------------------------
# Mqain
# -------------------------------------------------------
# Libraries
from pathlib import Path

# Configuration
id = '775243'
id = '1085721'
out_path = Path(f'./outputs/{id}')

try:
    flight_df = pd.read_json(out_path / 'roundtrips.json')
    flight_df['Outbound Date'] = \
        pd.to_datetime(flight_df['Outbound Date'], unit='ms')
    flight_df['Inbound Date'] = \
        pd.to_datetime(flight_df['Inbound Date'], unit='ms')

except FileNotFoundError:
    print(f"Error: File 'roundtrips.json' not found. Displaying sample data.")
    sample_data = [
        {"Outbound Date": "2024-01-15",
         "Inbound Date": "2024-02-10",
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
         "Voyage Code": "FR123"}
    ]

    flight_df = pd.DataFrame(sample_data)
    # Load DataFrame (as extracted)
    flight_df['Outbound Date'] = pd.to_datetime(flight_df['Outbound Date'])
    flight_df['Inbound Date'] = pd.to_datetime(flight_df['Inbound Date'])

display_flights_plotly_final(flight_df)

