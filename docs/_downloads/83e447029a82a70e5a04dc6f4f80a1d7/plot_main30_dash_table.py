"""
30. Dash Table app
====================

This script creates a Dash web application that demonstrates the interactive
capabilities of the dash_table.DataTable component. The application features
a table that is fully editable, allowing users to modify cell values directly.
Users can also dynamically add new rows with a button click and add new, named
columns using an input field.

The core feature of this example is the real-time link between the data table a
nd a dcc.Graph component. Any modification to the table's data or structure—such
as adding a row, deleting a column, or editing a cell's value—automatically
triggers an update to a heatmap below. This provides an immediate visual
representation of the table's data, showcasing how to build a fully interactive
dashboard where data manipulation and visualization are seamlessly connected.

.. note:: Open your browser and go to http://127.0.0.1:8050

"""

# sphinx_gallery_thumbnail_path = '_static/images/thumbnails/thumbnail-plotly-main30-dash-table.png'

# Libraries
import dash
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import dcc
from dash import html

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

# Create app
app = dash.Dash('table-app')

# Define layout
app.layout = html.Div([
    html.Div([
        dcc.Input(
            id='adding-rows-name',
            placeholder='Enter a column name...',
            value='',
            style={'padding': 10}
        ),
        html.Button('Add Column', id='adding-rows-button', n_clicks=0)
    ], style={'height': 50}),

    dash_table.DataTable(
        id='adding-rows-table',
        columns=[{
            'name': 'Column {}'.format(i),
            'id': 'column-{}'.format(i),
            'deletable': True,
            'renamable': True
        } for i in range(1, 5)],
        data=[
            {'column-{}'.format(i): (j + (i-1)*5) for i in range(1, 5)}
            for j in range(5)
        ],
        editable=True,
        row_deletable=True
    ),

    html.Button('Add Row', id='editing-rows-button', n_clicks=0),

    dcc.Graph(id='adding-rows-graph')
])


@app.callback(
    Output('adding-rows-table', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('adding-rows-table', 'data'),
    State('adding-rows-table', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


@app.callback(
    Output('adding-rows-table', 'columns'),
    Input('adding-rows-button', 'n_clicks'),
    State('adding-rows-name', 'value'),
    State('adding-rows-table', 'columns'))
def update_columns(n_clicks, value, existing_columns):
    if n_clicks > 0:
        existing_columns.append({
            'id': value, 'name': value,
            'renamable': True, 'deletable': True
        })
    return existing_columns


@app.callback(
    Output('adding-rows-graph', 'figure'),
    Input('adding-rows-table', 'data'),
    Input('adding-rows-table', 'columns'))
def display_output(rows, columns):
    return {
        'data': [{
            'type': 'heatmap',
            'z': [[row.get(c['id'], None) for c in columns] for row in rows],
            'x': [c['name'] for c in columns]
        }]
    }


if __name__ == '__main__':

    if TERMINAL:
        app.run_server(debug=True)
