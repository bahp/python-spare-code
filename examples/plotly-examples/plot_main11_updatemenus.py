import plotly
import plotly.graph_objs as go

def get_color_set(color_set_id):
    if color_set_id == 1:
        marker_color = ['red', 'green', 'blue']
    elif color_set_id == 2:
        marker_color = ['black', 'blue', 'red']

    return [{'marker.color': [marker_color]}];


trace = go.Scatter(
    x=[0,1,1],
    y=[1,0,1],
    marker=dict(color=['green','black','red']),
    mode='markers'
)
updatemenus=list([
            dict(
                buttons=list([
                    dict(label = 'Color Set 1',
                         method = 'update',
                         args=get_color_set(1)
                    ),
                    dict(label = 'Color Set 2',
                         method = 'update',
                         args=get_color_set(2)
                    ),
                ]),
                direction = 'left',
                pad = {'r': 10, 't': 10},
                showactive = True,
                type = 'buttons',
                x = 0.1,
                xanchor = 'left',
                y = 1.1,
                yanchor = 'top'
            )
        ])
layout = go.Layout(
    title='Scatter Color Switcher',
    updatemenus = updatemenus
)

fig = go.Figure(data=[trace], layout=layout)
plotly.offline.plot(fig, filename='plot_main11_updatemenus.html')