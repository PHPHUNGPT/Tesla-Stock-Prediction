import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load dữ liệu
data = pd.read_csv('data_preprocessed.csv')
data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)
numeric_columns = data.select_dtypes(include=[np.number]).columns

# Khởi tạo ứng dụng Dash
app = dash.Dash(__name__)
app.title = "Dashboard Phân Tích Cổ Phiếu Tesla"

# Bố cục giao diện với các lựa chọn điều khiển
app.layout = html.Div(children=[
    html.H1("Dashboard Phân Tích Cổ Phiếu Tesla", style={'text-align': 'center', 'color': '#003366'}),

    # Điều khiển cho Scatter Plot
    html.Label("Chọn biến X cho Biểu Đồ Phân Tán:"),
    dcc.Dropdown(id='scatter-x', options=[{'label': col, 'value': col} for col in numeric_columns], value='Vol.'),
    html.Label("Chọn biến Y cho Biểu Đồ Phân Tán:"),
    dcc.Dropdown(id='scatter-y', options=[{'label': col, 'value': col} for col in numeric_columns], value='Price'),
    dcc.Graph(id='scatter-plot'),

    # Điều khiển cho Bar Plot
    html.Label("Chọn biến X cho Biểu Đồ Cột:"),
    dcc.Dropdown(id='bar-x', options=[{'label': col, 'value': col} for col in numeric_columns], value='Month'),
    dcc.Graph(id='bar-plot'),

    # Điều khiển cho Line Plot
    html.Label("Chọn biến X cho Biểu Đồ Đường:"),
    dcc.Dropdown(id='line-x', options=[{'label': col, 'value': col} for col in numeric_columns], value='Day'),
    dcc.Graph(id='line-plot'),

    # Biểu Đồ Tròn (Pie Chart) cố định
    dcc.Graph(id='pie-chart', figure=px.pie(data, names='Year', values='Vol.', 
                                             title='Biểu Đồ Tròn: Tỷ Lệ Khối Lượng Theo Năm', 
                                             color_discrete_sequence=px.colors.qualitative.Dark24)),

    # Điều khiển cho Box Plot
    html.Label("Chọn biến X cho Biểu Đồ Hộp:"),
    dcc.Dropdown(id='box-x', options=[{'label': col, 'value': col} for col in numeric_columns], value='Month'),
    dcc.Graph(id='box-plot'),

    # Ma trận tương quan cố định
    dcc.Graph(id='heatmap')
], style={'background-color': '#f0f8ff'})

# Cập nhật biểu đồ phân tán dựa trên lựa chọn
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-x', 'value'),
    Input('scatter-y', 'value')
)
def update_scatter(x_col, y_col):
    return px.scatter(data, x=x_col, y=y_col, color='Year', 
                      title=f'Biểu Đồ Phân Tán: {x_col} vs {y_col}',
                      color_continuous_scale=px.colors.sequential.Rainbow)

# Cập nhật biểu đồ cột dựa trên lựa chọn
@app.callback(
    Output('bar-plot', 'figure'),
    Input('bar-x', 'value')
)
def update_bar(x_col):
    return px.bar(data, x=x_col, y='Price', color='Year', 
                  title=f'Biểu Đồ Cột: {x_col} vs Giá',
                  barmode='group', color_discrete_sequence=px.colors.qualitative.Bold)

# Cập nhật biểu đồ đường dựa trên lựa chọn
@app.callback(
    Output('line-plot', 'figure'),
    Input('line-x', 'value')
)
def update_line(x_col):
    return px.line(data, x=x_col, y='Price', color='Year', 
                   title=f'Biểu Đồ Đường: {x_col} vs Giá',
                   color_discrete_sequence=px.colors.qualitative.Plotly)

# Cập nhật biểu đồ hộp dựa trên lựa chọn
@app.callback(
    Output('box-plot', 'figure'),
    Input('box-x', 'value')
)
def update_box(x_col):
    return px.box(data, x=x_col, y='Price', color='Year', 
                  title=f'Biểu Đồ Hộp: {x_col} vs Giá',
                  color_discrete_sequence=px.colors.qualitative.Vivid)

# Ma trận tương quan cố định
@app.callback(
    Output('heatmap', 'figure'),
    Input('scatter-x', 'value')  # Sử dụng bất kỳ đầu vào nào để kích hoạt
)
def update_heatmap(_):
    correlation_matrix = numeric_columns = data.select_dtypes(include=[np.number]).corr()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'
    ))
    fig.update_layout(title='Ma Trận Tương Quan')
    return fig

# Khởi chạy ứng dụng
if __name__ == '__main__':
    app.run_server(debug=True)
