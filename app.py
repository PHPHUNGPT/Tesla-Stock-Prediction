import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta, datetime

# Hàm để tải dữ liệu đã được tiền xử lý
@st.cache_data
def load_data():
    data = pd.read_csv('data_preprocessed.csv')
    
    # Loại bỏ các cột không mong muốn (nếu có)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    return data

# Hàm dự đoán cho nhiều ngày liên tiếp và tính toán ngày tháng cho mỗi dự đoán
def predict_multiple_days(model, data, scaler, days, column_order):
    predictions = []
    dates = []

    # Sắp xếp dữ liệu từ đầu đến cuối và chọn 30 ngày từ đầu
    current_data = data[column_order].iloc[:30].copy()
    
    # Tạo ngày bắt đầu từ các cột 'Day', 'Month', và 'Year'
    start_date = datetime(
        year=int(data['Year'].iloc[0]),
        month=int(data['Month'].iloc[0]),
        day=int(data['Day'].iloc[0])
    )

    for i in range(days):
        # Loại bỏ 'Price' nếu tồn tại trong dữ liệu
        if 'Price' in current_data.columns:
            current_data = current_data.drop(columns=['Price'])
        
        # Chuẩn hóa dữ liệu và dự đoán ngày tiếp theo
        data_scaled = scaler.transform(current_data)
        next_pred = model.predict(data_scaled)

        # Lưu dự đoán và ngày dự đoán
        predictions.append(next_pred[0])
        dates.append(start_date + timedelta(days=i + 1))  # Tính ngày tiếp theo

        # Cập nhật dữ liệu: thêm ngày dự đoán vào cuối và bỏ ngày đầu tiên
        next_row = current_data.iloc[-1].copy()
        next_row['Price'] = next_pred[0]  # thêm giá cho ngày mới
        current_data = pd.concat([pd.DataFrame([next_row]), current_data.iloc[:-1]], ignore_index=True)
    
    return predictions, dates

# Hàm nạp mô hình
def load_model(model_name):
    if model_name == 'Linear Regression':
        return joblib.load('linear_regression_model.pkl')
    elif model_name == 'Ridge Regression':
        return joblib.load('ridge_regression_model.pkl')
    elif model_name == 'MLP Regressor':
        return joblib.load('mlp_model.pkl')
    elif model_name == 'Stacking Model':
        return joblib.load('stacking_model.pkl')
    else:
        st.error("Vui lòng chọn một mô hình hợp lệ.")
        return None

# Thiết lập giao diện Streamlit
st.title('Dự Đoán Giá Cổ Phiếu Tesla')
st.write("Ứng dụng dự đoán giá cổ phiếu Tesla bằng các mô hình học máy.")

# Tải dữ liệu đã được tiền xử lý và danh sách tên cột
data = load_data()
column_order = joblib.load('columns.pkl')  # Tải danh sách cột từ file

# Tải scaler và sắp xếp dữ liệu đầu vào
scaler = joblib.load('scaler.pkl')
X = data[column_order]  # Đảm bảo sắp xếp đúng thứ tự cột đã lưu

# Chọn mô hình và số ngày dự đoán
model_name = st.selectbox('Chọn Mô Hình', ['Linear Regression', 'Ridge Regression', 'MLP Regressor', 'Stacking Model'])
model = load_model(model_name)
days = st.slider('Số ngày cần dự đoán', min_value=1, max_value=30, value=5)

# Dự đoán khi người dùng nhấn nút
if st.button('Dự Đoán'):
    predictions, dates = predict_multiple_days(model, X, scaler, days, column_order)
    
    if predictions is not None:
        st.write(f'Kết quả dự đoán cho {days} ngày tiếp theo từ ngày cuối cùng:')
        results = pd.DataFrame({'Ngày': dates, 'Giá Dự Đoán': predictions})
        st.write(results)
        
        # Trực quan hóa kết quả dự đoán
        plt.figure(figsize=(10, 5))
        plt.plot(dates, predictions, marker='o', color='r', label='Giá Dự Đoán')
        plt.xlabel('Ngày')
        plt.ylabel('Giá Cổ Phiếu')
        plt.title(f'Dự Đoán Giá Cổ Phiếu Tesla với {model_name} cho {days} ngày tiếp theo')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)
