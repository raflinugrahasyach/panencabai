import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import io
import base64
import os
import pickle
import datetime

# Set page config
st.set_page_config(
    page_title="Prediksi Panen Cabai",
    page_icon="🌶️",
    layout="wide"
)

# Initialize session state variables
if 'data_malang' not in st.session_state:
    st.session_state.data_malang = {}
    
    # Data Malang 2018
    st.session_state.data_malang[2018] = pd.DataFrame({
        'X1(CURAH HUJAN)': [389.20, 383.30, 260.50, 107.10, 161.70, 42.50, 14.30, 24.50, 34.70, 24.90, 62.00, 314.50],
        'X2(SUHU)': [25.40, 25.10, 26.10, 26.10, 25.20, 24.80, 24.40, 14.30, 24.50, 34.70, 24.90, 26.30],
        'X3(LUAS PANEN)': [1345, 1337, 1440, 1443, 1836, 1729, 1475, 1224, 1235, 1226, 626, 418],  
        'Y': [54680, 61003, 66760, 56227, 64898, 72915, 61818, 50128, 51703, 71415, 24970, 19797]  
    })
    
    # Data Malang 2019
    st.session_state.data_malang[2019] = pd.DataFrame({
        'X1(CURAH HUJAN)': [497.70, 263.30, 456.40, 194.00, 45.90, 27.90, 10.70, 24.40, 15.40, 24.30, 28.10, 70.70],
        'X2(SUHU)': [26.30, 26.50, 25.90, 26.10, 25.20, 24.20, 24.00, 10.70, 24.40, 15.40, 24.30, 21.40],
        'X3(LUAS PANEN)': [735, 1341, 1631, 1696, 1718, 1720, 1737, 1271, 641, 1205, 969, 845],  
        'Y': [17225, 44331, 59105, 74422, 68398, 80848, 94669, 71294, 43928, 74590, 33657, 31433]  
    })
    
    # Data Malang 2020
    st.session_state.data_malang[2020] = pd.DataFrame({
        'X1(CURAH HUJAN)': [333.20, 403.00, 316.30, 225.30, 224.30, 65.90, 27.10, 25.30, 19.90, 25.60, 118.30, 211.90],
        'X2(SUHU)': [26.70, 26.30, 26.10, 26.10, 25.20, 25.40, 25.20, 27.10, 25.30, 19.90, 25.60, 62.30],
        'X3(LUAS PANEN)': [813, 891, 1410, 1518, 1685, 694, 1639, 1337, 869, 1011, 224, 275],  
        'Y': [55438, 34461, 73339, 109417, 127267, 47328, 90489, 66846, 66953, 78647, 14665, 14473]  
    })
    
    # Data Malang 2021
    st.session_state.data_malang[2021] = pd.DataFrame({
        'X1(CURAH HUJAN)': [610.20, 301.20, 302.80, 160.50, 74.60, 127.40, 15.20, 25.30, 28.60, 25.70, 198.80, 430.00],
        'X2(SUHU)': [24.90, 25.30, 25.40, 25.70, 25.20, 25.30, 25.00, 15.20, 25.30, 28.60, 25.70, 169.30],
        'X3(LUAS PANEN)': [589, 496, 1119, 857, 682, 837, 1609, 732, 1421, 754, 723, 817],  
        'Y': [46804, 40857, 91761, 70551, 55821, 68567, 123734, 56399, 116502, 59524, 58080, 48025]  
    })
    
    # Data Malang 2022
    st.session_state.data_malang[2022] = pd.DataFrame({
        'X1(CURAH HUJAN)': [392.30, 333.30, 333.10, 186.70, 204.80, 177.80, 40.80, 25.40, 24.10, 25.60, 280.40, 485.70],
        'X2(SUHU)': [26.00, 24.60, 26.00, 26.10, 26.20, 25.40, 25.30, 40.80, 25.40, 24.10, 25.60, 68.10],
        'X3(LUAS PANEN)': [785, 1156, 768, 1800, 1782.50, 1617.50, 1263, 1342.50, 1003.40, 964.75, 1243.65, 1046.10],  
        'Y': [140155, 77812, 48133, 93932, 92549, 93874, 71467, 76721, 48753.80, 45802.50, 64340, 20798]  
    })

if 'data_lumajang' not in st.session_state:
    st.session_state.data_lumajang = {}
    
    # Data Lumajang 2018
    st.session_state.data_lumajang[2018] = pd.DataFrame({
        'X1(CURAH HUJAN)': [351.80, 316.80, 221.90, 101.00, 167.30, 52.40, 25.40, 80.00, 40.00, 93.20, 389.00, 252.70],
        'X2(SUHU)': [26.50, 26.00, 27.20, 27.00, 26.30, 26.00, 25.60, 25.90, 26.30, 27.40, 27.20, 27.80],
        'X3(LUAS PANEN)': [69, 37, 49, 43, 16, 25, 15, 21, 51, 55, 104, 826],  
        'Y': [3814.40, 6878.00, 6758.00, 4890.00, 3800.00, 11650.00, 21790.00, 28400.00, 27836.00, 36910.00, 31050.00, 21365.00]  
    })
    
    # Data Lumajang 2019
    st.session_state.data_lumajang[2019] = pd.DataFrame({
        'X1(CURAH HUJAN)': [381.30, 176.20, 442.90, 136.90, 45.40, 36.70, 18.20, 33.50, 38.60, 49.30, 79.40, 214.30],
        'X2(SUHU)': [27.40, 27.40, 26.90, 27.10, 26.30, 25.30, 25.20, 25.80, 25.70, 27.80, 27.50, 27.50],
        'X3(LUAS PANEN)': [9, 83, 74, 11, 19, 15, 7, 13, 14, 240, 108, 13],  
        'Y': [3616.00, 8279.00, 6967.80, 5174.00, 3653.00, 19730.50, 19048.43, 13467.80, 15129.00, 28488.00, 25209.00, 13714.00]  
    })
    
    # Data Lumajang 2020
    st.session_state.data_lumajang[2020] = pd.DataFrame({
        'X1(CURAH HUJAN)': [253.80, 294.30, 326.30, 197.50, 236.30, 84.30, 35.10, 27.50, 85.20, 191.70, 251.30, 406.60],
        'X2(SUHU)': [27.90, 27.30, 27.10, 27.10, 26.30, 26.60, 26.50, 26.70, 27.20, 27.60, 27.70, 27.10],
        'X3(LUAS PANEN)': [351, 278, 25, 13, 21, 12, 13, 9, 20, 236, 17, 1405],  
        'Y': [16008, 8880, 3105, 2690, 2456, 5085, 9470, 29231, 32002, 37643, 40349, 24650]  
    })
    
    # Data Lumajang 2021
    st.session_state.data_lumajang[2021] = pd.DataFrame({
        'X1(CURAH HUJAN)': [540.40, 267.00, 323.80, 146.20, 70.80, 175.40, 23.40, 57.80, 245.90, 317.60, 531.30, 593.70],
        'X2(SUHU)': [26.00, 26.20, 26.50, 26.70, 26.30, 26.50, 26.40, 26.80, 27.30, 27.80, 26.60, 27.20],
        'X3(LUAS PANEN)': [573, 3, 9, 27, 15, 8, 31, 20, 229, 24, 647, 437],  
        'Y': [4649, 1870, 2028, 3070, 2479, 2294, 3956, 23625, 21829, 8921, 20837.80, 30741.70]  
    })
    
    # Data Lumajang 2022
    st.session_state.data_lumajang[2022] = pd.DataFrame({
        'X1(CURAH HUJAN)': [331.40, 274.30, 323.00, 186.60, 209.30, 216.10, 51.50, 48.30, 109.90, 400.20, 598.40, 360.50],
        'X2(SUHU)': [27.10, 25.60, 27.10, 27.10, 27.40, 26.70, 26.60, 26.90, 27.20, 26.80, 26.50, 27.20],
        'X3(LUAS PANEN)': [20, 23, 26, 44, 23, 8, 7, 29, 251, 8, 15, 288],  
        'Y': [3336, 3537, 5115, 4952.10, 4167, 3992, 16831, 17072, 15251.75, 9992, 7959, 8140.80]  
    })

if 'models_malang' not in st.session_state:
    st.session_state.models_malang = {}

if 'models_lumajang' not in st.session_state:
    st.session_state.models_lumajang = {}

if 'rmse_malang' not in st.session_state:
    st.session_state.rmse_malang = {}

if 'rmse_lumajang' not in st.session_state:
    st.session_state.rmse_lumajang = {}

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Function to save models
def save_model(city, year, model):
    if city == "Malang":
        st.session_state.models_malang[year] = model
    else:
        st.session_state.models_lumajang[year] = model
    
    # Calculate RMSE for next year if data exists
    next_year = year + 1
    if next_year <= 2022:
        rmse = evaluate_model(city, year, next_year)
        if rmse is not None:
            if city == "Malang":
                st.session_state.rmse_malang[next_year] = rmse
            else:
                st.session_state.rmse_lumajang[next_year] = rmse


# Function to train models
def train_model(city, year):
    if city == "Malang":
        data = st.session_state.data_malang[year]
    else:
        data = st.session_state.data_lumajang[year]
    
    if len(data) < 3:  # Need at least 3 data points for meaningful regression
        return None
    
    X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
    y = data['Y']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model


def evaluate_model(city, model_year, eval_year):
    """
    Evaluates a model by using model_year's model to predict eval_year's data
    and calculates RMSE
    """
    if city == "Malang":
        if model_year not in st.session_state.models_malang:
            return None
        if eval_year not in st.session_state.data_malang:
            return None
        model = st.session_state.models_malang[model_year]
        eval_data = st.session_state.data_malang[eval_year]
    else:
        if model_year not in st.session_state.models_lumajang:
            return None
        if eval_year not in st.session_state.data_lumajang:
            return None
        model = st.session_state.models_lumajang[model_year]
        eval_data = st.session_state.data_lumajang[eval_year]
    
    X_eval = eval_data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
    y_actual = eval_data['Y']
    
    y_pred = model.predict(X_eval)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    return rmse


# Function to make predictions
def predict(city, model_year, curah_hujan, suhu, luas_panen):
    if city == "Malang":
        if model_year not in st.session_state.models_malang:
            return None
        model = st.session_state.models_malang[model_year]
    else:
        if model_year not in st.session_state.models_lumajang:
            return None
        model = st.session_state.models_lumajang[model_year]
    
    prediction = model.predict([[curah_hujan, suhu, luas_panen]])
    return prediction[0]

# Function to download data as CSV
def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Main app header
st.title("🌶️ Sistem Prediksi Panen Cabai")
st.markdown("Prediksi hasil panen cabai menggunakan regresi linear berganda untuk Kabupaten Malang dan Lumajang")

st.sidebar.title("Menu Navigasi")
dashboard_btn = st.sidebar.button("Dashboard", use_container_width=True)
data_btn = st.sidebar.button("Data Aktual", use_container_width=True)
predict_btn = st.sidebar.button("Prediksi", use_container_width=True)
results_btn = st.sidebar.button("Hasil", use_container_width=True)

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

if dashboard_btn:
    st.session_state.current_page = "Dashboard"
if data_btn:
    st.session_state.current_page = "Data Aktual"
if predict_btn:
    st.session_state.current_page = "Prediksi"
if results_btn:
    st.session_state.current_page = "Hasil"

# Dashboard page
if st.session_state.current_page == "Dashboard":
    st.header("Dashboard Prediksi Panen Cabai")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kabupaten Malang")
        city_data = pd.DataFrame()
        
        for year in range(2018, 2023):
            if not st.session_state.data_malang[year].empty:
                temp_df = st.session_state.data_malang[year].copy()
                temp_df['Tahun'] = year
                temp_df['Bulan'] = range(1, len(temp_df) + 1)
                city_data = pd.concat([city_data, temp_df])
        
        if not city_data.empty:
            # Show yearly production trend
            st.markdown("### Trend Produksi Tahunan")
            yearly_production = city_data.groupby('Tahun')['Y'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(yearly_production['Tahun'], yearly_production['Y'], color='green')
            ax.set_xlabel('Tahun')
            ax.set_ylabel('Total Produksi')
            ax.set_title('Total Produksi Cabai per Tahun - Malang')
            st.pyplot(fig)
            
            # Show RMSE comparison if models exist
            if st.session_state.rmse_malang:
                st.markdown("### Akurasi Model (RMSE)")
                rmse_data = pd.DataFrame({
                    'Tahun': list(st.session_state.rmse_malang.keys()),
                    'RMSE': list(st.session_state.rmse_malang.values())
                })
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(rmse_data['Tahun'], rmse_data['RMSE'], color='orange')
                ax.set_xlabel('Tahun')
                ax.set_ylabel('RMSE')
                ax.set_title('RMSE per Tahun - Malang')
                st.pyplot(fig)
        else:
            st.info("Belum ada data untuk Malang. Silakan tambahkan data di menu 'Data Aktual'.")
    
    with col2:
        st.subheader("Kota Lumajang")
        city_data = pd.DataFrame()
        
        for year in range(2018, 2023):
            if not st.session_state.data_lumajang[year].empty:
                temp_df = st.session_state.data_lumajang[year].copy()
                temp_df['Tahun'] = year
                temp_df['Bulan'] = range(1, len(temp_df) + 1)
                city_data = pd.concat([city_data, temp_df])
        
        if not city_data.empty:
            # Show yearly production trend
            st.markdown("### Trend Produksi Tahunan")
            yearly_production = city_data.groupby('Tahun')['Y'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(yearly_production['Tahun'], yearly_production['Y'], color='blue')
            ax.set_xlabel('Tahun')
            ax.set_ylabel('Total Produksi')
            ax.set_title('Total Produksi Cabai per Tahun - Lumajang')
            st.pyplot(fig)
            
            # Show RMSE comparison if models exist
            if st.session_state.rmse_lumajang:
                st.markdown("### Akurasi Model (RMSE)")
                rmse_data = pd.DataFrame({
                    'Tahun': list(st.session_state.rmse_lumajang.keys()),
                    'RMSE': list(st.session_state.rmse_lumajang.values())
                })
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(rmse_data['Tahun'], rmse_data['RMSE'], color='orange')
                ax.set_xlabel('Tahun')
                ax.set_ylabel('RMSE')
                ax.set_title('RMSE per Tahun - Lumajang')
                st.pyplot(fig)
        else:
            st.info("Belum ada data untuk Lumajang. Silakan tambahkan data di menu 'Data Aktual'.")
    
    # Show comparison between cities
    st.header("Perbandingan Produksi Antar Kota")
    
    malang_data = pd.DataFrame()
    lumajang_data = pd.DataFrame()
    
    for year in range(2018, 2023):
        if not st.session_state.data_malang[year].empty:
            temp_df = st.session_state.data_malang[year].copy()
            temp_df['Tahun'] = year
            malang_data = pd.concat([malang_data, temp_df])
        
        if not st.session_state.data_lumajang[year].empty:
            temp_df = st.session_state.data_lumajang[year].copy()
            temp_df['Tahun'] = year
            lumajang_data = pd.concat([lumajang_data, temp_df])
    
    if not malang_data.empty and not lumajang_data.empty:
        malang_yearly = malang_data.groupby('Tahun')['Y'].sum().reset_index()
        malang_yearly['Kota'] = 'Malang'
        
        lumajang_yearly = lumajang_data.groupby('Tahun')['Y'].sum().reset_index()
        lumajang_yearly['Kota'] = 'Lumajang'
        
        combined_data = pd.concat([malang_yearly, lumajang_yearly])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Tahun', y='Y', hue='Kota', data=combined_data, ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Total Produksi')
        ax.set_title('Perbandingan Produksi Cabai Antar Kota')
        st.pyplot(fig)
    else:
        st.info("Data tidak cukup untuk perbandingan antar kota.")

# Data Aktual page
elif st.session_state.current_page == "Data Aktual":
    st.header("Data Aktual Panen Cabai")
    
    tab1, tab2 = st.tabs(["Malang", "Lumajang"])
    
    with tab1:
        st.subheader("Data Kabupaten Malang")
        year_malang = st.selectbox("Pilih Tahun (Malang)", list(range(2018, 2023)), key='year_malang')
        
        # CRUD operations
        st.markdown("### Manage Data")
        crud_tabs = st.tabs(["Lihat", "Tambah", "Edit", "Hapus"])
        
        with crud_tabs[0]:  # View
            if not st.session_state.data_malang[year_malang].empty:
                st.dataframe(st.session_state.data_malang[year_malang])
                st.markdown(get_download_link(st.session_state.data_malang[year_malang], f"malang_{year_malang}.csv"), unsafe_allow_html=True)
            else:
                st.info(f"Belum ada data untuk tahun {year_malang}.")
        
        with crud_tabs[1]:  # Add
            st.markdown("### Tambah Data Baru")
            with st.form("add_data_malang"):
                month = st.selectbox("Bulan", list(range(1, 13)))
                curah_hujan = st.number_input("Curah Hujan", min_value=0.0, format="%.2f")
                suhu = st.number_input("Suhu", min_value=0.0, format="%.2f")
                luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, format="%.3f")
                hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, format="%.3f")
                
                submit_button = st.form_submit_button("Tambah Data")
                if submit_button:
                    # Check if month already exists
                    if len(st.session_state.data_malang[year_malang]) >= month:
                        st.error(f"Data untuk bulan {month} sudah ada. Gunakan Edit untuk mengubah data.")
                    else:
                        new_data = pd.DataFrame({
                            'X1(CURAH HUJAN)': [curah_hujan],
                            'X2(SUHU)': [suhu],
                            'X3(LUAS PANEN)': [luas_panen],
                            'Y': [hasil_panen]
                        })
                        st.session_state.data_malang[year_malang] = pd.concat([st.session_state.data_malang[year_malang], new_data], ignore_index=True)
                        st.success("Data berhasil ditambahkan!")
                        
                        # Train model after adding data
                        model = train_model("Malang", year_malang)
                        if model is not None:
                            save_model("Malang", year_malang, model)
                            next_year = year_malang + 1
                            if next_year in st.session_state.rmse_malang:
                                st.success(f"Model untuk tahun {year_malang} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                            else:
                                st.success(f"Model untuk tahun {year_malang} berhasil dilatih!")
        
        with crud_tabs[2]:  # Edit
            st.markdown("### Edit Data")
            if st.session_state.data_malang[year_malang].empty:
                st.info(f"Belum ada data untuk tahun {year_malang}.")
            else:
                month_to_edit = st.selectbox("Pilih Bulan untuk Edit", list(range(1, len(st.session_state.data_malang[year_malang]) + 1)), key='edit_month_malang')
                
                idx = month_to_edit - 1
                if idx < len(st.session_state.data_malang[year_malang]):
                    with st.form("edit_data_malang"):
                        curah_hujan = st.number_input("Curah Hujan", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['X1(CURAH HUJAN)']), format="%.2f")
                        suhu = st.number_input("Suhu", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['X2(SUHU)']), format="%.2f")
                        luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['X3(LUAS PANEN)']), format="%.3f")
                        hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['Y']), format="%.3f")
                        
                        submit_button = st.form_submit_button("Update Data")
                        if submit_button:
                            st.session_state.data_malang[year_malang].iloc[idx] = [curah_hujan, suhu, luas_panen, hasil_panen]
                            st.success("Data berhasil diupdate!")
                            
                            # Retrain model after editing data
                            model = train_model("Malang", year_malang)
                            if model is not None:
                                save_model("Malang", year_malang, model)
                                next_year = year_malang + 1
                                if next_year in st.session_state.rmse_malang:
                                    st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                                else:
                                    st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang!")
        
        with crud_tabs[3]:  # Delete
            st.markdown("### Hapus Data")
            if st.session_state.data_malang[year_malang].empty:
                st.info(f"Belum ada data untuk tahun {year_malang}.")
            else:
                month_to_delete = st.selectbox("Pilih Bulan untuk Hapus", list(range(1, len(st.session_state.data_malang[year_malang]) + 1)), key='delete_month_malang')
                
                if st.button("Hapus Data"):
                    idx = month_to_delete - 1
                    if idx < len(st.session_state.data_malang[year_malang]):
                        st.session_state.data_malang[year_malang] = st.session_state.data_malang[year_malang].drop(st.session_state.data_malang[year_malang].index[idx]).reset_index(drop=True)
                        st.success("Data berhasil dihapus!")
                        
                        # Retrain model after deleting data
                        model = train_model("Malang", year_malang)
                        if model is not None:
                            save_model("Malang", year_malang, model)
                            next_year = year_malang + 1
                            if next_year in st.session_state.rmse_malang:
                                st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                            else:
                                st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang!")
        
        # Train all models button
        if st.button("Latih Semua Model untuk Malang"):
            success_count = 0
            for year in range(2018, 2022):  # Only train models for 2018-2021 (to predict 2019-2022)
                if not st.session_state.data_malang[year].empty and len(st.session_state.data_malang[year]) >= 3:
                    model = train_model("Malang", year)
                    if model is not None:
                        save_model("Malang", year, model)
                        success_count += 1
                        next_year = year + 1
                        if next_year in st.session_state.rmse_malang:
                            st.success(f"Model untuk tahun {year} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                        else:
                            st.success(f"Model untuk tahun {year} berhasil dilatih!")
            
            if success_count > 0:
                st.success(f"{success_count} model berhasil dilatih untuk Malang!")
            else:
                st.warning("Tidak ada model yang berhasil dilatih. Pastikan data cukup (minimal 3 baris per tahun).")

    
    with tab2:
        st.subheader("Data Kota Lumajang")
        year_lumajang = st.selectbox("Pilih Tahun (Lumajang)", list(range(2018, 2023)), key='year_lumajang')
        
        # CRUD operations
        st.markdown("### Manage Data")
        crud_tabs = st.tabs(["Lihat", "Tambah", "Edit", "Hapus"])
        
        with crud_tabs[0]:  # View
            if not st.session_state.data_lumajang[year_lumajang].empty:
                st.dataframe(st.session_state.data_lumajang[year_lumajang])
                st.markdown(get_download_link(st.session_state.data_lumajang[year_lumajang], f"lumajang_{year_lumajang}.csv"), unsafe_allow_html=True)
            else:
                st.info(f"Belum ada data untuk tahun {year_lumajang}.")
        
        with crud_tabs[1]:  # Add
            st.markdown("### Tambah Data Baru")
            with st.form("add_data_lumajang"):
                month = st.selectbox("Bulan", list(range(1, 13)), key='month_add_lumajang')
                curah_hujan = st.number_input("Curah Hujan", min_value=0.0, format="%.2f", key='ch_add_lumajang')
                suhu = st.number_input("Suhu", min_value=0.0, format="%.2f", key='suhu_add_lumajang')
                luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, format="%.3f", key='luas_add_lumajang')
                hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, format="%.3f", key='hasil_add_lumajang')
                
                submit_button = st.form_submit_button("Tambah Data")
                if submit_button:
                    # Check if month already exists
                    if len(st.session_state.data_lumajang[year_lumajang]) >= month:
                        st.error(f"Data untuk bulan {month} sudah ada. Gunakan Edit untuk mengubah data.")
                    else:
                        new_data = pd.DataFrame({
                            'X1(CURAH HUJAN)': [curah_hujan],
                            'X2(SUHU)': [suhu],
                            'X3(LUAS PANEN)': [luas_panen],
                            'Y': [hasil_panen]
                        })
                        st.session_state.data_lumajang[year_lumajang] = pd.concat([st.session_state.data_lumajang[year_lumajang], new_data], ignore_index=True)
                        st.success("Data berhasil ditambahkan!")
                        
                        # Train model after adding data
                        model = train_model("Lumajang", year_lumajang)
                        if model is not None:
                            save_model("Lumajang", year_lumajang, model)
                            next_year = year_lumajang + 1
                            if next_year in st.session_state.rmse_lumajang:
                                st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                            else:
                                st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih!")
        
        with crud_tabs[2]:  # Edit
            st.markdown("### Edit Data")
            if st.session_state.data_lumajang[year_lumajang].empty:
                st.info(f"Belum ada data untuk tahun {year_lumajang}.")
            else:
                month_to_edit = st.selectbox("Pilih Bulan untuk Edit", list(range(1, len(st.session_state.data_lumajang[year_lumajang]) + 1)), key='edit_month_lumajang')
                
                idx = month_to_edit - 1
                if idx < len(st.session_state.data_lumajang[year_lumajang]):
                    with st.form("edit_data_lumajang"):
                        curah_hujan = st.number_input("Curah Hujan", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['X1(CURAH HUJAN)']), format="%.2f", key='ch_edit_lumajang')
                        suhu = st.number_input("Suhu", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['X2(SUHU)']), format="%.2f", key='suhu_edit_lumajang')
                        luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['X3(LUAS PANEN)']), format="%.3f", key='luas_edit_lumajang')
                        hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['Y']), format="%.3f", key='hasil_edit_lumajang')
                        
                        submit_button = st.form_submit_button("Update Data")
                        if submit_button:
                            st.session_state.data_lumajang[year_lumajang].iloc[idx] = [curah_hujan, suhu, luas_panen, hasil_panen]
                            st.success("Data berhasil diupdate!")
                            
                            # Retrain model after editing data
                            model = train_model("Lumajang", year_lumajang)
                            if model is not None:
                                save_model("Lumajang", year_lumajang, model)
                                next_year = year_lumajang + 1
                                if next_year in st.session_state.rmse_lumajang:
                                    st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                                else:
                                    st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang!")
        
        with crud_tabs[3]:  # Delete
            st.markdown("### Hapus Data")
            if st.session_state.data_lumajang[year_lumajang].empty:
                st.info(f"Belum ada data untuk tahun {year_lumajang}.")
            else:
                month_to_delete = st.selectbox("Pilih Bulan untuk Hapus", list(range(1, len(st.session_state.data_lumajang[year_lumajang]) + 1)), key='delete_month_lumajang')
                
                if st.button("Hapus Data", key="delete_btn_lumajang"):
                    idx = month_to_delete - 1
                    if idx < len(st.session_state.data_lumajang[year_lumajang]):
                        st.session_state.data_lumajang[year_lumajang] = st.session_state.data_lumajang[year_lumajang].drop(st.session_state.data_lumajang[year_lumajang].index[idx]).reset_index(drop=True)
                        st.success("Data berhasil dihapus!")
                        
                        # Retrain model after deleting data
                        model = train_model("Lumajang", year_lumajang)
                        if model is not None:
                            save_model("Lumajang", year_lumajang, model)
                            next_year = year_lumajang + 1
                            if next_year in st.session_state.rmse_lumajang:
                                st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                            else:
                                st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang!")
        
        # Train all models button
        if st.button("Latih Semua Model untuk Lumajang"):
            success_count = 0
            for year in range(2018, 2022):  # Only train models for 2018-2021 (to predict 2019-2022)
                if not st.session_state.data_lumajang[year].empty and len(st.session_state.data_lumajang[year]) >= 3:
                    model = train_model("Lumajang", year)
                    if model is not None:
                        save_model("Lumajang", year, model)
                        success_count += 1
                        next_year = year + 1
                        if next_year in st.session_state.rmse_lumajang:
                            st.success(f"Model untuk tahun {year} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                        else:
                            st.success(f"Model untuk tahun {year} berhasil dilatih!")
            
            if success_count > 0:
                st.success(f"{success_count} model berhasil dilatih untuk Lumajang!")
            else:
                st.warning("Tidak ada model yang berhasil dilatih. Pastikan data cukup (minimal 3 baris per tahun).")

# Prediction page
elif st.session_state.current_page == "Prediksi":
    st.header("Prediksi Panen Cabai")
    
    # Add tabs for equation table and prediction forms
    pred_tabs = st.tabs(["Persamaan Regresi", "Form Prediksi"])
    
    with pred_tabs[0]:
        st.subheader("Persamaan Regresi Model per Kota dan Tahun")
        
        # Create dataframe to store all regression equations
        equation_data = []
        
        # Get equations for Malang models
        for year, model in st.session_state.models_malang.items():
            coef = model.coef_
            intercept = model.intercept_
            equation = f"Y = {intercept:.2f} + {coef[0]:.2f}*X₁ + {coef[1]:.2f}*X₂ + {coef[2]:.2f}*X₃"
            equation_data.append({
                "Kota": "Malang",
                "Tahun Model": year,
                "Persamaan Regresi": equation,
                "R²": model.score(st.session_state.data_malang[year][['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']], 
                                  st.session_state.data_malang[year]['Y'])
            })
        
        # Get equations for Lumajang models
        for year, model in st.session_state.models_lumajang.items():
            coef = model.coef_
            intercept = model.intercept_
            equation = f"Y = {intercept:.2f} + {coef[0]:.2f}*X₁ + {coef[1]:.2f}*X₂ + {coef[2]:.2f}*X₃"
            equation_data.append({
                "Kota": "Lumajang",
                "Tahun Model": year,
                "Persamaan Regresi": equation,
                "R²": model.score(st.session_state.data_lumajang[year][['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']], 
                                  st.session_state.data_lumajang[year]['Y'])
            })
        
        # Convert to DataFrame and display
        if equation_data:
            equation_df = pd.DataFrame(equation_data)
            st.dataframe(equation_df.sort_values(by=['Kota', 'Tahun Model']), use_container_width=True)
            
            # Display variable meanings
            st.markdown("""
            ### Keterangan Variabel:
            - **X₁**: Curah Hujan
            - **X₂**: Suhu
            - **X₃**: Luas Panen
            - **Y**: Hasil Panen (ton)
            """)
        else:
            st.warning("Belum ada model terlatih. Silakan latih model terlebih dahulu di menu 'Data Aktual'.")
        
        # Display RMSE table
        st.subheader("Tabel RMSE Model")
        
        rmse_data = []
        
        # RMSE for Malang
        for year, rmse in st.session_state.rmse_malang.items():
            rmse_data.append({
                "Kota": "Malang",
                "Tahun": year,
                "RMSE": rmse
            })
        
        # RMSE for Lumajang
        for year, rmse in st.session_state.rmse_lumajang.items():
            rmse_data.append({
                "Kota": "Lumajang",
                "Tahun": year,
                "RMSE": rmse
            })
        
        if rmse_data:
            rmse_df = pd.DataFrame(rmse_data)
            st.dataframe(rmse_df.sort_values(by=['Kota', 'Tahun']), use_container_width=True)
        else:
            st.info("Belum ada data RMSE. RMSE akan muncul setelah model dilatih dan divalidasi.")
    
    with pred_tabs[1]:
        st.subheader("Form Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox("Pilih Kota", ["Malang", "Lumajang"])
        
        with col2:
            model_years = []
            if city == "Malang":
                model_years = list(st.session_state.models_malang.keys())
            else:
                model_years = list(st.session_state.models_lumajang.keys())

            if not model_years:
                st.error(f"Belum ada model terlatih untuk kota {city}. Harap latih model terlebih dahulu di menu Data Aktual.")
                st.stop()

            model_year = st.selectbox("Pilih Tahun Model", sorted(model_years))
            prediction_year = st.selectbox("Pilih Tahun Prediksi", 
                                        [year for year in range(2019, 2026) if year > model_year], 
                                        format_func=lambda x: f"{x} (menggunakan model {model_year})")
        
        # Show the model equation for the selected model
        if city == "Malang" and model_year in st.session_state.models_malang:
            model = st.session_state.models_malang[model_year]
            coef = model.coef_
            intercept = model.intercept_
            st.markdown(f"""
            ### Model Regresi yang Digunakan:
            **Y = {intercept:.2f} + {coef[0]:.2f}*X₁ + {coef[1]:.2f}*X₂ + {coef[2]:.2f}*X₃**
            
            Dimana:
            - X₁: Curah Hujan
            - X₂: Suhu
            - X₃: Luas Panen
            """)
        elif city == "Lumajang" and model_year in st.session_state.models_lumajang:
            model = st.session_state.models_lumajang[model_year]
            coef = model.coef_
            intercept = model.intercept_
            st.markdown(f"""
            ### Model Regresi yang Digunakan:
            **Y = {intercept:.2f} + {coef[0]:.2f}*X₁ + {coef[1]:.2f}*X₂ + {coef[2]:.2f}*X₃**
            
            Dimana:
            - X₁: Curah Hujan
            - X₂: Suhu
            - X₃: Luas Panen
            """)
        
        st.subheader("Input Parameter")
        
        # Input form for prediction
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                curah_hujan = st.number_input("Curah Hujan", min_value=0.0, value=100.0, step=10.0, format="%.2f")
            
            with col2:
                suhu = st.number_input("Suhu (°C)", min_value=15.0, max_value=35.0, value=26.0, step=0.1, format="%.1f")
            
            with col3:
                luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, value=500.0, step=50.0, format="%.2f")
            
            submit_pred = st.form_submit_button("Prediksi Hasil Panen")
            
            if submit_pred:
                prediction = predict(city, model_year, curah_hujan, suhu, luas_panen)
                
                if prediction is not None:
                    # Store prediction in history
                    now = datetime.datetime.now()
                    st.session_state.prediction_history.append({
                        "timestamp": now,
                        "city": city,
                        "model_year": model_year,
                        "prediction_year": prediction_year,
                        "curah_hujan": curah_hujan,
                        "suhu": suhu,
                        "luas_panen": luas_panen,
                        "hasil_panen": prediction
                    })
                    
                    st.success(f"Prediksi Hasil Panen untuk {city} Tahun {prediction_year}: **{prediction:.2f} ton**")
                    
                    # Show more details after prediction
                    st.write("#### Detail Prediksi:")
                    st.write(f"- Menggunakan model tahun: {model_year}")
                    st.write(f"- Curah Hujan: {curah_hujan} mm")
                    st.write(f"- Suhu: {suhu} °C")
                    st.write(f"- Luas Panen: {luas_panen} ha")
                    
                    if city == "Malang" and model_year in st.session_state.models_malang:
                        model = st.session_state.models_malang[model_year]
                        coef = model.coef_
                        intercept = model.intercept_
                        st.write("#### Perhitungan:")
                        st.write(f"Y = {intercept:.2f} + ({coef[0]:.2f} × {curah_hujan}) + ({coef[1]:.2f} × {suhu}) + ({coef[2]:.2f} × {luas_panen})")
                        st.write(f"Y = {intercept:.2f} + {coef[0] * curah_hujan:.2f} + {coef[1] * suhu:.2f} + {coef[2] * luas_panen:.2f}")
                        st.write(f"Y = {prediction:.2f} ton")
                    
                    elif city == "Lumajang" and model_year in st.session_state.models_lumajang:
                        model = st.session_state.models_lumajang[model_year]
                        coef = model.coef_
                        intercept = model.intercept_
                        st.write("#### Perhitungan:")
                        st.write(f"Y = {intercept:.2f} + ({coef[0]:.2f} × {curah_hujan}) + ({coef[1]:.2f} × {suhu}) + ({coef[2]:.2f} × {luas_panen})")
                        st.write(f"Y = {intercept:.2f} + {coef[0] * curah_hujan:.2f} + {coef[1] * suhu:.2f} + {coef[2] * luas_panen:.2f}")
                        st.write(f"Y = {prediction:.2f} ton")
                else:
                    st.error("Prediksi gagal. Pastikan model telah terlatih dengan benar.")

    
    # ================= Tabel Penolong ==================
    st.markdown("### Tabel Penolong")
    # Ambil data sesuai kota dan tahun model
    if city == "Malang":
        df = st.session_state.data_malang[model_year].copy()
    else:
        df = st.session_state.data_lumajang[model_year].copy()

    # Hitung kolom-kolom penolong
    df["X12"]   = df["X1(CURAH HUJAN)"]       ** 2
    df["X22"]   = df["X2(SUHU)"]              ** 2
    df["X32"]   = df["X3(LUAS PANEN)"]        ** 2
    df["Y2"]    = df["Y"]                     ** 2
    df["X1Y"]   = df["X1(CURAH HUJAN)"]       * df["Y"]
    df["X2Y"]   = df["X2(SUHU)"]              * df["Y"]
    df["X3Y"]   = df["X3(LUAS PANEN)"]        * df["Y"]
    df["X1X2"]  = df["X1(CURAH HUJAN)"]       * df["X2(SUHU)"]
    df["X1X3"]  = df["X1(CURAH HUJAN)"]       * df["X3(LUAS PANEN)"]
    df["X2X3"]  = df["X2(SUHU)"]              * df["X3(LUAS PANEN)"]

    # Atur index menjadi label bulan
    # Cek jumlah data untuk menentukan label bulan yang sesuai
    month_labels = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    if len(df) <= len(month_labels):
        df.index = month_labels[:len(df)]
    else:
        # Jika data lebih dari 12 bulan, gunakan indeks numerik
        df.index = [f"Data {i+1}" for i in range(len(df))]

    # Tampilkan tabel penolong
    st.dataframe(df)
    # ============================================

    st.markdown("### Input Data Prediksi")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            curah_hujan = st.number_input("Curah Hujan", min_value=0.0, format="%.2f")
        
        with col2:
            suhu = st.number_input("Suhu", min_value=0.0, format="%.2f")
        
        with col3:
            luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, format="%.3f")
        
        submit_button = st.form_submit_button("Prediksi")
        
        if submit_button:
            result = predict(city, model_year, curah_hujan, suhu, luas_panen)
            
            if result is not None:
                st.success(f"Prediksi hasil panen cabai: {result:.3f} ton")
                
                # Add to prediction history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.prediction_history.append({
                    "timestamp": timestamp,
                    "city": city,
                    "year_model": model_year,
                    "curah_hujan": curah_hujan,
                    "suhu": suhu,
                    "luas_panen": luas_panen,
                    "prediction": result
                })
                
                # Show model details
                st.markdown("### Detail Model")

                if city == "Malang":
                    model = st.session_state.models_malang[model_year]
                    next_year = model_year + 1
                    rmse = st.session_state.rmse_malang.get(next_year, None)
                else:
                    model = st.session_state.models_lumajang[model_year]
                    next_year = model_year + 1
                    rmse = st.session_state.rmse_lumajang.get(next_year, None)

                coefficients = model.coef_
                intercept = model.intercept_
                
                st.markdown(f"**Persamaan Model:**")
                st.markdown(f"Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃")
                if rmse is not None:
                    st.markdown(f"**RMSE untuk prediksi tahun {next_year}:** {rmse:.4f}")
                else:
                    st.markdown("**RMSE:** Tidak tersedia")

                # Show visualization if we have data for that year
                if city == "Malang" and not st.session_state.data_malang[model_year].empty:
                    data = st.session_state.data_malang[model_year]
                    X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
                    y_actual = data['Y']
                    y_pred = model.predict(X)
                    
                    # Plot actual vs predicted
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(1, len(y_actual) + 1), y_actual, 'b-', label='Aktual')
                    ax.plot(range(1, len(y_pred) + 1), y_pred, 'r--', label='Prediksi')
                    ax.set_xlabel('Bulan')
                    ax.set_ylabel('Hasil Panen (ton)')
                    ax.set_title(f'Perbandingan Hasil Aktual vs Prediksi - {city} {model_year}')
                    ax.legend()
                    st.pyplot(fig)
                
                elif city == "Lumajang" and not st.session_state.data_lumajang[model_year].empty:
                    data = st.session_state.data_lumajang[model_year]
                    X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
                    y_actual = data['Y']
                    y_pred = model.predict(X)
                    
                    # Plot actual vs predicted
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(1, len(y_actual) + 1), y_actual, 'b-', label='Aktual')
                    ax.plot(range(1, len(y_pred) + 1), y_pred, 'r--', label='Prediksi')
                    ax.set_xlabel('Bulan')
                    ax.set_ylabel('Hasil Panen (ton)')
                    ax.set_title(f'Perbandingan Hasil Aktual vs Prediksi - {city} {model_year}')
                    ax.legend()
                    st.pyplot(fig)
                    
# Results page
elif st.session_state.current_page == "Hasil":
    st.header("Hasil Prediksi")
    
    if not st.session_state.prediction_history:
        st.info("Belum ada riwayat prediksi. Silakan buat prediksi terlebih dahulu di menu Prediksi.")
    else:
        # Convert prediction history to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['timestamp_str'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Sort by most recent first
        history_df = history_df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        # Display as table
        st.dataframe(
            history_df[['timestamp_str', 'city', 'model_year', 'prediction_year', 
                       'curah_hujan', 'suhu', 'luas_panen', 'hasil_panen']]
            .rename(columns={
                'timestamp_str': 'Waktu',
                'city': 'Kota',
                'model_year': 'Tahun Model',
                'prediction_year': 'Tahun Prediksi',
                'curah_hujan': 'Curah Hujan',
                'suhu': 'Suhu',
                'luas_panen': 'Luas Panen',
                'hasil_panen': 'Hasil Panen (ton)'
            }),
            use_container_width=True
        )
        
        # Option to clear history
        if st.button("Hapus Semua Riwayat"):
            st.session_state.prediction_history = []
            st.success("Semua riwayat prediksi telah dihapus.")
            st.experimental_rerun()
            
        # Visualization of prediction results
        st.subheader("Visualisasi Hasil Prediksi")
        
        # Group by city and calculate average predicted yield
        city_avg = history_df.groupby('city')['hasil_panen'].mean().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of predictions by city
            st.markdown("### Rata-rata Hasil Prediksi per Kota")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='city', y='hasil_panen', data=city_avg, ax=ax)
            ax.set_xlabel('Kota')
            ax.set_ylabel('Rata-rata Hasil Panen (ton)')
            ax.set_title('Rata-rata Hasil Prediksi per Kota')
            st.pyplot(fig)
        
        with col2:
            # Comparison of input parameters
            st.markdown("### Perbandingan Parameter Input")
            
            # Select parameter to visualize
            param = st.selectbox("Pilih Parameter", 
                               ["curah_hujan", "suhu", "luas_panen"], 
                               format_func=lambda x: {
                                   "curah_hujan": "Curah Hujan", 
                                   "suhu": "Suhu", 
                                   "luas_panen": "Luas Panen"
                               }[x])
            
            # Scatter plot of selected parameter vs prediction
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=param, y='hasil_panen', hue='city', data=history_df, ax=ax)
            ax.set_xlabel({
                "curah_hujan": "Curah Hujan", 
                "suhu": "Suhu", 
                "luas_panen": "Luas Panen"
            }[param])
            ax.set_ylabel('Hasil Panen (ton)')
            ax.set_title(f'Hubungan {param} dengan Hasil Prediksi')
            st.pyplot(fig)
        
        # Download results as CSV
        csv = history_df[['timestamp_str', 'city', 'model_year', 'prediction_year', 
                         'curah_hujan', 'suhu', 'luas_panen', 'hasil_panen']].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download Riwayat Prediksi (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)