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

# Initialize data structures for each year
for year in range(2018, 2023):
    if year not in st.session_state.data_malang:
        st.session_state.data_malang[year] = pd.DataFrame(columns=['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)', 'Y'])
    
    if year not in st.session_state.data_lumajang:
        st.session_state.data_lumajang[year] = pd.DataFrame(columns=['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)', 'Y'])

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

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
                        
                        # # Train model after adding data
                        # model = train_model("Malang", year_malang)
                        # if model is not None:
                        #     save_model("Malang", year_malang, model)
                        #     next_year = year_malang + 1
                        #     if next_year in st.session_state.rmse_malang:
                        #         st.success(f"Model untuk tahun {year_malang} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                        #     else:
                        #         st.success(f"Model untuk tahun {year_malang} berhasil dilatih!")
        
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
                        
                        # # Train model after adding data
                        # model = train_model("Lumajang", year_lumajang)
                        # if model is not None:
                        #     save_model("Lumajang", year_lumajang, model)
                        #     next_year = year_lumajang + 1
                        #     if next_year in st.session_state.rmse_lumajang:
                        #         st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                        #     else:
                        #         st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih!")
        
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

elif st.session_state.current_page == "Prediksi":
    st.header("Prediksi Panen Cabai")
    
    # Step 1: Pilih Data Aktual untuk Prediksi
    st.subheader("Step 1: Pilih Data Aktual untuk Prediksi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.selectbox("Pilih Kota", ["Malang", "Lumajang"])
    
    with col2:
        # Cari tahun yang memiliki data
        available_years = []
        if city == "Malang":
            for year in range(2018, 2023):
                if not st.session_state.data_malang[year].empty and len(st.session_state.data_malang[year]) >= 3:
                    available_years.append(year)
        else:
            for year in range(2018, 2023):
                if not st.session_state.data_lumajang[year].empty and len(st.session_state.data_lumajang[year]) >= 3:
                    available_years.append(year)
        
        if not available_years:
            st.error(f"Belum ada data yang cukup untuk kota {city}. Minimal 3 data per tahun diperlukan.")
            st.stop()
        
        selected_data_year = st.selectbox("Pilih Tahun Data untuk Model", available_years)
    
    # Step 2: Tentukan Tahun yang akan Diprediksi
    st.subheader("Step 2: Tentukan Tahun yang akan Diprediksi")
    
    prediction_year = st.selectbox("Pilih Tahun yang akan Diprediksi", 
                                 [year for year in range(2019, 2024) if year > selected_data_year])
    
    # Step 3: Input Data untuk Prediksi
    st.subheader("Step 3: Input Data untuk Prediksi")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            curah_hujan = st.number_input("Curah Hujan", min_value=0.0, format="%.2f")
        
        with col2:
            suhu = st.number_input("Suhu", min_value=0.0, format="%.2f")
        
        with col3:
            luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, format="%.3f")
        
        submit_button = st.form_submit_button("Lakukan Prediksi")
        
        if submit_button:
            # Train model dengan data yang dipilih
            if city == "Malang":
                data = st.session_state.data_malang[selected_data_year]
            else:
                data = st.session_state.data_lumajang[selected_data_year]
            
            # Train model
            X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
            y = data['Y']
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Lakukan prediksi
            prediction = model.predict([[curah_hujan, suhu, luas_panen]])[0]
            
            # ===== TAMPILKAN TABEL PENOLONG =====
            st.subheader("Tabel Penolong")
            
            df_helper = data.copy()
            # Hitung kolom-kolom penolong
            df_helper["X12"] = df_helper["X1(CURAH HUJAN)"] ** 2
            df_helper["X22"] = df_helper["X2(SUHU)"] ** 2
            df_helper["X32"] = df_helper["X3(LUAS PANEN)"] ** 2
            df_helper["Y2"] = df_helper["Y"] ** 2
            df_helper["X1Y"] = df_helper["X1(CURAH HUJAN)"] * df_helper["Y"]
            df_helper["X2Y"] = df_helper["X2(SUHU)"] * df_helper["Y"]
            df_helper["X3Y"] = df_helper["X3(LUAS PANEN)"] * df_helper["Y"]
            df_helper["X1X2"] = df_helper["X1(CURAH HUJAN)"] * df_helper["X2(SUHU)"]
            df_helper["X1X3"] = df_helper["X1(CURAH HUJAN)"] * df_helper["X3(LUAS PANEN)"]
            df_helper["X2X3"] = df_helper["X2(SUHU)"] * df_helper["X3(LUAS PANEN)"]
            
            # Atur index menjadi label bulan
            month_labels = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
            if len(df_helper) <= len(month_labels):
                df_helper.index = month_labels[:len(df_helper)]
            else:
                df_helper.index = [f"Data {i+1}" for i in range(len(df_helper))]
            
            st.dataframe(df_helper)
            
            # Simpan hasil prediksi dengan key yang konsisten
            prediction_key = f"{city}_{selected_data_year}_{prediction_year}"
            st.session_state.prediction_results[prediction_key] = {
                'city': city,
                'data_year': selected_data_year,
                'prediction_year': prediction_year,
                'model': model,
                'training_data': data,  # Data untuk training (tahun yang dipilih)
                'input_data': {
                    'curah_hujan': curah_hujan,
                    'suhu': suhu,
                    'luas_panen': luas_panen
                },
                'prediction': prediction,
                'timestamp': datetime.datetime.now()
            }
            
            st.success(f"Prediksi berhasil! Hasil prediksi untuk tahun {prediction_year}: {prediction:.3f} ton")
            st.info("Hasil lengkap prediksi dapat dilihat di menu 'Hasil'")
            
            # Tampilkan persamaan regresi
            st.subheader("Persamaan Regresi")
            coefficients = model.coef_
            intercept = model.intercept_
            
            st.markdown(f"**Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃**")
            st.markdown(f"- a (konstanta) = {intercept:.4f}")
            st.markdown(f"- b₁ (koefisien curah hujan) = {coefficients[0]:.4f}")
            st.markdown(f"- b₂ (koefisien suhu) = {coefficients[1]:.4f}")
            st.markdown(f"- b₃ (koefisien luas panen) = {coefficients[2]:.4f}")
                    
elif st.session_state.current_page == "Hasil":
    st.header("Hasil Prediksi")
    
    if not st.session_state.prediction_results:
        st.info("Belum ada prediksi yang dilakukan. Silakan lakukan prediksi terlebih dahulu di menu 'Prediksi'.")
    else:
        # Pilih hasil prediksi yang akan ditampilkan
        prediction_options = []
        for key, value in st.session_state.prediction_results.items():
            option_text = f"{value['city']} - Model {value['data_year']} untuk prediksi {value['prediction_year']}"
            prediction_options.append((key, option_text))
        
        selected_key = st.selectbox(
            "Pilih Hasil Prediksi yang akan Ditampilkan",
            options=[key for key, _ in prediction_options],
            format_func=lambda x: next(text for key, text in prediction_options if key == x)
        )
        
        if selected_key:
            result = st.session_state.prediction_results[selected_key]
            
            # 1. PERSAMAAN
            st.subheader("1. Persamaan Regresi")
            model = result['model']
            coefficients = model.coef_
            intercept = model.intercept_
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃**")
            with col2:
                st.markdown(f"- a = {intercept:.4f}")
                st.markdown(f"- b₁ = {coefficients[0]:.4f}")
                st.markdown(f"- b₂ = {coefficients[1]:.4f}")
                st.markdown(f"- b₃ = {coefficients[2]:.4f}")
            
            # 2. HASIL PREDIKSI UNTUK TAHUN YANG DIPREDIKSI
            st.subheader(f"2. Hasil Prediksi untuk Tahun {result['prediction_year']}")
            
            # Tampilkan input dan hasil prediksi baru
            input_data = result['input_data']
            st.markdown(f"**Input untuk tahun {result['prediction_year']}:**")
            st.markdown(f"- Curah Hujan = {input_data['curah_hujan']}")
            st.markdown(f"- Suhu = {input_data['suhu']}")
            st.markdown(f"- Luas Panen = {input_data['luas_panen']}")
            st.markdown(f"**Hasil Prediksi untuk tahun {result['prediction_year']}: {result['prediction']:.3f} ton**")
            
            # 3. EVALUASI MODEL DENGAN DATA TRAINING
            st.subheader(f"3. Evaluasi Model pada Data Training (Tahun {result['data_year']})")
            
            training_data = result['training_data']
            X_train = training_data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
            y_actual_train = training_data['Y']
            y_pred_train = model.predict(X_train)
            
            # Buat tabel perbandingan untuk data training
            comparison_df = pd.DataFrame({
                'Bulan': [f"Bulan {i+1}" for i in range(len(training_data))],
                'Curah Hujan': training_data['X1(CURAH HUJAN)'].values,
                'Suhu': training_data['X2(SUHU)'].values,
                'Luas Panen': training_data['X3(LUAS PANEN)'].values,
                'Hasil Aktual': y_actual_train.values,
                'Hasil Prediksi Model': y_pred_train,
                'Selisih': abs(y_actual_train.values - y_pred_train)
            })
            
            st.dataframe(comparison_df)
            
            # 4. UJI AKURASI RMSE - VERSI YANG DIPERBAIKI
            st.subheader("4. Uji Akurasi Model")
            
            # RMSE Training (pada data yang sama)
            rmse_training = np.sqrt(mean_squared_error(y_actual_train, y_pred_train))
            
            # RMSE Cross-Validation (seperti di halaman Data Aktual)
            prediction_year = result['prediction_year']
            city = result['city']
            data_year = result['data_year']
            
            # Cek apakah ada RMSE cross-validation yang tersimpan
            rmse_cross_val = None
            if city == "Malang" and prediction_year in st.session_state.rmse_malang:
                rmse_cross_val = st.session_state.rmse_malang[prediction_year]
            elif city == "Lumajang" and prediction_year in st.session_state.rmse_lumajang:
                rmse_cross_val = st.session_state.rmse_lumajang[prediction_year]
            
            # Jika tidak ada RMSE cross-validation tersimpan, hitung secara langsung
            if rmse_cross_val is None:
                # Coba hitung RMSE cross-validation jika ada data untuk tahun yang diprediksi
                if city == "Malang":
                    eval_data = st.session_state.data_malang.get(prediction_year)
                else:
                    eval_data = st.session_state.data_lumajang.get(prediction_year)
                
                if eval_data is not None and not eval_data.empty:
                    X_eval = eval_data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
                    y_actual_eval = eval_data['Y']
                    y_pred_eval = model.predict(X_eval)
                    rmse_cross_val = np.sqrt(mean_squared_error(y_actual_eval, y_pred_eval))
            
            # Tampilkan RMSE dengan penjelasan yang jelas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 RMSE Training Data")
                st.markdown(f"**{rmse_training:.4f}**")
                st.markdown("*(Model dievaluasi pada data training tahun {0})*".format(data_year))
                
                # Interpretasi RMSE Training
                mean_actual_train = np.mean(y_actual_train)
                rmse_percentage_train = (rmse_training / mean_actual_train) * 100
                st.markdown(f"**{rmse_percentage_train:.2f}%** dari rata-rata aktual")
            
            with col2:
                st.markdown("### 🎯 RMSE Cross-Validation")
                if rmse_cross_val is not None:
                    st.markdown(f"**{rmse_cross_val:.4f}**")
                    st.markdown("*(Model tahun {0} diuji pada data tahun {1})*".format(data_year, prediction_year))
                    
                    # Interpretasi RMSE Cross-Validation
                    if city == "Malang":
                        eval_data = st.session_state.data_malang.get(prediction_year)
                    else:
                        eval_data = st.session_state.data_lumajang.get(prediction_year)
                    
                    if eval_data is not None and not eval_data.empty:
                        mean_actual_eval = np.mean(eval_data['Y'])
                        rmse_percentage_eval = (rmse_cross_val / mean_actual_eval) * 100
                        st.markdown(f"**{rmse_percentage_eval:.2f}%** dari rata-rata aktual")
                else:
                    st.markdown("**Tidak tersedia**")
                    st.markdown(f"*(Data aktual tahun {prediction_year} belum ada)*")
            
            # Info box tentang perbedaan RMSE
            st.info("""
            📍 **Penjelasan RMSE:**
            - **Training RMSE:** Kesalahan model pada data yang digunakan untuk training (cenderung lebih rendah)
            - **Cross-Validation RMSE:** Kesalahan model saat memprediksi data tahun yang berbeda (lebih representatif untuk performa sesungguhnya)
            - Cross-Validation RMSE biasanya lebih tinggi karena menguji kemampuan generalisasi model
            """)
            
            # Tentukan RMSE mana yang akan digunakan untuk evaluasi utama
            primary_rmse = rmse_cross_val if rmse_cross_val is not None else rmse_training
            primary_rmse_type = "Cross-Validation" if rmse_cross_val is not None else "Training"
            
            # Interpretasi akurasi berdasarkan RMSE utama
            if rmse_cross_val is not None:
                if city == "Malang":
                    eval_data = st.session_state.data_malang.get(prediction_year)
                else:
                    eval_data = st.session_state.data_lumajang.get(prediction_year)
                mean_for_percentage = np.mean(eval_data['Y']) if eval_data is not None and not eval_data.empty else mean_actual_train
            else:
                mean_for_percentage = mean_actual_train
            
            primary_rmse_percentage = (primary_rmse / mean_for_percentage) * 100
            
            st.markdown(f"### 📈 Evaluasi Akurasi Model ({primary_rmse_type})")
            if primary_rmse_percentage < 10:
                st.success(f"Model memiliki akurasi yang sangat baik (RMSE {primary_rmse_type}: {primary_rmse_percentage:.2f}% < 10%)")
            elif primary_rmse_percentage < 20:
                st.warning(f"Model memiliki akurasi yang baik (RMSE {primary_rmse_type}: {primary_rmse_percentage:.2f}% = 10-20%)")
            else:
                st.error(f"Model memiliki akurasi yang perlu diperbaiki (RMSE {primary_rmse_type}: {primary_rmse_percentage:.2f}% > 20%)")
            
            # 5. JIKA ADA DATA AKTUAL UNTUK TAHUN YANG DIPREDIKSI - EVALUASI LENGKAP
            st.subheader(f"5. Evaluasi Cross-Validation (Data Aktual Tahun {prediction_year})")
            
            if city == "Malang":
                actual_data_pred_year = st.session_state.data_malang.get(prediction_year)
            else:
                actual_data_pred_year = st.session_state.data_lumajang.get(prediction_year)
            
            if actual_data_pred_year is not None and not actual_data_pred_year.empty:
                st.markdown(f"**Data aktual untuk tahun {prediction_year} ditemukan. Berikut evaluasi cross-validation:**")
                
                # Evaluasi cross-validation
                X_eval = actual_data_pred_year[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
                y_actual_eval = actual_data_pred_year['Y']
                y_pred_eval = model.predict(X_eval)
                
                # Tabel perbandingan cross-validation
                comparison_eval_df = pd.DataFrame({
                    'Bulan': [f"Bulan {i+1}" for i in range(len(actual_data_pred_year))],
                    'Curah Hujan': actual_data_pred_year['X1(CURAH HUJAN)'].values,
                    'Suhu': actual_data_pred_year['X2(SUHU)'].values,
                    'Luas Panen': actual_data_pred_year['X3(LUAS PANEN)'].values,
                    'Hasil Aktual': y_actual_eval.values,
                    'Hasil Prediksi Model': y_pred_eval,
                    'Selisih': abs(y_actual_eval.values - y_pred_eval)
                })
                
                st.dataframe(comparison_eval_df)
                
                # Hitung ulang RMSE cross-validation untuk memastikan konsistensi
                rmse_cross_val_calculated = np.sqrt(mean_squared_error(y_actual_eval, y_pred_eval))
                st.markdown(f"**RMSE Cross-Validation (Terhitung): {rmse_cross_val_calculated:.4f}**")
                
                # Bandingkan dengan RMSE yang tersimpan jika ada
                if rmse_cross_val is not None:
                    if abs(rmse_cross_val - rmse_cross_val_calculated) < 0.0001:
                        st.success("✅ RMSE konsisten dengan perhitungan di halaman Data Aktual")
                    else:
                        st.warning(f"⚠️ RMSE berbeda dengan yang tersimpan ({rmse_cross_val:.4f}). Gunakan nilai terhitung.")
                        rmse_cross_val = rmse_cross_val_calculated
            else:
                st.info(f"Data aktual untuk tahun {prediction_year} belum tersedia. Cross-validation tidak dapat dilakukan.")
            
            # 6. VISUALISASI PERBANDINGAN
            st.subheader("6. Visualisasi Evaluasi Model")
            
            # Tentukan data mana yang akan divisualisasikan
            if actual_data_pred_year is not None and not actual_data_pred_year.empty:
                # Visualisasi cross-validation
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: Cross-validation comparison
                months_eval = range(1, len(y_actual_eval) + 1)
                ax1.plot(months_eval, y_actual_eval, 'b-o', label='Aktual', linewidth=2, markersize=8)
                ax1.plot(months_eval, y_pred_eval, 'r--s', label='Prediksi Model', linewidth=2, markersize=8)
                ax1.set_xlabel('Bulan')
                ax1.set_ylabel('Hasil Panen (ton)')
                ax1.set_title(f'Cross-Validation: Model {data_year} → Data {prediction_year}\n{city}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Scatter plot cross-validation
                ax2.scatter(y_actual_eval, y_pred_eval, alpha=0.7, s=100, color='red', label='Cross-Validation')
                ax2.plot([y_actual_eval.min(), y_actual_eval.max()], [y_actual_eval.min(), y_actual_eval.max()], 'r--', lw=2)
                ax2.set_xlabel('Hasil Aktual (ton)')
                ax2.set_ylabel('Hasil Prediksi Model (ton)')
                ax2.set_title('Cross-Validation: Aktual vs Prediksi')
                ax2.grid(True, alpha=0.3)
                
                # R² untuk cross-validation
                from sklearn.metrics import r2_score
                r2_cross = r2_score(y_actual_eval, y_pred_eval)
                ax2.text(0.05, 0.95, f'R² = {r2_cross:.4f}', transform=ax2.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Metrik untuk cross-validation
                mae_cross = np.mean(abs(y_actual_eval - y_pred_eval))
                mape_cross = np.mean(abs((y_actual_eval - y_pred_eval) / y_actual_eval)) * 100
                
            else:
                # Visualisasi training data saja
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: Training data comparison
                months_train = range(1, len(y_actual_train) + 1)
                ax1.plot(months_train, y_actual_train, 'b-o', label='Aktual', linewidth=2, markersize=8)
                ax1.plot(months_train, y_pred_train, 'g--s', label='Prediksi Model', linewidth=2, markersize=8)
                ax1.set_xlabel('Bulan')
                ax1.set_ylabel('Hasil Panen (ton)')
                ax1.set_title(f'Training Data: Model dan Data Tahun {data_year}\n{city}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Scatter plot training
                ax2.scatter(y_actual_train, y_pred_train, alpha=0.7, s=100, color='green', label='Training')
                ax2.plot([y_actual_train.min(), y_actual_train.max()], [y_actual_train.min(), y_actual_train.max()], 'g--', lw=2)
                ax2.set_xlabel('Hasil Aktual (ton)')
                ax2.set_ylabel('Hasil Prediksi Model (ton)')
                ax2.set_title('Training Data: Aktual vs Prediksi')
                ax2.grid(True, alpha=0.3)
                
                # R² untuk training
                from sklearn.metrics import r2_score
                r2_train = r2_score(y_actual_train, y_pred_train)
                ax2.text(0.05, 0.95, f'R² = {r2_train:.4f}', transform=ax2.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Metrik untuk training
                mae_cross = np.mean(abs(y_actual_train - y_pred_train))
                mape_cross = np.mean(abs((y_actual_train - y_pred_train) / y_actual_train)) * 100
                r2_cross = r2_train
            
            # 7. RINGKASAN STATISTIK MODEL
            st.subheader("7. Ringkasan Statistik Model")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Gunakan metrik cross-validation jika ada, jika tidak gunakan training
            display_rmse = rmse_cross_val if rmse_cross_val is not None else rmse_training
            display_type = "Cross-Val" if rmse_cross_val is not None else "Training"
            
            with col1:
                st.metric(f"RMSE ({display_type})", f"{display_rmse:.4f}")
            
            with col2:
                st.metric("R²", f"{r2_cross:.4f}")
            
            with col3:
                st.metric("MAE", f"{mae_cross:.4f}")
            
            with col4:
                st.metric("MAPE (%)", f"{mape_cross:.2f}")
            
            # Tampilkan juga metrik training jika cross-validation tersedia
            if rmse_cross_val is not None:
                st.markdown("**Perbandingan Metrik:**")
                comparison_metrics = pd.DataFrame({
                    'Metrik': ['RMSE', 'R²', 'MAE', 'MAPE (%)'],
                    'Training Data': [
                        f"{rmse_training:.4f}",
                        f"{r2_score(y_actual_train, y_pred_train):.4f}",
                        f"{np.mean(abs(y_actual_train - y_pred_train)):.4f}",
                        f"{np.mean(abs((y_actual_train - y_pred_train) / y_actual_train)) * 100:.2f}"
                    ],
                    'Cross-Validation': [
                        f"{rmse_cross_val:.4f}",
                        f"{r2_cross:.4f}",
                        f"{mae_cross:.4f}",
                        f"{mape_cross:.2f}"
                    ]
                })
                st.dataframe(comparison_metrics)
            
            # 8. DOWNLOAD HASIL
            st.subheader("8. Download Hasil")
            
            # Create download button
            csv_data = []
            csv_data.append("=== INFORMASI MODEL ===")
            csv_data.append(f"Kota,{result['city']}")
            csv_data.append(f"Tahun Data Training,{result['data_year']}")
            csv_data.append(f"Tahun Prediksi,{result['prediction_year']}")
            csv_data.append(f"Persamaan,Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃")
            csv_data.append("")
            csv_data.append("=== PREDIKSI TAHUN {0} ===".format(result['prediction_year']))
            csv_data.append(f"Input Curah Hujan,{input_data['curah_hujan']}")
            csv_data.append(f"Input Suhu,{input_data['suhu']}")
            csv_data.append(f"Input Luas Panen,{input_data['luas_panen']}")
            csv_data.append(f"Hasil Prediksi,{result['prediction']:.3f}")
            csv_data.append("")
            csv_data.append("=== EVALUASI MODEL ===")
            csv_data.append(f"RMSE Training,{rmse_training:.4f}")
            if rmse_cross_val is not None:
                csv_data.append(f"RMSE Cross-Validation,{rmse_cross_val:.4f}")
            csv_data.append(f"R²,{r2_cross:.4f}")
            csv_data.append(f"MAE,{mae_cross:.4f}")
            csv_data.append(f"MAPE (%),{mape_cross:.2f}")
            csv_data.append("")
            csv_data.append("=== DATA TRAINING ({0}) ===".format(result['data_year']))
            csv_data.append(comparison_df.to_csv(index=False))
            
            if actual_data_pred_year is not None and not actual_data_pred_year.empty:
                csv_data.append("=== DATA CROSS-VALIDATION ({0}) ===".format(prediction_year))
                csv_data.append(comparison_eval_df.to_csv(index=False))
            
            csv_string = "\n".join(csv_data)
            
            st.download_button(
                label="Download Hasil Lengkap (CSV)",
                data=csv_string,
                file_name=f"hasil_prediksi_{result['city']}_{result['data_year']}_to_{result['prediction_year']}.csv",
                mime="text/csv"
            )