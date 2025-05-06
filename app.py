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
    for year in range(2018, 2023):
        st.session_state.data_malang[year] = pd.DataFrame(columns=['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)', 'Y'])

if 'data_lumajang' not in st.session_state:
    st.session_state.data_lumajang = {}
    for year in range(2018, 2023):
        st.session_state.data_lumajang[year] = pd.DataFrame(columns=['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)', 'Y'])

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
def save_model(city, year, model, rmse):
    if city == "Malang":
        st.session_state.models_malang[year] = model
        st.session_state.rmse_malang[year] = rmse
    else:
        st.session_state.models_lumajang[year] = model
        st.session_state.rmse_lumajang[year] = rmse

# Function to load example data
def load_example_data():
    # Example data for Malang 2018
    malang_2018 = pd.DataFrame({
        'X1(CURAH HUJAN)': [389, 383, 261, 107, 162, 43, 14, 25, 35, 25, 62, 315],
        'X2(SUHU)': [25, 25, 26, 26, 25, 25, 24, 14, 25, 35, 25, 26],
        'X3(LUAS PANEN)': [1.345, 1.337, 1.440, 1.443, 1.836, 1.729, 1.475, 1.224, 1.235, 1.226, 0.626, 0.418],
        'Y': [54.680, 61.003, 66.760, 56.227, 64.898, 72.915, 61.818, 50.128, 51.703, 71.415, 24.970, 19.797]
    })
    st.session_state.data_malang[2018] = malang_2018

    # Create some sample data for other years and cities
    for year in range(2018, 2023):
        if year != 2018:  # We already have 2018 data for Malang
            st.session_state.data_malang[year] = pd.DataFrame({
                'X1(CURAH HUJAN)': np.random.randint(10, 400, 12),
                'X2(SUHU)': np.random.randint(14, 36, 12),
                'X3(LUAS PANEN)': np.round(np.random.uniform(0.4, 2.0, 12), 3),
                'Y': np.round(np.random.uniform(19.0, 75.0, 12), 3)
            })
        
        st.session_state.data_lumajang[year] = pd.DataFrame({
            'X1(CURAH HUJAN)': np.random.randint(10, 400, 12),
            'X2(SUHU)': np.random.randint(14, 36, 12),
            'X3(LUAS PANEN)': np.round(np.random.uniform(0.4, 2.0, 12), 3),
            'Y': np.round(np.random.uniform(19.0, 75.0, 12), 3)
        })

# Function to train models
def train_model(city, year):
    if city == "Malang":
        data = st.session_state.data_malang[year]
    else:
        data = st.session_state.data_lumajang[year]
    
    if len(data) < 3:  # Need at least 3 data points for meaningful regression
        return None, None
    
    X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
    y = data['Y']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate RMSE
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return model, rmse

# Function to make predictions
def predict(city, year, curah_hujan, suhu, luas_panen):
    if city == "Malang":
        if year not in st.session_state.models_malang:
            return None
        model = st.session_state.models_malang[year]
    else:
        if year not in st.session_state.models_lumajang:
            return None
        model = st.session_state.models_lumajang[year]
    
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
st.markdown("Prediksi hasil panen cabai menggunakan regresi linear berganda untuk kota Malang dan Lumajang")

# Create sidebar menu
menu = st.sidebar.selectbox("Menu", ["Dashboard", "Data Aktual", "Prediksi", "Hasil"])

# Load example data if button is clicked
if st.sidebar.button("Load Data Contoh"):
    load_example_data()
    st.sidebar.success("Data contoh berhasil dimuat!")

# Dashboard page
if menu == "Dashboard":
    st.header("Dashboard Prediksi Panen Cabai")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kota Malang")
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
elif menu == "Data Aktual":
    st.header("Data Aktual Panen Cabai")
    
    tab1, tab2 = st.tabs(["Malang", "Lumajang"])
    
    with tab1:
        st.subheader("Data Kota Malang")
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
                        model, rmse = train_model("Malang", year_malang)
                        if model is not None:
                            save_model("Malang", year_malang, model, rmse)
                            st.success(f"Model untuk tahun {year_malang} berhasil dilatih! RMSE: {rmse:.4f}")
        
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
                            model, rmse = train_model("Malang", year_malang)
                            if model is not None:
                                save_model("Malang", year_malang, model, rmse)
                                st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang! RMSE: {rmse:.4f}")
        
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
                        model, rmse = train_model("Malang", year_malang)
                        if model is not None:
                            save_model("Malang", year_malang, model, rmse)
                            st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang! RMSE: {rmse:.4f}")
        
        # Train all models button
        if st.button("Latih Semua Model untuk Malang"):
            success_count = 0
            for year in range(2018, 2023):
                if not st.session_state.data_malang[year].empty and len(st.session_state.data_malang[year]) >= 3:
                    model, rmse = train_model("Malang", year)
                    if model is not None:
                        save_model("Malang", year, model, rmse)
                        success_count += 1
            
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
                        model, rmse = train_model("Lumajang", year_lumajang)
                        if model is not None:
                            save_model("Lumajang", year_lumajang, model, rmse)
                            st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih! RMSE: {rmse:.4f}")
        
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
                            model, rmse = train_model("Lumajang", year_lumajang)
                            if model is not None:
                                save_model("Lumajang", year_lumajang, model, rmse)
                                st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang! RMSE: {rmse:.4f}")
        
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
                        model, rmse = train_model("Lumajang", year_lumajang)
                        if model is not None:
                            save_model("Lumajang", year_lumajang, model, rmse)
                            st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang! RMSE: {rmse:.4f}")
        
        # Train all models button
        if st.button("Latih Semua Model untuk Lumajang"):
            success_count = 0
            for year in range(2018, 2023):
                if not st.session_state.data_lumajang[year].empty and len(st.session_state.data_lumajang[year]) >= 3:
                    model, rmse = train_model("Lumajang", year)
                    if model is not None:
                        save_model("Lumajang", year, model, rmse)
                        success_count += 1
            
            if success_count > 0:
                st.success(f"{success_count} model berhasil dilatih untuk Lumajang!")
            else:
                st.warning("Tidak ada model yang berhasil dilatih. Pastikan data cukup (minimal 3 baris per tahun).")

# Prediction page
elif menu == "Prediksi":
    st.header("Prediksi Panen Cabai")
    
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
        
        year = st.selectbox("Pilih Tahun Model", sorted(model_years))
    
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
            result = predict(city, year, curah_hujan, suhu, luas_panen)
            
            if result is not None:
                st.success(f"Prediksi hasil panen cabai: {result:.3f} ton")
                
                # Add to prediction history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.prediction_history.append({
                    "timestamp": timestamp,
                    "city": city,
                    "year_model": year,
                    "curah_hujan": curah_hujan,
                    "suhu": suhu,
                    "luas_panen": luas_panen,
                    "prediction": result
                })
                
                # Show model details
                st.markdown("### Detail Model")
                
                if city == "Malang":
                    model = st.session_state.models_malang[year]
                    rmse = st.session_state.rmse_malang[year]
                else:
                    model = st.session_state.models_lumajang[year]
                    rmse = st.session_state.rmse_lumajang[year]
                
                coefficients = model.coef_
                intercept = model.intercept_
                
                st.markdown(f"**Persamaan Model:**")
                st.markdown(f"Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃")
                st.markdown(f"**RMSE:** {rmse:.4f}")
                
                # Show visualization if we have data for that year
                if city == "Malang" and not st.session_state.data_malang[year].empty:
                    data = st.session_state.data_malang[year]
                    X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
                    y_actual = data['Y']
                    y_pred = model.predict(X)
                    
                    # Plot actual vs predicted
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(1, len(y_actual) + 1), y_actual, 'b-', label='Aktual')
                    ax.plot(range(1, len(y_pred) + 1), y_pred, 'r--', label='Prediksi')
                    ax.set_xlabel('Bulan')
                    ax.set_ylabel('Hasil Panen (ton)')
                    ax.set_title(f'Perbandingan Hasil Aktual vs Prediksi - {city} {year}')
                    ax.legend()
                    st.pyplot(fig)
                
                elif city == "Lumajang" and not st.session_state.data_lumajang[year].empty:
                    data = st.session_state.data_lumajang[year]
                    X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
                    y_actual = data['Y']
                    y_pred = model.predict(X)
                    
                    # Plot actual vs predicted
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(1, len(y_actual) + 1), y_actual, 'b-', label='Aktual')
                    ax.plot(range(1, len(y_pred) + 1), y_pred, 'r--', label='Prediksi')
                    ax.set_xlabel('Bulan')
                    ax.set_ylabel('Hasil Panen (ton)')
                    ax.set_title(f'Perbandingan Hasil Aktual vs Prediksi - {city} {year}')
                    ax.legend()
                    st.pyplot(fig)

# Results page
elif menu == "Hasil":
    st.header("Riwayat Hasil Prediksi")
    
    if not st.session_state.prediction_history:
        st.info("Belum ada hasil prediksi yang tersimpan. Silakan lakukan prediksi terlebih dahulu.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_city = st.multiselect("Filter berdasarkan Kota", options=["Malang", "Lumajang"], default=["Malang", "Lumajang"])
        
        with col2:
            model_years = set()
            for item in st.session_state.prediction_history:
                model_years.add(item["year_model"])
            filter_year = st.multiselect("Filter berdasarkan Tahun Model", options=sorted(list(model_years)), default=sorted(list(model_years)))
        
        # Apply filters
        filtered_df = history_df[
            (history_df["city"].isin(filter_city)) & 
            (history_df["year_model"].isin(filter_year))
        ]
        
        # Display filtered results
        if not filtered_df.empty:
            st.dataframe(filtered_df)
            
            # Download results
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download Riwayat Prediksi (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Visualization
            st.markdown("### Visualisasi Hasil Prediksi")
            
            # Plot predictions over time
            fig, ax = plt.subplots(figsize=(12, 6))
            for city in filter_city:
                city_data = filtered_df[filtered_df["city"] == city]
                if not city_data.empty:
                    ax.plot(city_data.index, city_data["prediction"], 'o-', label=f"{city}")
            
            ax.set_xlabel("Urutan Prediksi")
            ax.set_ylabel("Hasil Prediksi (ton)")
            ax.set_title("Hasil Prediksi Berdasarkan Waktu")
            ax.legend()
            st.pyplot(fig)
            
            # Distribution of predictions
            fig, ax = plt.subplots(figsize=(12, 6))
            for city in filter_city:
                city_data = filtered_df[filtered_df["city"] == city]
                if not city_data.empty:
                    sns.histplot(city_data["prediction"], kde=True, label=city, ax=ax, alpha=0.6)
            
            ax.set_xlabel("Hasil Prediksi (ton)")
            ax.set_ylabel("Frekuensi")
            ax.set_title("Distribusi Hasil Prediksi")
            ax.legend()
            st.pyplot(fig)
            
            # Correlation between input variables and prediction
            st.markdown("### Korelasi Antar Variabel")
            
            corr_vars = ["curah_hujan", "suhu", "luas_panen", "prediction"]
            for city in filter_city:
                city_data = filtered_df[filtered_df["city"] == city]
                if not city_data.empty:
                    st.markdown(f"**Kota {city}**")
                    correlation = city_data[corr_vars].corr()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
                    ax.set_title(f"Korelasi Antar Variabel - {city}")
                    st.pyplot(fig)
        else:
            st.info("Tidak ada data yang sesuai dengan filter.")
            
        # Add a reset button
        if st.button("Reset Riwayat Prediksi"):
            st.session_state.prediction_history = []
            st.success("Riwayat prediksi berhasil direset!")
            st.rerun()