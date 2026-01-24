import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Clustering & Prediction", layout="wide", page_icon=None)

# Path to the training data
TRAINING_DATA_PATH = "model_training.csv"

# Cluster descriptions based on business insights
CLUSTER_DESCRIPTIONS = {
    0: {
        "name": "Karyawan Berkembang Cepat",
        "characteristics": [
            "Masa kerja pendek",
            "Kecepatan performa tinggi",
            "Intensitas pelatihan sangat tinggi",
            "Kepuasan kerja lebih rendah"
        ],
        "insight": "Karyawan awal karir yang berkembang pesat namun menghadapi risiko tinggi terhadap kelelahan (burnout) dan turnover dini akibat tekanan dan beban pelatihan yang berlebihan."
    },
    1: {
        "name": "Performa Stagnan",
        "characteristics": [
            "Jam pelatihan tinggi",
            "Masa kerja relatif panjang",
            "Kecepatan performa rendah",
            "Efisiensi pelatihan rendah"
        ],
        "insight": "Sebagian besar karyawan telah mengikuti banyak program pelatihan, namun peningkatan performa berjalan lambat. Program pelatihan yang ada tidak lagi memberikan dampak signifikan."
    },
    2: {
        "name": "Bintang ROI Tinggi",
        "characteristics": [
            "Efisiensi pelatihan sangat tinggi",
            "Jam pelatihan rendah",
            "Kesenjangan potensi tinggi",
            "Masa kerja stabil"
        ],
        "insight": "Karyawan berpotensi tinggi yang menghasilkan dampak besar dengan pelatihan minimal. Cluster ini memberikan ROI pelatihan tertinggi."
    }
}

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'cluster_profiles' not in st.session_state:
    st.session_state.cluster_profiles = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'algorithm_type' not in st.session_state:
    st.session_state.algorithm_type = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []

# Header
st.title("Dashboard Clustering & Prediction System")
st.markdown("Sistem prediksi cluster menggunakan **Pre-trained Model GMM & Random Forest**")
st.markdown("---")

# Helper function
def calculate_distance_to_centroid(X, labels, model):
    """Calculate distance from each point to its assigned cluster centroid"""
    distances = np.zeros(X.shape[0])
    centers = model.means_
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        cluster_points = X[mask]
        centroid = centers[label]
        dist = np.linalg.norm(cluster_points - centroid, axis=1)
        distances[mask] = dist
        
    return distances

# Function to train model from CSV
def train_model_from_csv():
    """Load data and train model automatically"""
    try:
        # Load the training data
        df = pd.read_csv(TRAINING_DATA_PATH)
        
        # Define features to use for clustering (excluding Cluster column and categorical)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove 'Cluster' if it exists
        if 'Cluster' in numeric_cols:
            numeric_cols.remove('Cluster')
        
        # Select features (you can customize this list)
        selected_features = [
            'Age', 'EducationLevel', 'YearsAtCompany_log', 'MonthlyIncome_log', 
            'JobSatisfaction', 'PerformanceScore', 'CompetencyScore',
            'PerformanceVelocity', 'PotentialGap', 'TrainingHours_log',
            'TrainingIntensity', 'TrainingEfficiency'
        ]
        
        # Filter features that exist in the dataframe
        selected_features = [f for f in selected_features if f in df.columns]
        
        if len(selected_features) < 2:
            return None, "Tidak cukup fitur numerik dalam dataset"
        
        # Prepare data
        X = df[selected_features].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=selected_features)
        
        # Scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Training GMM (3 clusters)
        n_clusters = 3
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        model.fit(X_scaled)
        
        labels = model.predict(X_scaled)
        algorithm_name = f"GMM Tuned (K={n_clusters}, full)"
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, labels)
        
        feature_importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Calculate distances
        distances = calculate_distance_to_centroid(X_scaled, labels, model)
        
        # Add results to dataframe
        df['Cluster'] = labels
        df['Distance_to_Centroid'] = distances
        
        # Calculate profiles
        cluster_profiles = df.groupby('Cluster')[selected_features].agg(['mean', 'std', 'min', 'max'])
        
        # Return all components
        return {
            'model': model,
            'rf_model': rf,
            'scaler': scaler,
            'imputer': imputer,
            'selected_features': selected_features,
            'algorithm_name': algorithm_name,
            'feature_importances': feature_importances,
            'cluster_profiles': cluster_profiles,
            'df_clustered': df,
            'X_scaled': X_scaled
        }, None
        
    except Exception as e:
        return None, str(e)

# Auto-load model on startup
if not st.session_state.model_loaded:
    with st.spinner("Memuat dan melatih model dari data training..."):
        result, error = train_model_from_csv()
        
        if result is not None:
            st.session_state.trained_model = result['model']
            st.session_state.rf_model = result['rf_model']
            st.session_state.scaler = result['scaler']
            st.session_state.imputer = result['imputer']
            st.session_state.selected_features = result['selected_features']
            st.session_state.feature_names = result['selected_features']
            st.session_state.algorithm_type = result['algorithm_name']
            st.session_state.feature_importances = result['feature_importances']
            st.session_state.cluster_profiles = result['cluster_profiles']
            st.session_state.df_clustered = result['df_clustered']
            st.session_state.X_scaled = result['X_scaled']
            st.session_state.model_loaded = True
            st.success("Model berhasil dimuat !")
        else:
            st.error(f"Error memuat model: {error}")

# Sidebar
with st.sidebar:
    # Logo - lebih kecil dan centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("sylva-no-bg-white.png", width=100)
    
    st.markdown("")  # Single space instead of divider
    
    # Navigation
    st.markdown("### Navigation")
    mode = st.radio(
        "Pilih Mode:",
        ["Info Model", "Prediksi Data Baru"],
        help="Info Model: Lihat detail model | Prediksi: Klasifikasikan data baru",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model status - more compact
    st.markdown("### Status Model")
    if st.session_state.trained_model is not None:
        # Compact status display
        st.markdown("""
        <div style='background-color: #1e4620; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
            <p style='color: white; margin: 0; font-size: 14px;'>Model Aktif</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Compact info boxes
        st.markdown(f"""
        <div style='background-color: #1f4d7a; padding: 8px; border-radius: 5px; margin-bottom: 5px;'>
            <p style='color: white; margin: 0; font-size: 12px;'><b>Algoritma:</b><br>{st.session_state.algorithm_type}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background-color: #1f4d7a; padding: 8px; border-radius: 5px; margin-bottom: 10px;'>
            <p style='color: white; margin: 0; font-size: 12px;'><b>Jumlah Fitur:</b> {len(st.session_state.selected_features)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Detail Fitur"):
            for i, feat in enumerate(st.session_state.selected_features, 1):
                st.caption(f"{i}. {feat}")
    else:
        st.markdown("""
        <div style='background-color: #7a1f1f; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
            <p style='color: white; margin: 0; font-size: 14px;'>Model Tidak Tersedia</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Pastikan file CSV tersedia")
    
    st.markdown("---")
    
    # Reload model button
    if st.button("Reload Model", use_container_width=True, type="secondary"):
        st.session_state.model_loaded = False
        st.rerun()

# MODE 1: INFO MODEL
if mode == "Info Model":
    st.header("Informasi Model yang Dimuat")
    
    if st.session_state.trained_model is None:
        st.warning("### Model Belum Dimuat")
        st.markdown(f"""
        Pastikan file **{TRAINING_DATA_PATH}** tersedia di direktori yang sama dengan aplikasi ini.
        """)
    else:
        # Model Overview
        st.subheader("Overview Model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Algoritma", st.session_state.algorithm_type)
        
        with col2:
            st.metric("Jumlah Fitur", len(st.session_state.selected_features))
        
        with col3:
            if hasattr(st.session_state.trained_model, 'n_components'):
                st.metric("Jumlah Cluster", st.session_state.trained_model.n_components)
        
        st.markdown("---")
        
        # Model Metrics
        if st.session_state.X_scaled is not None and st.session_state.df_clustered is not None:
            st.subheader("Evaluasi Model")
            
            X_scaled = st.session_state.X_scaled
            labels = st.session_state.df_clustered['Cluster'].values
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                n_clusters_found = len(np.unique(labels))
                st.metric("Jumlah Cluster", n_clusters_found)
            
            with col2:
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    st.metric("Silhouette Score", f"{silhouette:.4f}")
            
            with col3:
                if len(np.unique(labels)) > 1:
                    db_score = davies_bouldin_score(X_scaled, labels)
                    st.metric("Davies-Bouldin", f"{db_score:.4f}")
            
            with col4:
                if len(np.unique(labels)) > 1:
                    ch_score = calinski_harabasz_score(X_scaled, labels)
                    st.metric("Calinski-Harabasz", f"{ch_score:.1f}")
            
            with col5:
                if 'Distance_to_Centroid' in st.session_state.df_clustered.columns:
                    avg_distance = st.session_state.df_clustered['Distance_to_Centroid'].mean()
                    st.metric("Avg Distance to Centroid", f"{avg_distance:.4f}")
        
        st.markdown("---")
        
        # Features Used
        st.subheader("Fitur yang Digunakan")
        
        features_df = pd.DataFrame({
            'No': range(1, len(st.session_state.selected_features) + 1),
            'Nama Fitur': st.session_state.selected_features
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Feature Importance
        # if st.session_state.feature_importances is not None:
        #     st.subheader("Feature Importance (Random Forest)")
            
        #     fi_df = st.session_state.feature_importances
            
        #     col1, col2 = st.columns([2, 1])
            
        #     with col1:
        #         fig_fi = px.bar(
        #             fi_df, 
        #             x='Importance', 
        #             y='Feature', 
        #             orientation='h',
        #             title="Kontribusi Fitur terhadap Pembentukan Cluster",
        #             color='Importance',
        #             color_continuous_scale='Blues'
        #         )
        #         fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        #         st.plotly_chart(fig_fi, use_container_width=True)
            
        #     with col2:
        #         st.markdown("#### Top Features")
        #         for idx, row in fi_df.head(5).iterrows():
        #             st.metric(
        #                 row['Feature'],
        #                 f"{row['Importance']:.4f}"
        #             )
            
        #     st.info("**Interpretasi**: Fitur dengan nilai importance tertinggi adalah fitur yang paling membedakan/mempengaruhi pembentukan cluster.")
        
        # st.markdown("---")
        
        # Cluster Profiles
        if st.session_state.cluster_profiles is not None:
            st.subheader("Profil Cluster")
            
            st.markdown("#### Statistik per Cluster")
            st.dataframe(
                st.session_state.cluster_profiles.style.background_gradient(cmap='RdYlGn', axis=1),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Cluster Distribution
        if st.session_state.df_clustered is not None:
            st.subheader("Distribusi Data per Cluster")
            
            cluster_counts = st.session_state.df_clustered['Cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    marker_color='lightblue',
                    text=cluster_counts.values,
                    textposition='auto'
                )])
                fig.update_layout(
                    title="Jumlah Data per Cluster",
                    xaxis_title="Cluster",
                    yaxis_title="Jumlah Data",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Detail Distribusi")
                total_data = len(st.session_state.df_clustered)
                for cluster_id in sorted(cluster_counts.index):
                    count = cluster_counts[cluster_id]
                    percentage = count / total_data * 100
                    st.metric(
                        f"Cluster {cluster_id}",
                        f"{count} data",
                        f"{percentage:.1f}%"
                    )

# MODE 2: PREDICTION
else:  # mode == "Prediksi Data Baru"
    st.header("Prediksi Cluster untuk Data Baru")
    
    if st.session_state.trained_model is None:
        st.warning("### Model Belum Dimuat")
        st.markdown("""
        Model gagal dimuat.
        gunakan tombol **Reload Model** di sidebar.
        """)
        
    else:
        # Show model info
        st.success(f"Model Aktif: {st.session_state.algorithm_type}")
        
        with st.expander("Informasi Model"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Fitur yang Digunakan:**")
                for i, feat in enumerate(st.session_state.selected_features, 1):
                    st.text(f"{i}. {feat}")
            with col2:
                if hasattr(st.session_state.trained_model, 'n_components'):
                    st.markdown("**Jumlah Cluster:**")
                    st.metric("Total Cluster", st.session_state.trained_model.n_components)
        
        st.markdown("---")
        
        # Input method selection
        st.subheader("1. Pilih Metode Input Data")
        
        input_method = st.radio(
            "Metode Input:",
            ["Manual Input", "Upload CSV"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # MANUAL INPUT
        if input_method == "Manual Input":
            st.subheader("2. Masukkan Nilai Fitur")
            
            # Education Level Guide
            st.info("""
            Panduan Education Level:
            - High School = 1
            - Bachelor = 2
            - Master = 3
            - PhD = 4
            """)
            
            input_data = {}
            
            # Create input form in columns
            n_features = len(st.session_state.selected_features)
            n_cols = min(3, n_features)
            cols = st.columns(n_cols)
            
            for i, feature in enumerate(st.session_state.selected_features):
                with cols[i % n_cols]:
                    # Special handling for EducationLevel
                    if feature == 'EducationLevel':
                        input_data[feature] = st.number_input(
                            feature,
                            min_value=1,
                            max_value=4,
                            value=2,
                            step=1,
                            help="1=High School, 2=Bachelor, 3=Master, 4=PhD",
                            key=f"input_{feature}"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            feature,
                            value=0.0,
                            format="%.4f",
                            key=f"input_{feature}"
                        )
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                run_prediction = st.button(
                    "Prediksi Cluster",
                    type="primary",
                    use_container_width=True
                )
            
            if run_prediction:
                try:
                    # Prepare input
                    input_df = pd.DataFrame([input_data])
                    
                    st.markdown("---")
                    st.subheader("Hasil Prediksi")
                    
                    # Show input
                    st.markdown("#### Data Input Anda:")
                    st.dataframe(input_df.T.rename(columns={0: 'Nilai'}), use_container_width=True)
                    
                    # Impute and scale
                    input_imputed = st.session_state.imputer.transform(input_df)
                    input_scaled = st.session_state.scaler.transform(input_imputed)
                    
                    # Predict
                    predicted_cluster = st.session_state.trained_model.predict(input_scaled)[0]
                    
                    # Show prediction
                    st.markdown("---")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### Hasil Prediksi")
                        st.markdown(f"# **Cluster {predicted_cluster}**")
                        
                        # Show cluster name and description
                        if predicted_cluster in CLUSTER_DESCRIPTIONS:
                            cluster_info = CLUSTER_DESCRIPTIONS[predicted_cluster]
                            st.markdown(f"**{cluster_info['name']}**")
                    
                    with col2:
                        st.markdown("### Profil Cluster Ini")
                        
                        if st.session_state.cluster_profiles is not None:
                            try:
                                # Extract mean values for this cluster
                                cluster_means = {}
                                for feature in st.session_state.selected_features:
                                    cluster_means[feature] = st.session_state.cluster_profiles.loc[predicted_cluster, (feature, 'mean')]
                                
                                profile_df = pd.DataFrame({
                                    'Feature': list(cluster_means.keys()),
                                    'Mean Value': list(cluster_means.values())
                                })
                                st.dataframe(profile_df, use_container_width=True, hide_index=True)
                            except:
                                st.info("Profil cluster tidak tersedia.")
                    
                    # Show cluster insights FIRST
                    st.markdown("---")
                    if predicted_cluster in CLUSTER_DESCRIPTIONS:
                        cluster_info = CLUSTER_DESCRIPTIONS[predicted_cluster]
                        st.markdown(f"### Cluster Profile: {cluster_info['name']}")
                        
                        st.markdown("**Karakteristik:**")
                        for char in cluster_info['characteristics']:
                            st.markdown(f"- {char}")
                        
                        st.markdown("**Business Insight:**")
                        st.info(cluster_info['insight'])
                    
                    # Detailed explanation
                    st.markdown("---")
                    st.subheader("Penjelasan Detail Prediksi")
                    
                    if st.session_state.cluster_profiles is not None:
                        try:
                            comparison_data = []
                            for feature in st.session_state.selected_features:
                                input_value = input_data[feature]
                                
                                # Get cluster statistics
                                cluster_mean = st.session_state.cluster_profiles.loc[predicted_cluster, (feature, 'mean')]
                                cluster_std = st.session_state.cluster_profiles.loc[predicted_cluster, (feature, 'std')]
                                cluster_min = st.session_state.cluster_profiles.loc[predicted_cluster, (feature, 'min')]
                                cluster_max = st.session_state.cluster_profiles.loc[predicted_cluster, (feature, 'max')]
                                
                                # Calculate z-score
                                if cluster_std > 0:
                                    z_score = (input_value - cluster_mean) / cluster_std
                                else:
                                    z_score = 0
                                
                                # Determine status
                                if abs(z_score) < 1:
                                    status = 'Normal'
                                elif abs(z_score) < 2:
                                    status = 'Moderate'
                                else:
                                    status = 'Outlier'
                                
                                comparison_data.append({
                                    'Fitur': feature,
                                    'Nilai Input': f"{input_value:.2f}",
                                    'Mean Cluster': f"{cluster_mean:.2f}",
                                    'Std Cluster': f"{cluster_std:.2f}",
                                    'Range': f"{cluster_min:.2f} - {cluster_max:.2f}",
                                    'Z-Score': f"{z_score:.2f}",
                                    'Status': status
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Generate interpretations
                            st.markdown("---")
                            st.markdown("### Interpretasi:")
                            
                            for _, row in comparison_df.iterrows():
                                z = float(row['Z-Score'])
                                
                                if abs(z) < 0.5:
                                    explanation = f"**{row['Fitur']}**: Sangat sesuai dengan profil cluster (nilai sangat dekat dengan rata-rata)"
                                elif abs(z) < 1:
                                    explanation = f"**{row['Fitur']}**: Sesuai dengan profil cluster (dalam range normal)"
                                elif abs(z) < 2:
                                    if z > 0:
                                        explanation = f"**{row['Fitur']}**: Sedikit di atas rata-rata cluster"
                                    else:
                                        explanation = f"**{row['Fitur']}**: Sedikit di bawah rata-rata cluster"
                                else:
                                    if z > 0:
                                        explanation = f"**{row['Fitur']}**: Jauh di atas rata-rata cluster (outlier)"
                                    else:
                                        explanation = f"**{row['Fitur']}**: Jauh di bawah rata-rata cluster (outlier)"
                                
                                st.markdown(f"- {explanation}")
                            
                            # Overall conclusion
                            st.markdown("---")
                            
                            st.success(f"""
                            **Kesimpulan**: 
                            
                            Data ini diprediksi masuk ke **Cluster {predicted_cluster}** karena nilai-nilai fiturnya 
                            paling mirip dengan karakteristik anggota cluster tersebut. Dari {len(st.session_state.selected_features)} fitur yang dianalisis, 
                            mayoritas nilai berada dalam range normal cluster ini.
                            """)
                        except Exception as e:
                            st.warning("Detail perbandingan tidak tersedia.")
                
                except Exception as e:
                    st.error(f"Error saat prediksi: {str(e)}")
                    st.exception(e)
        
        # BATCH PREDICTION (Upload CSV)
        else:  # Upload CSV
            st.subheader("2. Upload File CSV untuk Prediksi Batch")
            
            predict_file = st.file_uploader(
                "Upload CSV yang berisi data untuk diprediksi",
                type=['csv'],
                key="predict_csv",
                help="File harus memiliki kolom yang sama dengan fitur training"
            )
            
            if predict_file is not None:
                df_predict = pd.read_csv(predict_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Data", df_predict.shape[0])
                with col2:
                    st.metric("Total Kolom", df_predict.shape[1])
                
                # Preview
                with st.expander("Preview Data"):
                    st.dataframe(df_predict.head(), use_container_width=True)
                
                # Check features
                missing_features = [f for f in st.session_state.selected_features if f not in df_predict.columns]
                
                if missing_features:
                    st.error(f"File tidak memiliki fitur yang diperlukan: {', '.join(missing_features)}")
                else:
                    st.success("Semua fitur yang diperlukan tersedia!")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        run_prediction = st.button(
                            "Prediksi Semua Data",
                            type="primary",
                            use_container_width=True
                        )
                    
                    if run_prediction:
                        # Reset filter state saat prediction baru dijalankan
                        if 'current_batch_filter' in st.session_state:
                            del st.session_state.current_batch_filter
                        
                        try:
                            X_predict = df_predict[st.session_state.selected_features].copy()
                            
                            # Impute and scale
                            X_predict_imputed = st.session_state.imputer.transform(X_predict)
                            X_predict_scaled = st.session_state.scaler.transform(X_predict_imputed)
                            
                            # Predict
                            predictions = st.session_state.trained_model.predict(X_predict_scaled)
                            
                            # Add predictions
                            df_predict['Predicted_Cluster'] = predictions
                            
                            # Show results
                            st.markdown("---")
                            st.subheader("Hasil Prediksi Batch")
                            
                            # Distribution
                            pred_counts = df_predict['Predicted_Cluster'].value_counts().sort_index()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Distribusi Cluster")
                                fig = go.Figure(data=[go.Bar(
                                    x=pred_counts.index,
                                    y=pred_counts.values,
                                    marker_color='lightgreen',
                                    text=pred_counts.values,
                                    textposition='auto'
                                )])
                                fig.update_layout(
                                    xaxis_title="Cluster",
                                    yaxis_title="Jumlah Data",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("#### Ringkasan Prediksi")
                                for cluster_id in sorted(pred_counts.index):
                                    count = pred_counts[cluster_id]
                                    percentage = count / len(df_predict) * 100
                                    
                                    # Add cluster name if available
                                    cluster_label = f"Cluster {cluster_id}"
                                    if cluster_id in CLUSTER_DESCRIPTIONS:
                                        cluster_label += f" - {CLUSTER_DESCRIPTIONS[cluster_id]['name']}"
                                    
                                    st.metric(
                                        cluster_label,
                                        f"{count} data",
                                        f"{percentage:.1f}%"
                                    )
                            
                            # Show data
                            st.markdown("---")
                            st.markdown("#### Data dengan Hasil Prediksi")
                            
                            # Filter option with cluster names
                            all_clusters = sorted(df_predict['Predicted_Cluster'].unique())
                            
                            # Create options with cluster names
                            cluster_options = []
                            for c in all_clusters:
                                if c in CLUSTER_DESCRIPTIONS:
                                    cluster_options.append(f"Cluster {c} - {CLUSTER_DESCRIPTIONS[c]['name']}")
                                else:
                                    cluster_options.append(f"Cluster {c}")
                            
                            # Initialize session state untuk filter jika belum ada
                            if 'current_batch_filter' not in st.session_state:
                                st.session_state.current_batch_filter = cluster_options.copy()
                            
                            # Gunakan session state sebagai default
                            selected_display = st.multiselect(
                                "Filter berdasarkan cluster",
                                options=cluster_options,
                                default=st.session_state.current_batch_filter,
                                key="batch_filter_widget"
                            )
                            
                            # Update session state dengan pilihan saat ini
                            st.session_state.current_batch_filter = selected_display
                            
                            # Map back to cluster IDs
                            selected_clusters = []
                            for display in selected_display:
                                cluster_id = int(display.split()[1])
                                selected_clusters.append(cluster_id)
                            
                            # Apply filter - show all data if no filter selected, otherwise show filtered data
                            if selected_clusters:
                                filtered = df_predict[df_predict['Predicted_Cluster'].isin(selected_clusters)].copy()
                                st.caption(f"Menampilkan {len(filtered)} dari {len(df_predict)} data")
                            else:
                                # When no filter is selected, show all data
                                filtered = df_predict.copy()
                                st.caption(f"Menampilkan semua data ({len(filtered)} data)")
                            
                            st.dataframe(filtered, use_container_width=True, height=400)
                            
                            # Download - use filtered data
                            if len(filtered) > 0:
                                csv = filtered.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Hasil Prediksi (CSV)",
                                    data=csv,
                                    file_name="prediction_results.csv",
                                    mime="text/csv",
                                    help=f"Download {len(filtered)} data yang ditampilkan"
                                )
                        
                        except Exception as e:
                            st.error(f"Error saat prediksi batch: {str(e)}")
                            st.exception(e)
            
            else:
                st.info("Upload file CSV untuk prediksi batch")
                
                st.markdown("### Format File yang Dibutuhkan:")
                st.markdown(f"""
                File CSV harus memiliki kolom-kolom berikut:
                """)
                
                for i, feat in enumerate(st.session_state.selected_features, 1):
                    st.markdown(f"{i}. `{feat}`")
                
                st.markdown("### Contoh Format:")
                example_df = pd.DataFrame({
                    feat: [0.0, 0.0, 0.0] for feat in st.session_state.selected_features
                })
                st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><strong>Sylva Consultant Prediction System</strong></p>
<p style='font-size: 0.9em;'>Powered by GMM & Random Forest</p>
</div>
""", unsafe_allow_html=True)