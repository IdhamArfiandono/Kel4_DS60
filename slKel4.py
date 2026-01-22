import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
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
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []

# Header
st.title("Dashboard Clustering & Prediction System")
st.markdown("Sistem clustering dengan algoritma **GMM Tuned** dan integrasi **Random Forest**")
st.markdown("---")

# Helper function
def calculate_distance_to_centroid(X, labels, model):
    """Calculate distance from each point to its assigned cluster centroid"""
    distances = np.zeros(X.shape[0])
    # For GMM, cluster centers are the means
    centers = model.means_
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        cluster_points = X[mask]
        centroid = centers[label]
        # Euclidean distance
        dist = np.linalg.norm(cluster_points - centroid, axis=1)
        distances[mask] = dist
        
    return distances

# Mode selection di sidebar
with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Pilih Mode:",
        ["Training Model", "Prediksi Data Baru"],
        help="Training: Latih model clustering | Prediksi: Klasifikasikan data baru"
    )
    
    st.markdown("---")
    
    # Model status
    st.subheader("Status Model")
    if st.session_state.trained_model is not None:
        st.success("Model Aktif")
        st.info(f"Algoritma: {st.session_state.algorithm_type}")
        st.info(f"Fitur: {len(st.session_state.selected_features)}")
        
        with st.expander("Detail Fitur"):
            for i, feat in enumerate(st.session_state.selected_features, 1):
                st.text(f"{i}. {feat}")
    else:
        st.warning("Belum ada model")
        st.caption("Latih model di mode Training")

# ========================================
# MODE 1: TRAINING MODEL
# ========================================
if mode == "Training Model":
    st.header("Training Model Clustering (GMM Tuned + Random Forest)")
    
    # Upload Section
    st.subheader("1. Upload Data Training")
    uploaded_file = st.file_uploader(
        "Upload file CSV untuk training",
        type=['csv'],
        help="File harus berformat CSV dengan header"
    )
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Baris", df.shape[0])
        with col2:
            st.metric("Total Kolom", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Preview data
        with st.expander("Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Feature Selection
        st.subheader("2. Pilih Fitur untuk Clustering")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_features = st.multiselect(
                "Pilih fitur numerik (minimal 2 fitur)",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                help="Fitur yang dipilih akan digunakan untuk clustering"
            )
        
        with col2:
            if len(selected_features) >= 2:
                st.success(f"{len(selected_features)} fitur terpilih")
            else:
                st.error("Pilih minimal 2 fitur")
        
        if len(selected_features) >= 2:
            # Preview selected features
            with st.expander("Statistik Fitur Terpilih"):
                st.dataframe(df[selected_features].describe().T, use_container_width=True)
            
            st.markdown("---")
            
            # Configuration
            st.subheader("3. Konfigurasi Model")
            
            st.info("Sistem menggunakan **GMM (Gaussian Mixture Model)** dengan opsi Auto-Tuning dan validasi **Random Forest**.")
            
            # Parameters
            st.markdown("##### Parameter GMM Tuned")
            
            n_clusters = 3
            st.info(f"Jumlah Cluster diset tetap ke: **{n_clusters}**")
            
            col1, col2 = st.columns(2)
            with col1:
                covariance_type = st.selectbox(
                    "Covariance Type",
                    ['full', 'tied', 'diag', 'spherical'],
                    index=0
                )
            with col2:
                n_init = st.slider("N Init (Restarts)", 5, 20, 10)
            
            st.markdown("---")
            
            # Train Button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                run_clustering = st.button(
                    "Latih Model GMM & RF",
                    type="primary",
                    use_container_width=True
                )
            
            # Training Process
            if run_clustering:
                with st.spinner("Memproses data dan melatih model (3 Cluster)..."):
                    try:
                        # Prepare data
                        X = df[selected_features].copy()
                        
                        # Handle missing values with imputation
                        imputer = SimpleImputer(strategy='median')
                        X_imputed = imputer.fit_transform(X)
                        X = pd.DataFrame(X_imputed, columns=selected_features)
                        
                        st.info(f"Imputation completed: {len(selected_features)} features")
                        
                        # Scaling with RobustScaler
                        scaler = RobustScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Training GMM
                        # Fixed to 3 clusters
                        best_k = n_clusters
                        
                        model = GaussianMixture(
                            n_components=n_clusters,
                            covariance_type=covariance_type,
                            n_init=n_init,
                            random_state=42
                        )
                        model.fit(X_scaled)
                            
                        labels = model.predict(X_scaled)
                        algorithm_name = f"GMM Tuned (K={best_k}, {covariance_type})"
                        
                        # ----------------------------------------
                        # RANDOM FOREST INTEGRATION
                        # ----------------------------------------
                        st.text("Melatih Random Forest untuk Feature Importance...")
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X_scaled, labels)
                        
                        feature_importances = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': rf.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # ----------------------------------------
                        # DISTANCE TO CENTROID
                        # ----------------------------------------
                        distances = calculate_distance_to_centroid(X_scaled, labels, model)
                        
                        # Save to session state
                        st.session_state.trained_model = model
                        st.session_state.rf_model = rf
                        st.session_state.feature_importances = feature_importances
                        st.session_state.scaler = scaler
                        st.session_state.imputer = imputer
                        st.session_state.selected_features = selected_features
                        st.session_state.algorithm_type = algorithm_name
                        st.session_state.X_scaled = X_scaled
                        st.session_state.feature_names = selected_features
                        
                        # Add results to dataframe
                        df['Cluster'] = labels
                        df['Distance_to_Centroid'] = distances
                        st.session_state.df_clustered = df
                        
                        # Calculate profiles
                        cluster_profiles = df.groupby('Cluster')[selected_features].agg(['mean', 'std', 'min', 'max'])
                        st.session_state.cluster_profiles = cluster_profiles
                        
                        # Set training completed
                        st.session_state.training_completed = True
                        
                        st.success("Model Clustering & Random Forest berhasil dilatih!")
                        
                    except Exception as e:
                        st.error(f"Error saat training: {str(e)}")
                        st.exception(e)
            
            # Show results if training completed
            if st.session_state.training_completed and st.session_state.df_clustered is not None:
                df = st.session_state.df_clustered
                X_scaled = st.session_state.X_scaled
                labels = df['Cluster'].values
                
                # Metrics
                st.markdown("---")
                st.subheader("Evaluasi Model")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    n_clusters_found = len(np.unique(labels))
                    st.metric("Jumlah Cluster", n_clusters_found)
                
                with col2:
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(X_scaled, labels)
                        st.metric("Silhouette Score", f"{silhouette:.4f}")
                    else:
                        st.metric("Silhouette Score", "N/A")
                
                with col3:
                    if len(np.unique(labels)) > 1:
                        db_score = davies_bouldin_score(X_scaled, labels)
                        st.metric("Davies-Bouldin", f"{db_score:.4f}")
                    else:
                         st.metric("Davies-Bouldin", "N/A")
                
                with col4:
                    if len(np.unique(labels)) > 1:
                        ch_score = calinski_harabasz_score(X_scaled, labels)
                        st.metric("Calinski-Harabasz", f"{ch_score:.1f}")
                    else:
                        st.metric("Calinski-Harabasz", "N/A")

                
                # Visualizations
                st.markdown("---")
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Visualisasi Cluster",
                    "Profil Cluster",
                    "Feature Importance (RF)",
                    "PCA 2D & 3D",
                    "Data Hasil"
                ])
                
                with tab1:
                    st.subheader("Visualisasi Clustering")
                    
                    if len(st.session_state.selected_features) >= 2:
                        col_x, col_y = st.columns(2)
                        with col_x:
                            x_axis = st.selectbox(
                                "X-axis",
                                st.session_state.selected_features,
                                index=0,
                                key="vis_x"
                            )
                        with col_y:
                            y_axis = st.selectbox(
                                "Y-axis",
                                st.session_state.selected_features,
                                index=min(1, len(st.session_state.selected_features)-1),
                                key="vis_y"
                            )
                        
                        fig = px.scatter(
                            df, x=x_axis, y=y_axis, color='Cluster',
                            title=f"Clustering: {x_axis} vs {y_axis}",
                            height=600,
                            color_continuous_scale='viridis',
                            hover_data=['Distance_to_Centroid']
                        )
                        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Profil Setiap Cluster")
                    
                    cluster_stats = df.groupby('Cluster')[st.session_state.selected_features].mean()
                    
                    st.markdown("#### Rata-rata Fitur per Cluster")
                    st.dataframe(
                        cluster_stats.style.background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True
                    )
                    
                    # Detailed characteristics
                    st.markdown("#### Karakteristik Detail per Cluster")
                    cluster_counts = df['Cluster'].value_counts().sort_index()
                    for cluster_id in sorted(df['Cluster'].unique()):
                        with st.expander(f"**Cluster {cluster_id}** ({cluster_counts[cluster_id]} anggota)"):
                            cluster_data = df[df['Cluster'] == cluster_id][st.session_state.selected_features]
                            
                            stats_df = pd.DataFrame({
                                'Mean': cluster_data.mean(),
                                'Std': cluster_data.std(),
                                'Min': cluster_data.min(),
                                'Max': cluster_data.max()
                            })
                            st.dataframe(stats_df.style.format('{:.2f}'), use_container_width=True)

                with tab3:
                    st.subheader("Feature Importance (Random Forest)")
                    
                    if st.session_state.feature_importances is not None:
                        fi_df = st.session_state.feature_importances
                        
                        fig_fi = px.bar(
                            fi_df, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title="Kontribusi Fitur terhadap Pembentukan Cluster",
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_fi, use_container_width=True)
                        
                        st.info("**Interpretasi**: Fitur dengan nilai importance tertinggi adalah fitur yang paling membedakan/mempengaruhi pembentukan cluster.")
                    else:
                        st.warning("Feature importance belum tersedia.")
                
                with tab4:
                    st.subheader("Visualisasi PCA")
                    
                    # PCA 2D
                    st.markdown("#### PCA 2D Projection")
                    pca_2d = PCA(n_components=2, random_state=42)
                    X_pca_2d = pca_2d.fit_transform(X_scaled)
                    
                    df_pca_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
                    df_pca_2d['Cluster'] = labels
                    
                    fig = px.scatter(
                        df_pca_2d, x='PC1', y='PC2', color='Cluster',
                        title=f"PCA 2D (Variance Explained: {pca_2d.explained_variance_ratio_.sum():.2%})",
                        height=500
                    )
                    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"PC1: {pca_2d.explained_variance_ratio_[0]:.2%} | PC2: {pca_2d.explained_variance_ratio_[1]:.2%}")
                    
                    # PCA 3D
                    st.markdown("#### PCA 3D Projection")
                    pca_3d = PCA(n_components=3, random_state=42)
                    X_pca_3d = pca_3d.fit_transform(X_scaled)
                    
                    df_pca_3d = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
                    df_pca_3d['Cluster'] = labels
                    
                    fig_3d = px.scatter_3d(
                        df_pca_3d, x='PC1', y='PC2', z='PC3', color='Cluster',
                        title=f"PCA 3D (Variance Explained: {pca_3d.explained_variance_ratio_.sum():.2%})",
                        height=600
                    )
                    fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.3, color='white')))
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    st.info(f"PC1: {pca_3d.explained_variance_ratio_[0]:.2%} | PC2: {pca_3d.explained_variance_ratio_[1]:.2%} | PC3: {pca_3d.explained_variance_ratio_[2]:.2%}")
                
                with tab5:
                    st.subheader("Data dengan Label Cluster & Distance")
                    
                    selected_cluster = st.multiselect(
                        "Filter berdasarkan cluster",
                        options=sorted(df['Cluster'].unique()),
                        default=sorted(df['Cluster'].unique()),
                        key="filter_cluster"
                    )
                    
                    filtered_df = df[df['Cluster'].isin(selected_cluster)]
                    
                    st.markdown("#### Data Hasil (Preview 100 baris)")
                    st.dataframe(filtered_df.head(100), use_container_width=True, height=400)
                    
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Hasil Clustering (CSV)",
                        data=csv,
                        file_name="clustering_results_gmm_rf.csv",
                        mime="text/csv"
                    )
    
    else:
        st.info("Silakan upload file CSV untuk memulai training model")
        
        # Reset training flag
        if 'training_completed' in st.session_state:
            st.session_state.training_completed = False
        
        st.markdown("### Format Data yang Dibutuhkan:")
        st.markdown("""
        - File CSV dengan header
        - Minimal 2 kolom numerik
        - Missing values akan di-handle otomatis dengan median imputation
        """)
        
        st.markdown("### Algoritma yang Digunakan:")
        st.markdown("""
        1. **GMM Tuned (Gaussian Mixture Model)**
           - Algoritma clustering utama yang digunakan
           - Dilengkapi dengan *Auto-Tuning* untuk mencari jumlah cluster optimal secara otomatis
           
        2. **Random Forest Classifier**
           - Digunakan untuk validasi dan analisis *Feature Importance*
           - Membantu menjelaskan fitur mana yang paling berpengaruh dalam pembentukan cluster
        """)

# ========================================
# MODE 2: PREDICTION
# ========================================
else:  # mode == "Prediksi Data Baru"
    st.header("Prediksi Cluster untuk Data Baru")
    
    if st.session_state.trained_model is None:
        st.warning("### Belum Ada Model yang Dilatih")
        st.markdown("""
        Untuk menggunakan fitur prediksi, Anda perlu:
        
        1. **Beralih ke mode Training** menggunakan toggle di sidebar
        2. **Upload data training** dan latih model clustering
        3. **Kembali ke mode Prediksi** untuk mengklasifikasikan data baru
        
        Model yang dilatih akan disimpan otomatis dan siap digunakan!
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
                if st.session_state.df_clustered is not None:
                    st.markdown("**Jumlah Cluster:**")
                    n_clusters = len(st.session_state.df_clustered['Cluster'].unique())
                    st.metric("Total Cluster", n_clusters)
        
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
            
            input_data = {}
            
            # Create input form in columns
            n_features = len(st.session_state.selected_features)
            n_cols = min(3, n_features)
            cols = st.columns(n_cols)
            
            for i, feature in enumerate(st.session_state.selected_features):
                with cols[i % n_cols]:
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
                    
                    # Calculate distance to centroid
                    if hasattr(st.session_state.trained_model, 'means_'):
                        centroid = st.session_state.trained_model.means_[predicted_cluster]
                        dist_to_centroid = np.linalg.norm(input_scaled - centroid)
                    else:
                        dist_to_centroid = 0.0
                    
                    # Calculate confidence (if available)
                    confidence = None
                    if hasattr(st.session_state.trained_model, 'predict_proba'):
                        proba = st.session_state.trained_model.predict_proba(input_scaled)[0]
                        confidence = proba[predicted_cluster] * 100
                    elif hasattr(st.session_state.trained_model, 'cluster_centers_'):
                        distances = np.linalg.norm(
                            st.session_state.trained_model.cluster_centers_ - input_scaled,
                            axis=1
                        )
                        closest_distance = distances[predicted_cluster]
                        confidence = 100 * (1 - closest_distance / np.sum(distances))
                    
                    # Show prediction
                    st.markdown("---")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### Hasil Prediksi")
                        st.markdown(f"# **Cluster {predicted_cluster}**")
                        
                        if confidence is not None:
                            st.metric("Confidence Score", f"{confidence:.1f}%")
                        
                        st.metric("Distance to Centroid", f"{dist_to_centroid:.4f}")
                    
                    with col2:
                        st.markdown("### Profil Cluster Ini")
                        
                        if st.session_state.df_clustered is not None:
                            cluster_data = st.session_state.df_clustered[
                                st.session_state.df_clustered['Cluster'] == predicted_cluster
                            ][st.session_state.selected_features]
                            
                            profile_df = pd.DataFrame({
                                'Mean': cluster_data.mean(),
                                'Std': cluster_data.std(),
                                'Min': cluster_data.min(),
                                'Max': cluster_data.max()
                            })
                            
                            st.dataframe(profile_df.style.format('{:.2f}'), use_container_width=True)
                    
                    # Detailed explanation
                    st.markdown("---")
                    st.subheader("Penjelasan Detail Prediksi")
                    
                    if st.session_state.df_clustered is not None:
                        cluster_data = st.session_state.df_clustered[
                            st.session_state.df_clustered['Cluster'] == predicted_cluster
                        ][st.session_state.selected_features]
                        
                        comparison_data = []
                        for feature in st.session_state.selected_features:
                            input_value = input_data[feature]
                            cluster_mean = cluster_data[feature].mean()
                            cluster_std = cluster_data[feature].std()
                            cluster_min = cluster_data[feature].min()
                            cluster_max = cluster_data[feature].max()
                            
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
                                icon = ""
                                explanation = f"**{row['Fitur']}**: Sangat sesuai dengan profil cluster (nilai sangat dekat dengan rata-rata)"
                            elif abs(z) < 1:
                                icon = ""
                                explanation = f"**{row['Fitur']}**: Sesuai dengan profil cluster (dalam range normal)"
                            elif abs(z) < 2:
                                icon = ""
                                if z > 0:
                                    explanation = f"**{row['Fitur']}**: Sedikit di atas rata-rata cluster"
                                else:
                                    explanation = f"**{row['Fitur']}**: Sedikit di bawah rata-rata cluster"
                            else:
                                icon = ""
                                if z > 0:
                                    explanation = f"**{row['Fitur']}**: Jauh di atas rata-rata cluster (outlier)"
                                else:
                                    explanation = f"**{row['Fitur']}**: Jauh di bawah rata-rata cluster (outlier)"
                            
                            st.markdown(f"{icon} {explanation}")
                        
                        # Overall conclusion
                        st.markdown("---")
                        st.success(f"""
                        **Kesimpulan**: 
                        
                        Data ini diprediksi masuk ke **Cluster {predicted_cluster}** karena nilai-nilai fiturnya 
                        paling mirip dengan karakteristik anggota cluster tersebut. Dari {len(st.session_state.selected_features)} fitur yang dianalisis, 
                        mayoritas nilai berada dalam range normal cluster ini.
                        """)
                
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
                        try:
                            X_predict = df_predict[st.session_state.selected_features].copy()
                            
                            # Impute and scale
                            X_predict_imputed = st.session_state.imputer.transform(X_predict)
                            X_predict_scaled = st.session_state.scaler.transform(X_predict_imputed)
                            
                            # Predict
                            predictions = st.session_state.trained_model.predict(X_predict_scaled)
                            
                            # Calculate distances
                            if hasattr(st.session_state.trained_model, 'means_'):
                                distances = []
                                means = st.session_state.trained_model.means_
                                for i, pred_label in enumerate(predictions):
                                    centroid = means[pred_label]
                                    point = X_predict_scaled[i]
                                    dist = np.linalg.norm(point - centroid)
                                    distances.append(dist)
                            else:
                                distances = [0.0] * len(predictions)
                            
                            # Add predictions
                            df_predict['Predicted_Cluster'] = predictions
                            df_predict['Distance_to_Centroid'] = distances
                            
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
                                    st.metric(
                                        f"Cluster {cluster_id}",
                                        f"{count} data",
                                        f"{percentage:.1f}%"
                                    )
                            
                            # Show data
                            st.markdown("---")
                            st.markdown("#### Data dengan Hasil Prediksi")
                            
                            # Filter option
                            selected_clusters = st.multiselect(
                                "Filter berdasarkan cluster",
                                options=sorted(df_predict['Predicted_Cluster'].unique()),
                                default=sorted(df_predict['Predicted_Cluster'].unique()),
                                key="batch_filter"
                            )
                            
                            filtered = df_predict[df_predict['Predicted_Cluster'].isin(selected_clusters)]
                            st.dataframe(filtered, use_container_width=True, height=400)
                            
                            # Download
                            csv = df_predict.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Hasil Prediksi (CSV)",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv"
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
    <p><strong>Dashboard Clustering & Prediction System</strong></p>
    <p>Powered by Streamlit | Advanced Clustering Algorithms</p>
</div>
""", unsafe_allow_html=True)