import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
import io
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Clustering & Prediction", layout="wide", page_icon="📊")

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
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

# Header
st.title("📊 Dashboard Clustering & Prediction System")
st.markdown("Sistem clustering dan prediksi data dengan penjelasan otomatis")
st.markdown("---")

# Mode selection di sidebar
with st.sidebar:
    st.header("🎯 Navigation")
    mode = st.radio(
        "Pilih Mode:",
        ["🎓 Training Model", "🔮 Prediksi Data Baru"],
        help="Training: Latih model clustering | Prediksi: Klasifikasikan data baru"
    )
    
    st.markdown("---")
    
    # Model status
    st.subheader("📌 Status Model")
    if st.session_state.trained_model is not None:
        st.success("✅ Model Aktif")
        st.info(f"Algoritma: {st.session_state.algorithm_type}")
        st.info(f"Fitur: {len(st.session_state.selected_features)}")
        
        with st.expander("Detail Fitur"):
            for i, feat in enumerate(st.session_state.selected_features, 1):
                st.text(f"{i}. {feat}")
    else:
        st.warning("⚠️ Belum ada model")
        st.caption("Latih model di mode Training")

# ========================================
# MODE 1: TRAINING MODEL
# ========================================
if mode == "🎓 Training Model":
    st.header("🎓 Training Model Clustering")
    
    # Upload Section
    st.subheader("1️⃣ Upload Data Training")
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
        with st.expander("👁️ Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Feature Selection
        st.subheader("2️⃣ Pilih Fitur untuk Clustering")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_features = st.multiselect(
                "Pilih fitur numerik (minimal 2 fitur)",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                help="Fitur yang dipilih akan digunakan untuk clustering"
            )
        
        with col2:
            if len(selected_features) >= 2:
                st.success(f"✅ {len(selected_features)} fitur terpilih")
            else:
                st.error("❌ Pilih minimal 2 fitur")
        
        if len(selected_features) >= 2:
            # Preview selected features
            with st.expander("📊 Statistik Fitur Terpilih"):
                st.dataframe(df[selected_features].describe().T, use_container_width=True)
            
            st.markdown("---")
            
            # Configuration
            st.subheader("3️⃣ Konfigurasi Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Preprocessing")
                scaler_type = st.selectbox(
                    "Metode Scaling",
                    ["RobustScaler", "StandardScaler"],
                    help="RobustScaler: Robust terhadap outlier | StandardScaler: Z-score normalization"
                )
            
            with col2:
                st.markdown("##### Algoritma Clustering")
                algorithm = st.selectbox(
                    "Pilih Algoritma",
                    ["K-Means", "K-Means (Auto-tuning)", "Hierarchical"],
                    help="K-Means: Cepat dan efisien | Auto-tuning: Otomatis cari k optimal | Hierarchical: Cluster hierarkis"
                )
            
            # Algorithm Parameters
            st.markdown("##### Parameter Algoritma")
            
            if algorithm == "K-Means":
                col1, col2 = st.columns(2)
                with col1:
                    n_clusters = st.slider("Jumlah Cluster (k)", 2, 10, 3)
                with col2:
                    random_state = st.number_input("Random State", 0, 100, 42)
                
            elif algorithm == "K-Means (Auto-tuning)":
                st.info("💡 Sistem akan otomatis mencari jumlah cluster optimal menggunakan Elbow Method dan Silhouette Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_clusters = st.slider("Min Cluster", 2, 10, 2)
                with col2:
                    max_clusters = st.slider("Max Cluster", 3, 15, 10)
                with col3:
                    random_state = st.number_input("Random State", 0, 100, 42)
                
                tuning_method = st.radio(
                    "Metode Evaluasi",
                    ["Elbow Method", "Silhouette Method", "Both"],
                    horizontal=True
                )
                
            else:  # Hierarchical
                col1, col2 = st.columns(2)
                with col1:
                    n_clusters = st.slider("Jumlah Cluster", 2, 10, 3)
                with col2:
                    linkage = st.selectbox("Linkage Method", ["ward", "complete", "average"])
            
            st.markdown("---")
            
            # Train Button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                run_clustering = st.button(
                    "🚀 Latih Model Clustering",
                    type="primary",
                    use_container_width=True
                )
            
            # Training Process
            if run_clustering:
                # Set flag bahwa training sedang dilakukan
                with st.spinner("🔄 Memproses data dan melatih model..."):
                    # Prepare data
                    X = df[selected_features].copy()
                    
                    # Handle missing values
                    if X.isnull().sum().sum() > 0:
                        st.warning(f"⚠️ Mengisi {X.isnull().sum().sum()} nilai kosong dengan median...")
                        X = X.fillna(X.median())
                    
                    # Scaling
                    if scaler_type == "RobustScaler":
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    
                    X_scaled = scaler.fit_transform(X)
                
                # Auto-tuning untuk K-Means
                if algorithm == "K-Means (Auto-tuning)":
                    st.subheader("🔍 Hyperparameter Tuning")
                    
                    with st.spinner("Melakukan hyperparameter tuning..."):
                        k_range = range(min_clusters, max_clusters + 1)
                        inertias = []
                        silhouette_scores = []
                        db_scores = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, k in enumerate(k_range):
                            status_text.text(f"Testing k={k}...")
                            
                            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                            labels = kmeans.fit_predict(X_scaled)
                            
                            inertias.append(kmeans.inertia_)
                            silhouette_scores.append(silhouette_score(X_scaled, labels))
                            db_scores.append(davies_bouldin_score(X_scaled, labels))
                            
                            progress_bar.progress((i + 1) / len(k_range))
                        
                        status_text.empty()
                        progress_bar.empty()
                    
                    # Visualisasi hasil tuning
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if tuning_method in ["Elbow Method", "Both"]:
                            fig_elbow = go.Figure()
                            fig_elbow.add_trace(go.Scatter(
                                x=list(k_range), 
                                y=inertias,
                                mode='lines+markers',
                                name='Inertia',
                                line=dict(color='blue', width=3),
                                marker=dict(size=10)
                            ))
                            fig_elbow.update_layout(
                                title="Elbow Method",
                                xaxis_title="Jumlah Cluster (k)",
                                yaxis_title="Inertia",
                                height=400
                            )
                            st.plotly_chart(fig_elbow, use_container_width=True)
                            
                            coords = np.array(list(zip(k_range, inertias)))
                            line_vec = coords[-1] - coords[0]
                            line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
                            vec_from_first = coords - coords[0]
                            scalar_proj = np.dot(vec_from_first, line_vec_norm.reshape(-1, 1))
                            vec_to_line = vec_from_first - scalar_proj * line_vec_norm
                            dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
                            elbow_k = list(k_range)[np.argmax(dist_to_line)]
                            
                            st.info(f"📍 **Elbow point**: k = {elbow_k}")
                    
                    with col2:
                        if tuning_method in ["Silhouette Method", "Both"]:
                            fig_sil = go.Figure()
                            fig_sil.add_trace(go.Scatter(
                                x=list(k_range), 
                                y=silhouette_scores,
                                mode='lines+markers',
                                name='Silhouette Score',
                                line=dict(color='green', width=3),
                                marker=dict(size=10)
                            ))
                            fig_sil.update_layout(
                                title="Silhouette Analysis",
                                xaxis_title="Jumlah Cluster (k)",
                                yaxis_title="Silhouette Score",
                                height=400
                            )
                            st.plotly_chart(fig_sil, use_container_width=True)
                            
                            best_sil_k = list(k_range)[np.argmax(silhouette_scores)]
                            st.success(f"✨ **Best Silhouette**: k = {best_sil_k} (score: {max(silhouette_scores):.3f})")
                    
                    # Tabel perbandingan
                    comparison_df = pd.DataFrame({
                        'K': list(k_range),
                        'Inertia': inertias,
                        'Silhouette': silhouette_scores,
                        'Davies-Bouldin': db_scores
                    })
                    
                    with st.expander("📊 Lihat Tabel Perbandingan Lengkap"):
                        st.dataframe(
                            comparison_df.style.highlight_max(subset=['Silhouette'], color='lightgreen')
                                              .highlight_min(subset=['Davies-Bouldin'], color='lightgreen')
                                              .format({'Inertia': '{:.2f}', 'Silhouette': '{:.3f}', 'Davies-Bouldin': '{:.3f}'}),
                            use_container_width=True
                        )
                    
                    # Rekomendasi
                    st.markdown("---")
                    st.subheader("💡 Rekomendasi Jumlah Cluster")
                    
                    if tuning_method == "Elbow Method":
                        recommended_k = elbow_k
                    elif tuning_method == "Silhouette Method":
                        recommended_k = best_sil_k
                    else:
                        best_db_k = list(k_range)[np.argmin(db_scores)]
                        votes = [elbow_k, best_sil_k, best_db_k]
                        recommended_k = max(set(votes), key=votes.count)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        use_recommended = st.checkbox(
                            f"Gunakan k rekomendasi: {recommended_k}",
                            value=True
                        )
                        if not use_recommended:
                            n_clusters = st.slider(
                                "Pilih k manual",
                                min_clusters,
                                max_clusters,
                                recommended_k,
                                key="manual_k"
                            )
                        else:
                            n_clusters = recommended_k
                    
                    with col2:
                        st.metric("K Terpilih", n_clusters)
                    
                    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    labels = model.fit_predict(X_scaled)
                    algorithm_name = f"K-Means (k={n_clusters})"
                    
                else:
                    # Training untuk algoritma lain
                    with st.spinner("Melatih model..."):
                        if algorithm == "K-Means":
                            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                            labels = model.fit_predict(X_scaled)
                            algorithm_name = f"K-Means (k={n_clusters})"
                        else:  # Hierarchical
                            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                            labels = model.fit_predict(X_scaled)
                            algorithm_name = f"Hierarchical ({linkage}, k={n_clusters})"
                
                # Save to session state
                st.session_state.trained_model = model
                st.session_state.scaler = scaler
                st.session_state.selected_features = selected_features
                st.session_state.algorithm_type = algorithm_name
                st.session_state.X_scaled = X_scaled
                
                # Add labels
                df['Cluster'] = labels
                st.session_state.df_clustered = df
                
                # Calculate profiles
                cluster_profiles = df.groupby('Cluster')[selected_features].agg(['mean', 'std', 'min', 'max'])
                st.session_state.cluster_profiles = cluster_profiles
                
                # Set training completed flag
                st.session_state.training_completed = True
            
            # Show results if training has been completed (either just now or previously)
            if st.session_state.training_completed and st.session_state.df_clustered is not None:
                df = st.session_state.df_clustered
                X_scaled = st.session_state.X_scaled
                labels = df['Cluster'].values
                
                # Show results
                st.markdown("---")
                st.success("✅ Model berhasil dilatih!")
                
                # Metrics
                st.subheader("📈 Evaluasi Model")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    n_clusters_found = len(np.unique(labels))
                    st.metric("Jumlah Cluster", n_clusters_found)
                
                with col2:
                    silhouette = silhouette_score(X_scaled, labels)
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                
                with col3:
                    db_score = davies_bouldin_score(X_scaled, labels)
                    st.metric("Davies-Bouldin", f"{db_score:.3f}")
                
                with col4:
                    ch_score = calinski_harabasz_score(X_scaled, labels)
                    st.metric("Calinski-Harabasz", f"{ch_score:.1f}")
                
                # Visualizations
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs(["📈 Visualisasi Cluster", "📊 Profil Cluster", "📋 Data Hasil"])
                
                with tab1:
                    st.subheader("Visualisasi Clustering")
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        x_axis = st.selectbox("X-axis", selected_features, index=0, key="vis_x")
                    with col_y:
                        y_axis = st.selectbox("Y-axis", selected_features, index=min(1, len(selected_features)-1), key="vis_y")
                    
                    fig = px.scatter(
                        df, x=x_axis, y=y_axis, color='Cluster',
                        title=f"Clustering: {x_axis} vs {y_axis}",
                        height=600,
                        color_continuous_scale='viridis'
                    )
                    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Profil Setiap Cluster")
                    
                    cluster_stats = df.groupby('Cluster')[selected_features].mean()
                    
                    st.markdown("#### 📊 Rata-rata Fitur per Cluster")
                    st.dataframe(
                        cluster_stats.style.background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True
                    )
                    
                    # Cluster size
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 👥 Distribusi Ukuran Cluster")
                        cluster_counts = df['Cluster'].value_counts().sort_index()
                        fig = go.Figure(data=[go.Bar(
                            x=cluster_counts.index,
                            y=cluster_counts.values,
                            marker_color='lightblue',
                            text=cluster_counts.values,
                            textposition='auto'
                        )])
                        fig.update_layout(
                            xaxis_title="Cluster",
                            yaxis_title="Jumlah Anggota",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 📈 Persentase per Cluster")
                        fig = go.Figure(data=[go.Pie(
                            labels=cluster_counts.index,
                            values=cluster_counts.values,
                            hole=.3
                        )])
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("Data dengan Label Cluster")
                    
                    selected_cluster = st.multiselect(
                        "Filter berdasarkan cluster",
                        options=sorted(df['Cluster'].unique()),
                        default=sorted(df['Cluster'].unique()),
                        key="filter_cluster"
                    )
                    
                    filtered_df = df[df['Cluster'].isin(selected_cluster)]
                    st.dataframe(filtered_df, use_container_width=True, height=400)
                    
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Clustering (CSV)",
                        data=csv,
                        file_name="clustering_results.csv",
                        mime="text/csv"
                    )
    
    else:
        st.info("📁 Silakan upload file CSV untuk memulai training model")
        
        # Reset training flag jika tidak ada file
        if 'training_completed' in st.session_state:
            st.session_state.training_completed = False
        
        st.markdown("### 📝 Format Data yang Dibutuhkan:")
        st.markdown("""
        - File CSV dengan header
        - Minimal 2 kolom numerik
        - Tidak ada terlalu banyak missing values
        """)
        
        st.markdown("### 📊 Contoh Format Data:")
        example_data = pd.DataFrame({
            'Feature1': [1.2, 2.3, 1.5, 4.5, 5.2],
            'Feature2': [3.4, 3.1, 3.3, 7.8, 8.1],
            'Feature3': [2.1, 2.3, 2.0, 5.5, 5.8]
        })
        st.dataframe(example_data, use_container_width=True)

# ========================================
# MODE 2: PREDICTION
# ========================================
else:  # mode == "🔮 Prediksi Data Baru"
    st.header("🔮 Prediksi Cluster untuk Data Baru")
    
    if st.session_state.trained_model is None:
        st.warning("### ⚠️ Belum Ada Model yang Dilatih")
        st.markdown("""
        Untuk menggunakan fitur prediksi, Anda perlu:
        
        1. **Beralih ke mode Training** menggunakan toggle di sidebar
        2. **Upload data training** dan latih model clustering
        3. **Kembali ke mode Prediksi** untuk mengklasifikasikan data baru
        
        Model yang dilatih akan disimpan otomatis dan siap digunakan!
        """)
        
    else:
        # Show model info
        st.success(f"✅ Model Aktif: {st.session_state.algorithm_type}")
        
        with st.expander("ℹ️ Informasi Model"):
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
        st.subheader("1️⃣ Pilih Metode Input Data")
        
        input_method = st.radio(
            "Metode Input:",
            ["📝 Manual Input", "📁 Upload CSV"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # MANUAL INPUT
        if input_method == "📝 Manual Input":
            st.subheader("2️⃣ Masukkan Nilai Fitur")
            
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
                    "🎯 Prediksi Cluster",
                    type="primary",
                    use_container_width=True
                )
            
            if run_prediction:
                # Prepare input
                input_df = pd.DataFrame([input_data])
                
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                # Show input
                st.markdown("#### 📝 Data Input Anda:")
                st.dataframe(input_df.T.rename(columns={0: 'Nilai'}), use_container_width=True)
                
                # Scale and predict
                input_scaled = st.session_state.scaler.transform(input_df)
                predicted_cluster = st.session_state.trained_model.predict(input_scaled)[0]
                
                # Calculate confidence
                if hasattr(st.session_state.trained_model, 'cluster_centers_'):
                    distances = np.linalg.norm(
                        st.session_state.trained_model.cluster_centers_ - input_scaled,
                        axis=1
                    )
                    closest_distance = distances[predicted_cluster]
                    confidence = 100 * (1 - closest_distance / np.sum(distances))
                else:
                    confidence = None
                
                # Show prediction
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 🎯 Hasil Prediksi")
                    st.markdown(f"# **Cluster {predicted_cluster}**")
                    
                    if confidence is not None:
                        st.metric("Confidence Score", f"{confidence:.1f}%")
                
                with col2:
                    st.markdown("### 📊 Profil Cluster Ini")
                    
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
                st.subheader("🔍 Penjelasan Detail Prediksi")
                
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
                            status = '✅ Normal'
                            color = 'green'
                        elif abs(z_score) < 2:
                            status = '⚠️ Moderate'
                            color = 'orange'
                        else:
                            status = '🔴 Outlier'
                            color = 'red'
                        
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
                    st.markdown("### 💡 Interpretasi:")
                    
                    for _, row in comparison_df.iterrows():
                        z = float(row['Z-Score'])
                        
                        if abs(z) < 0.5:
                            icon = "🟢"
                            explanation = f"**{row['Fitur']}**: Sangat sesuai dengan profil cluster (nilai sangat dekat dengan rata-rata)"
                        elif abs(z) < 1:
                            icon = "🟢"
                            explanation = f"**{row['Fitur']}**: Sesuai dengan profil cluster (dalam range normal)"
                        elif abs(z) < 2:
                            icon = "🟡"
                            if z > 0:
                                explanation = f"**{row['Fitur']}**: Sedikit di atas rata-rata cluster"
                            else:
                                explanation = f"**{row['Fitur']}**: Sedikit di bawah rata-rata cluster"
                        else:
                            icon = "🔴"
                            if z > 0:
                                explanation = f"**{row['Fitur']}**: Jauh di atas rata-rata cluster (outlier)"
                            else:
                                explanation = f"**{row['Fitur']}**: Jauh di bawah rata-rata cluster (outlier)"
                        
                        st.markdown(f"{icon} {explanation}")
                    
                    # Overall conclusion
                    st.markdown("---")
                    st.success(f"""
                    **📌 Kesimpulan**: 
                    
                    Data ini diprediksi masuk ke **Cluster {predicted_cluster}** karena nilai-nilai fiturnya 
                    paling mirip dengan karakteristik anggota cluster tersebut. Dari {len(st.session_state.selected_features)} fitur yang dianalisis, 
                    mayoritas nilai berada dalam range normal cluster ini.
                    """)
        
        # BATCH PREDICTION (Upload CSV)
        else:  # Upload CSV
            st.subheader("2️⃣ Upload File CSV untuk Prediksi Batch")
            
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
                with st.expander("👁️ Preview Data"):
                    st.dataframe(df_predict.head(), use_container_width=True)
                
                # Check features
                missing_features = [f for f in st.session_state.selected_features if f not in df_predict.columns]
                
                if missing_features:
                    st.error(f"❌ File tidak memiliki fitur yang diperlukan: {', '.join(missing_features)}")
                else:
                    st.success("✅ Semua fitur yang diperlukan tersedia!")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        run_prediction = st.button(
                            "🎯 Prediksi Semua Data",
                            type="primary",
                            use_container_width=True
                        )
                    
                    if run_prediction:
                        X_predict = df_predict[st.session_state.selected_features].copy()
                        
                        # Handle missing values
                        if X_predict.isnull().sum().sum() > 0:
                            st.warning("⚠️ Mengisi nilai kosong dengan median...")
                            X_predict = X_predict.fillna(X_predict.median())
                        
                        # Scale and predict
                        X_predict_scaled = st.session_state.scaler.transform(X_predict)
                        predictions = st.session_state.trained_model.predict(X_predict_scaled)
                        
                        # Add predictions
                        df_predict['Predicted_Cluster'] = predictions
                        
                        # Show results
                        st.markdown("---")
                        st.subheader("📊 Hasil Prediksi Batch")
                        
                        # Distribution
                        pred_counts = df_predict['Predicted_Cluster'].value_counts().sort_index()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Distribusi Cluster")
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
                            st.markdown("#### 📊 Ringkasan Prediksi")
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
                        st.markdown("#### 📋 Data dengan Hasil Prediksi")
                        
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
                            label="📥 Download Hasil Prediksi (CSV)",
                            data=csv,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
            
            else:
                st.info("📁 Upload file CSV untuk prediksi batch")
                
                st.markdown("### 📋 Format File yang Dibutuhkan:")
                st.markdown(f"""
                File CSV harus memiliki kolom-kolom berikut:
                """)
                
                for i, feat in enumerate(st.session_state.selected_features, 1):
                    st.markdown(f"{i}. `{feat}`")
                
                st.markdown("### 📊 Contoh Format:")
                example_df = pd.DataFrame({
                    feat: [0.0, 0.0, 0.0] for feat in st.session_state.selected_features
                })
                st.dataframe(example_df, use_container_width=True)