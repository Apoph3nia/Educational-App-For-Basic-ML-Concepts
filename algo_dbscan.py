import streamlit as st, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from utils import apply_global_styles
from conclusions import render_conclusion
from quizzes import render_quiz

def show_dbscan():
    apply_global_styles()
    RANDOM_SEED = 11
    
#   TITLE & THEORY
    st.title("🔵 DBSCAN Clustering", anchor = False)
    
    st.markdown("""
                Ο αλγόριθμος **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) 
                είναι ένας density-based αλγόριθμος clustering που δεν χρειάζεται να ορίσουμε τον αριθμό clusters!
                
                ### 🎯 Κεντρική Ιδέα: "Πυκνότητα"
                
                Ο DBSCAN ομαδοποιεί σημεία που είναι **πυκνά μεταξύ τους** και σημειώνει ως **θόρυβο (noise)**
                τα σημεία που βρίσκονται σε αραιές περιοχές.
                """)
    st.divider()
    
#   THEORY: DBSCAN CONCEPTS
    with st.expander("📚 Βασικές Έννοιες του DBSCAN", expanded = False):
        st.markdown("""### 🔑 Τρεις τύποι σημείων:""")
        
        col_types1, col_types2, col_types3 = st.columns(3)
        
        with col_types1:
            st.success("""
                    **Core Point (Πυρήνας)**
                    
                    Ένα σημείο είναι **core point** αν έχει τουλάχιστον **minPts** σημεία 
                    εντός απόστασης **ε** (epsilon).
                    
                    👉 *Αυτά είναι τα "σημαντικά" σημεία που ξεκινούν clusters.*
                    """)
        
        with col_types2:
            st.info("""
                    **Border Point (Σύνορο)**
                    
                    Ένα σημείο είναι **border point** αν:
                    * Δεν είναι core point
                    * Αλλά βρίσκεται εντός της ακτίνας ε ενός core point
                    
                    👉 *Αυτά είναι τα "περιφερειακά" σημεία.*
                    """)
        
        with col_types3:
            st.error("""
                    **Noise Point (Θόρυβος)**
                    
                    Ένα σημείο είναι **noise** αν:
                    * Δεν είναι core point
                    * Δεν είναι border point
                    
                    👉 *Αυτά είναι τα "ακραία" σημεία - outliers.*
                    """)
        
        st.divider()
        st.markdown("### 📐 Παράμετροι:")
        
        col_params1, col_params2 = st.columns(2)
        
        with col_params1:
            st.markdown("""
                        **ε (Epsilon)**
                        
                        Η ακτίνα αναζήτησης γειτόνων.
                        
                        * Μικρό ε → Πολλά μικρά clusters, περισσότερα noise
                        * Μεγάλο ε → Λιγότερα clusters, λιγότερα noise
                        """)
        
        with col_params2:
            st.markdown("""
                        **minPts (Minimum Points)**
                        
                        Ελάχιστος αριθμός γειτόνων για core point.
                        
                        * Μικρό minPts → Πιο επιεικής, περισσότερα core points
                        * Μεγάλο minPts → Πιο αυστηρός, λιγότερα clusters
                        """)
    st.divider()
    
#   SIDEBAR: DATA GENERATION & DBSCAN PARAMETERS
    st.sidebar.header("1. Δημιουργία Δεδομένων")
    
    #   Dataset selection
    dataset_type = st.sidebar.selectbox(
                                        "Τύπος Δεδομένων:",
                                        ["Moons", "Circles", "Blobs", "Varied Density"],
                                        key = "dbscan_dataset"
                                        )
    
    n_samples = st.sidebar.slider("Αριθμός Σημείων", 100, 300, 200, key = "dbscan_samples")
    noise_level = st.sidebar.slider("Θόρυβος Δεδομένων", 0.0, 0.2, 0.05, 0.01, key = "dbscan_data_noise")
    
    # Generate data based on selection
    if dataset_type == "Moons":
        X, _ = make_moons(n_samples = n_samples, noise = noise_level, random_state = RANDOM_SEED )
    elif dataset_type == "Circles":
        X, _ = make_circles(n_samples = n_samples, noise = noise_level, factor = 0.5, random_state = RANDOM_SEED   )
    elif dataset_type == "Blobs":
        X, _ = make_blobs(n_samples = n_samples, centers = 3, cluster_std = 1.0, random_state = RANDOM_SEED    )
    else:
        #   Create clusters with different densities
        cluster1 = np.random.randn(100, 2) * 0.3 + [0, 0]
        cluster2 = np.random.randn(100, 2) * 0.8 + [3, 3]
        cluster3 = np.random.randn(100, 2) * 0.5 + [0, 4]
        X = np.vstack([cluster1, cluster2, cluster3])
    
    st.sidebar.divider()
    
#   DBSCAN PARAMETERS
    st.sidebar.header("2. Παράμετροι DBSCAN")
    
    eps = st.sidebar.slider("ε (Epsilon)", 0.1, 2.0, 0.5, 0.05, key = "dbscan_eps")
    min_samples = st.sidebar.slider("minPts", 2, 20, 5, key = "dbscan_minpts")
    
    st.sidebar.markdown("""
                        **💡 Tips:**
                        * Ξεκίνα με ε = 0.5
                        * minPts ≥ dimensions + 1
                        * Προσαρμογή βάσει αποτελέσματος
                        """)
    
#   EPSILON CIRCLES VISUALIZATION
    st.header("🔍 Εξερεύνηση Epsilon")
    
    st.markdown("""Διάλεξε ένα σημείο για να δεις την **ε-γειτονικότητα** (epsilon neighborhood) του:""")
    
    #   Select point to examine
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        point_idx = st.slider("Επίλεξε Σημείο (ID)", 0, len(X)-1, 0, key = "dbscan_point_idx")
    
    #   Count neighbors within epsilon
    distances = np.sqrt(np.sum((X - X[point_idx])**2, axis = 1))
    neighbors_count = np.sum(distances <= eps) - 1              #   Exclude the point itself
    is_core = neighbors_count >= min_samples
    
    with col_select2:
        if is_core:
            st.success(f"✅ **Core Point** με {neighbors_count} γείτονες (minPts = {min_samples})")
        else:
            if neighbors_count > 0:
                st.warning(f"⚠️ **Border Point** με {neighbors_count} γείτονες (< {min_samples})")
            else:
                st.error(f"❌ **Noise Point** με {neighbors_count} γείτονες")
    
    #   Plot with epsilon circle
    fig_eps, ax_eps = plt.subplots(figsize = (10, 8))
    
    #   Plot all points
    ax_eps.scatter(X[:, 0], X[:, 1], c = 'lightgray', s = 50, alpha = 0.5, label = 'Άλλα σημεία')
    
    #   Highlight neighbors
    neighbor_mask = distances <= eps
    ax_eps.scatter(X[neighbor_mask, 0], X[neighbor_mask, 1], c = 'green', s = 80, alpha = 0.7, label = 'Γείτονες')
    
    #   Highlight selected point
    ax_eps.scatter(X[point_idx, 0], X[point_idx, 1], c = 'red', s = 200, marker = '*', edgecolors = 'black', linewidths = 2, label = 'Επιλεγμένο Σημείο', zorder = 5)
    
    #   Draw epsilon circle
    circle = plt.Circle((X[point_idx, 0], X[point_idx, 1]), eps, fill = False, color = 'red', linewidth = 2, linestyle = '--', label = f'ε = {eps}')
    ax_eps.add_patch(circle)
    
    #   Fill epsilon region
    circle_fill = plt.Circle((X[point_idx, 0], X[point_idx, 1]), eps, fill = True, color = 'red', alpha = 0.1)
    ax_eps.add_patch(circle_fill)
    
    ax_eps.set_xlabel('Feature 1')
    ax_eps.set_ylabel('Feature 2')
    ax_eps.set_title(f'Epsilon Neighborhood (ε = {eps}, minPts = {min_samples})')
    ax_eps.legend(loc='upper right')
    ax_eps.grid(True, alpha = 0.3)
    ax_eps.set_aspect('equal')
    
    st.pyplot(fig_eps)
    plt.close(fig_eps)

#   DBSCAN CLUSTERING
    st.divider()
    st.header("📊 DBSCAN Clustering Result")
    
    #   Run DBSCAN
    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    labels = dbscan.fit_predict(X)
    
    #   Number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    #   Create color map
    unique_labels = set(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    #   Plot DBSCAN result
    fig_db, ax_db = plt.subplots(figsize = (10, 8))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            #   Black for noise | No edge for unfilled markers
            col, marker, s, label, edgecolors, lw = 'black', 'x', 50, 'Noise', None, 1.5
        else:
            marker, s, label, edgecolors, lw = 'o', 80, f'Cluster {k}', 'white', 0.5
        
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax_db.scatter(
            xy[:, 0], xy[:, 1], 
            c = [col], 
            s = s, 
            marker = marker, 
            alpha = 0.7 if k != -1 else 1, 
            label = label, 
            edgecolors = edgecolors, 
            linewidths = lw
        )
    ax_db.set_xlabel('Feature 1')
    ax_db.set_ylabel('Feature 2')
    ax_db.set_title(f'DBSCAN: {n_clusters} Clusters, {n_noise} Noise Points')
    ax_db.legend(loc = 'upper right')
    ax_db.grid(True, alpha = 0.3)
    
    st.pyplot(fig_db)
    plt.close(fig_db)
    
    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric("Clusters", n_clusters, help = "Ο αριθμός των διακριτών ομάδων που εντόπισε ο αλγόριθμος.\n\nΔεν συμπεριλαμβάνονται τα σημεία θορύβου.")
    
    with col_m2:
        st.metric("Noise Points", n_noise, f"{n_noise/len(X):.1%}", help = "Το πλήθος και το ποσοστό των σημείων (outliers) που δεν πληρούσαν\n\n τα κριτήρια πυκνότητας για να ενταχθούν σε κάποιο cluster.")
    
    with col_m3:
        if n_clusters > 1:
            #   Exclude noise for silhouette
            mask = labels != -1
            if np.sum(mask) > n_clusters:
                silhouette = silhouette_score(X[mask], labels[mask])
                st.metric("Silhouette Score", f"{silhouette:.3f}", help = "Κυμαίνεται από -1 έως 1 (όσο υψηλότερο, τόσο καλύτερο).\n\nΜετράει την ποιότητα του clustering.\n\nΕδώ υπολογίζεται εξαιρώντας τα σημεία θορύβου.")
            else:
                st.metric("Silhouette Score", "N/A", help = "Απαιτούνται τουλάχιστον 2 clusters και επαρκή σημεία (εκτός θορύβου) για τον υπολογισμό.")
        else:
            st.metric("Silhouette Score", "N/A", help = "Δεν μπορεί να υπολογιστεί με λιγότερα από 2 clusters.")

#   PARAMETER SENSITIVITY
    st.divider()
    st.header("📈 Επίδραση Παραμέτρων")
    
    st.markdown("""Δες πώς αλλάζουν τα αποτελέσματα με διαφορετικές τιμές ε και minPts:""")
    
    #   Epsilon comparison
    st.subheader("🔬 Διαφορετικά Epsilon")
    
    eps_values = [0.3, 0.5, 0.8, 1.0]
    
    fig_eps_comp, axes_eps_comp = plt.subplots(1, 4, figsize = (16, 4))
    
    for idx, eps_comp in enumerate(eps_values):
        db_comp = DBSCAN(eps = eps_comp, min_samples = min_samples)
        labels_comp = db_comp.fit_predict(X)
        n_clust_comp = len(set(labels_comp)) - (1 if -1 in labels_comp else 0)
        n_noise_comp = list(labels_comp).count(-1)
        
        unique_comp = set(labels_comp)
        colors_comp = plt.cm.Set1(np.linspace(0, 1, len(unique_comp)))
        
        for k, col in zip(unique_comp, colors_comp):
            if k == -1:
                col, marker = 'black', 'x'
            else:
                marker = 'o'
            
            mask = labels_comp == k
            axes_eps_comp[idx].scatter(X[mask, 0], X[mask, 1], c = [col], s = 30, marker = marker, alpha = 0.7)
        
        axes_eps_comp[idx].set_title(f'ε = {eps_comp}\n{n_clust_comp} clusters, {n_noise_comp} noise')
        axes_eps_comp[idx].set_xlabel('Feature 1')
        axes_eps_comp[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    st.pyplot(fig_eps_comp)
    plt.close(fig_eps_comp)
    
    #   MinPts comparison
    st.subheader("🔬 Διαφορετικά minPts")
    
    minpts_values = [3, 5, 10, 15]
    
    fig_minpts_comp, axes_minpts_comp = plt.subplots(1, 4, figsize = (16, 4))
    
    for idx, minpts_comp in enumerate(minpts_values):
        db_comp = DBSCAN(eps = eps, min_samples = minpts_comp)
        labels_comp = db_comp.fit_predict(X)
        n_clust_comp = len(set(labels_comp)) - (1 if -1 in labels_comp else 0)
        n_noise_comp = list(labels_comp).count(-1)
        
        unique_comp = set(labels_comp)
        colors_comp = plt.cm.Set1(np.linspace(0, 1, len(unique_comp)))
        
        for k, col in zip(unique_comp, colors_comp):
            if k == -1:
                col, marker = 'black', 'x'
            else:
                marker = 'o'
            
            mask = labels_comp == k
            axes_minpts_comp[idx].scatter(X[mask, 0], X[mask, 1], c = [col], s = 30, marker = marker, alpha = 0.7)
        
        axes_minpts_comp[idx].set_title(f'minPts = {minpts_comp}\n{n_clust_comp} clusters, {n_noise_comp} noise')
        axes_minpts_comp[idx].set_xlabel('Feature 1')
        axes_minpts_comp[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    st.pyplot(fig_minpts_comp)
    plt.close(fig_minpts_comp)
    
#   K-DISTANCE GRAPH (Knee Method)
    st.divider()
    st.header("📐 K-Distance Graph (Εύρεση βέλτιστου ε)")
    
    st.markdown("""
                Το **K-Distance Graph** μας βοηθά να βρούμε μια καλή τιμή για το ε:
                
                1. Για κάθε σημείο, υπολογίζουμε την απόσταση στον k-οστό κοντινότερο γείτονα
                2. Ταξινομούμε τις αποστάσεις σε φθίνουσα σειρά
                3. Ψάχνουμε το "γόνατο" (knee) στο γράφημα
                """)
    
    #   Calculate k-distances
    k_distance = min_samples
    distances_matrix = np.zeros((len(X), len(X)))
    
    #   Calculate pairwise distances
    for i in range(len(X)):
        for j in range(len(X)):
            distances_matrix[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
    
    #   Sort distances for each point
    k_distances = np.sort(distances_matrix, axis = 1)[:, k_distance]
    k_distances_sorted = np.sort(k_distances)[::-1]
    
    fig_kdist, ax_kdist = plt.subplots(figsize = (10, 5))
    ax_kdist.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-', linewidth = 2)
    ax_kdist.axhline(y = eps, color = 'red', linestyle = '--', label = f'Current ε = {eps}')
    ax_kdist.set_xlabel('Σημεία (ταξινομημένα)')
    ax_kdist.set_ylabel(f'Απόσταση στον {k_distance}ο Γείτονα (ε)')
    ax_kdist.set_title('K-Distance Graph')
    ax_kdist.legend()
    ax_kdist.grid(True, alpha = 0.3)
    
    st.pyplot(fig_kdist)
    plt.close(fig_kdist)
    
    st.info(f"""
            💡 **Συμβουλή:** Το "γόνατο" του γραφήματος δείχνει μια καλή τιμή για το ε.
            Αν η γραμμή πέφτει απότομα, σημαίνει ότι μετά από εκείνα τα σημεία η πυκνότητα μειώνεται σημαντικά.
            """)

#   COMPARISON WITH K-MEANS
    with st.expander("🔬 DBSCAN vs K-Means", expanded = False):
        st.markdown("""
                    ### Σύγκριση των δύο αλγορίθμων
                    
                    Δες πώς συμπεριφέρονται DBSCAN και K-Means στο ίδιο dataset:
                    """)

        fig_vs, axes_vs = plt.subplots(1, 2, figsize = (14, 5))
        
        #   DBSCAN
        unique_vs = set(labels)
        colors_vs = plt.cm.Set1(np.linspace(0, 1, len(unique_vs)))
        
        for k, col in zip(unique_vs, colors_vs):
            if k == -1:
                col, marker = 'black', 'x'
            else:
                marker = 'o'
            
            mask = labels == k
            axes_vs[0].scatter(X[mask, 0], X[mask, 1], c = [col], s = 40, marker = marker, alpha = 0.7)
        
        axes_vs[0].set_title(f'DBSCAN (ε = {eps}, minPts = {min_samples})\n{n_clusters} clusters')
        axes_vs[0].set_xlabel('Feature 1')
        axes_vs[0].set_ylabel('Feature 2')
        
        #   K-Means
        kmeans_vs = KMeans(n_clusters = max(n_clusters, 2), n_init = "auto", random_state = RANDOM_SEED)
        labels_km = kmeans_vs.fit_predict(X)
        
        for k in range(max(n_clusters, 2)):
            mask = labels_km == k
            axes_vs[1].scatter(X[mask, 0], X[mask, 1], c = [plt.cm.Set1(k/max(n_clusters, 2))], s = 40, alpha = 0.7)
        
        axes_vs[1].scatter(kmeans_vs.cluster_centers_[:, 0], kmeans_vs.cluster_centers_[:, 1], c = 'black', s = 150, marker = 'X', edgecolors = 'white', linewidths = 2)
        axes_vs[1].set_title(f'K-Means (K = {max(n_clusters, 2)})')
        axes_vs[1].set_xlabel('Feature 1')
        axes_vs[1].set_ylabel('Feature 2')
        
        plt.tight_layout()
        st.pyplot(fig_vs)
        plt.close(fig_vs)
        
        st.success("""
                **DBSCAN** είναι καλύτερος όταν:
                * Δεν ξέρουμε τον αριθμό clusters
                * Τα clusters έχουν ακανόνιστα σχήματα
                * Υπάρχουν outliers που θέλουμε να εντοπίσουμε
                """)
        
        st.warning("""
                **K-Means** είναι καλύτερος όταν:
                * Ξέρουμε τον αριθμό clusters
                * Τα clusters είναι σφαιρικά
                * Θέλουμε γρήγορη εκτέλεση σε μεγάλα datasets
                """)
    
#   QUIZ
    render_quiz("dbscan")
    
#   CONCLUSION
    render_conclusion("dbscan")