import streamlit as st, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import apply_global_styles
from quizzes import render_quiz
from conclusions import render_conclusion

def show_kmeans():
    apply_global_styles()
    RANDOM_SEED = 11
    
#   TITLE & THEORY
    st.title("🔵 K-Means Clustering", anchor = False)
    
    st.markdown("""
                Ο αλγόριθμος **K-Means** είναι ένας από τους πιο δημοφιλείς αλγόριθμους clustering.
                
                ### 🎯 Στόχος: Ομαδοποίηση παρόμοιων σημείων
                
                Ο αλγόριθμος χωρίζει τα δεδομένα σε **K ομάδες (clusters)** έτσι ώστε:
                * Τα σημεία μέσα σε κάθε ομάδα να είναι όσο το δυνατόν πιο κοντά
                * Τα κέντρα των ομάδων (centroids) να αντιπροσωπεύουν καλά τα δεδομένα
                """)
    
    st.latex(r"\text{Minimize: } J = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2")
    
    st.markdown("""
                Όπου:
                * $K$ = Αριθμός clusters
                * $C_i$ = Το i-οστό cluster
                * $\mu_i$ = Το κέντρο (centroid) του cluster $C_i$
                """)
    st.divider()
    
#   THEORY: K-MEANS ALGORITHM
    with st.expander("📚 Πώς λειτουργεί ο K-Means;", expanded = False):
        st.markdown("""
                    ### 🔄 Ο Αλγόριθμος σε 2 Βήματα
                    
                    Ο K-Means επαναλαμβάνει δύο απλά βήματα μέχρι να συγκλίνει:
                    """)
        
        col_step1, col_step2 = st.columns(2)
        
        with col_step1:
            st.info("""
                    **Βήμα 1: Βήμα ανάθεσης**
                    
                    Ανάθεση κάθε σημείου στο κοντινότερο centroid:
                    
                    $c^{(i)} = \\arg\\min_j ||x^{(i)} - \\mu_j||^2$
                    
                    *Κάθε σημείο "διαλέγει" την ομάδα της!*
                    """)
        
        with col_step2:
            st.success("""
                    **Βήμα 2: Βήμα ενημέρωσης**
                    
                    Υπολογισμός νέων centroids:
                    
                    $\\mu_j = \\frac{1}{|C_j|} \\sum_{i \\in C_j} x^{(i)}$
                    
                    *Το κέντρο μετακινείται στο μέσο των σημείων!*
                    """)
        
        st.markdown("""
                    ---
                    
                    ### ⚠️ Σημαντικές Παρατηρήσεις:
                    
                    * **Initialization:** Τα αρχικά centroids επηρεάζουν το τελικό αποτέλεσμα
                    * **Convergence:** Ο αλγόριθμος σταματά όταν τα centroids δεν αλλάζουν
                    * **Local Minima:** Μπορεί να "κολλήσει" σε τοπικά ελάχιστα
                    * **K Selection:** Πρέπει να ορίσουμε εκ των προτέρων το K
                    """)
    st.divider()

#   SIDEBAR: DATASET & PARAMETERS
    st.sidebar.header("1. Δημιουργία Δεδομένων")
    
    #   Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Τύπος Δεδομένων:",
        ["Blobs (Well-separated)", "Anisotropic (Elongated)", "Noisy Blobs"],
        key = "kmeans_dataset"
    )
    
    n_samples = st.sidebar.slider("Αριθμός Σημείων", 100, 500, 300, key = "kmeans_samples")
    true_k = st.sidebar.slider("Πραγματικά Clusters", 2, 6, 3, key = "kmeans_true_k")
    
    #   Generate data based on selection
    if dataset_type == "Blobs (Well-separated)":
        X, _ = make_blobs(n_samples = n_samples, centers = true_k, cluster_std = 1.0, random_state = RANDOM_SEED)
    elif dataset_type == "Anisotropic (Elongated)":
        X, _ = make_blobs(n_samples = n_samples, centers = true_k, cluster_std = 1.0, random_state = RANDOM_SEED)
        #   Anisotropic transformation
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    else:  # Noisy Blobs
        X, _ = make_blobs(n_samples = n_samples, centers = true_k, cluster_std = 1.5, random_state = RANDOM_SEED)
        #   Add some noise
        np.random.seed(RANDOM_SEED)
        noise = np.random.randn(20, 2) * 3
        X = np.vstack([X, noise])
    st.sidebar.divider()
    
#   K-MEANS PARAMETERS
    st.sidebar.header("2. Παράμετροι K-Means")
    
    k_value = st.sidebar.slider("K (Clusters)", 1, 8, 3, key = "kmeans_k")
    
    init_method = st.sidebar.selectbox("Initialization", ["random", "k-means++"], key = "kmeans_init")
    
    n_init = st.sidebar.slider("n_init (Επαναλήψεις)", 1, 20, 10, key = "kmeans_ninit")
    
#   STEP-BY-STEP VISUALIZATION
    st.header("🎬 Step-by-Step Visualization")
    
    st.markdown("""
                Παρακολούθησε τον αλγόριθμο K-Means βήμα-βήμα!
                """)
    
    # Initialize session state
    if 'kmeans_step' not in st.session_state:
        st.session_state.kmeans_step = 0
        st.session_state.kmeans_centroids = None
        st.session_state.kmeans_labels = None
        st.session_state.kmeans_max_iter_stored = 10
    
    #   Layout: Plot on left, controls on right
    col_plot, col_controls = st.columns([3, 1])
    
    with col_controls:
        st.markdown("#### ⚙️ Έλεγχος Βημάτων")
        
        #   Maximum iterations control
        max_iterations = st.number_input("Μέγιστες Επαναλήψεις", 1, 20, 10, key = "kmeans_max_iter")
        
        #   Reset if max_iterations changed
        if max_iterations != st.session_state.kmeans_max_iter_stored:
            st.session_state.kmeans_step = 0
            st.session_state.kmeans_centroids = None
            st.session_state.kmeans_max_iter_stored = max_iterations
            
        st.divider()
        
        #   Navigation buttons
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.kmeans_step = 0
            st.session_state.kmeans_centroids = None
            
        btn_prev, btn_next = st.columns(2)
        with btn_prev:
            if st.button("⬅️ Prev", use_container_width=True):
                if st.session_state.kmeans_step > 0:
                    st.session_state.kmeans_step -= 1
        with btn_next:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.kmeans_step < st.session_state.kmeans_max_iter_stored * 2:
                    st.session_state.kmeans_step += 1
        
        st.info(f"📊 Βήμα: {st.session_state.kmeans_step} / {max_iterations * 2}")
    
    
    #   Run K-Means step by step
    current_step = st.session_state.kmeans_step
    
    #   Initialize centroids randomly if first step
    if st.session_state.kmeans_centroids is None or current_step == 0:
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X), k_value, replace=False)
        centroids = X[indices].copy()
        st.session_state.kmeans_centroids = centroids
        st.session_state.kmeans_labels = np.zeros(len(X), dtype = int)
    
    centroids = st.session_state.kmeans_centroids.copy()
    labels = st.session_state.kmeans_labels.copy()
    
    #   Perform iterations up to current step
    iteration = current_step // 2
    is_update_step = (current_step % 2 == 1)
    
    #   Recalculate from scratch for consistent display
    if current_step > 0:
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X), k_value, replace = False)
        centroids = X[indices].copy()
        
        for i in range(iteration):
            #   Assignment step
            distances = np.zeros((len(X), k_value))
            for j in range(k_value):
                distances[:, j] = np.sqrt(np.sum((X - centroids[j])**2, axis = 1))
            labels = np.argmin(distances, axis=1)
            
            #   Update step
            new_centroids = np.zeros_like(centroids)
            for j in range(k_value):
                if np.sum(labels == j) > 0:
                    new_centroids[j] = X[labels == j].mean(axis = 0)
                else:
                    new_centroids[j] = centroids[j]
            centroids = new_centroids
        
        #   If we're in the middle of an iteration (assignment step only)
        if is_update_step:
            distances = np.zeros((len(X), k_value))
            for j in range(k_value):
                distances[:, j] = np.sqrt(np.sum((X - centroids[j])**2, axis = 1))
            labels = np.argmin(distances, axis = 1)
    
    #   Calculate current distances for assignment
    distances = np.zeros((len(X), k_value))
    for j in range(k_value):
        distances[:, j] = np.sqrt(np.sum((X - centroids[j])**2, axis = 1))
    labels = np.argmin(distances, axis = 1)
    
    #   Determine step type
    if current_step == 0:
        step_type = "Αρχικοποίηση Centroids"
    elif current_step % 2 == 1:
        step_type = "Βήμα Ανάθεσης"
    else:
        step_type = "Βήμα Ενημέρωσης"
        
    #   Plot current state
    fig_anim, ax_anim = plt.subplots(figsize = (10, 8))
    
    #   Colors for clusters
    colors = plt.cm.Set1(np.linspace(0, 1, max(k_value, 3)))
    
    #   Plot data points colored by cluster
    for k in range(k_value):
        mask = labels == k
        ax_anim.scatter(X[mask, 0], X[mask, 1], c=[colors[k]], s = 50, alpha = 0.6, label = f'Cluster {k + 1}')
    
    #   Plot centroids
    ax_anim.scatter(centroids[:, 0], centroids[:, 1], c = 'black', s = 300, marker = 'X', edgecolors = 'white', linewidths = 2, label = 'Centroids', zorder = 5)
    
    #   Draw lines from points to centroids (if not first step)
    if current_step > 0:
        for k in range(k_value):
            mask = labels == k
            #   Show only 10 lines per cluster for clarity
            for i in np.where(mask)[0][:10]:
                ax_anim.plot([X[i, 0], centroids[k, 0]], [X[i, 1], centroids[k, 1]], c = colors[k], alpha = 0.2, linewidth = 0.5)
    
    ax_anim.set_xlabel('Feature 1')
    ax_anim.set_ylabel('Feature 2')
    ax_anim.set_title(f'K-Means: {step_type} (Επανάληψη {max(0, (current_step-1)//2 + 1)})')
    ax_anim.legend(loc = 'upper right')
    ax_anim.grid(True, alpha = 0.3)
    
    #   Display plot
    with col_plot:
        st.pyplot(fig_anim)
        
    plt.close(fig_anim)
    
#   FINAL CLUSTERING RESULT
    st.divider()
    st.header("📊 Τελικό Αποτέλεσμα Clustering")
    
    #   Run sklearn KMeans
    kmeans_final = KMeans(n_clusters = k_value, init = init_method, n_init = n_init, random_state = RANDOM_SEED)
    kmeans_final.fit(X)
    labels_final = kmeans_final.labels_
    centroids_final = kmeans_final.cluster_centers_
    
    #   Calculate silhouette score if more than 1 cluster
    if k_value > 1:
        silhouette = silhouette_score(X, labels_final)
    else:
        silhouette = 0
    
    #   Plot final result
    fig_final, ax_final = plt.subplots(figsize = (10, 8))
    
    for k in range(k_value):
        mask = labels_final == k
        ax_final.scatter(X[mask, 0], X[mask, 1], c=[colors[k]], s = 50, alpha = 0.6, label = f'Cluster {k + 1}')
    
    ax_final.scatter(centroids_final[:, 0], centroids_final[:, 1], c = 'black', s = 300, marker = 'X', edgecolors = 'white', linewidths = 2, label = 'Final Centroids', zorder = 5)
    
    ax_final.set_xlabel('Feature 1')
    ax_final.set_ylabel('Feature 2')
    ax_final.set_title(f'K-Means Final Result (K = {k_value})')
    ax_final.legend(loc = 'upper right')
    ax_final.grid(True)
    
    st.pyplot(fig_final)
    plt.close(fig_final)
    
    #   Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric("Inertia", f"{kmeans_final.inertia_:.2f}", help = "Άθροισμα τετραγώνων αποστάσεων από κεντροειδή")
    
    with col_m2:
        st.metric("Silhouette Score", f"{silhouette:.3f}", help = "-1 έως 1, όσο υψηλότερο τόσο καλύτερο")
    
    with col_m3:
        st.metric("Iterations", kmeans_final.n_iter_, help = "Επαναλήψεις μέχρι τη σύγκλιση")
    
#   ELBOW METHOD
    st.divider()
    st.header("📈 Elbow Method: Βρες το βέλτιστο K")
    
    st.markdown("""
                Το **Elbow Method** μας βοηθά να βρούμε τον βέλτιστο αριθμό clusters:
                
                * Αυξάνουμε το K και μετράμε την Inertia (πόσο "σφιχτά" είναι τα clusters)
                * Ψάχνουμε το "αγκώνα" (elbow) στο γράφημα
                """)
    
    k_range, inertias, silhouettes = range(1, 11), [], []
    
    for k in k_range:
        km = KMeans(n_clusters = k, init = init_method, n_init = "auto", random_state = RANDOM_SEED)
        km.fit(X)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X, km.labels_))
        else:
            silhouettes.append(0)
    
    fig_elbow, ax_elbow = plt.subplots(figsize = (10, 6))
    
    #   Inertia plot (left Y-axis)
    ax_elbow.set_xlabel('Number of Clusters (K)')
    ax_elbow.set_ylabel('Inertia', color = 'blue')
    line1 = ax_elbow.plot(k_range, inertias, 'bo-', linewidth = 2, markersize = 8, label = 'Inertia')
    ax_elbow.tick_params(axis = 'y', labelcolor = 'blue')
    ax_elbow.set_xticks(k_range)
    ax_elbow.grid(True, alpha = 0.3)
    
    #   Silhouette plot (right Y-axis)
    ax_silhouette = ax_elbow.twinx()
    ax_silhouette.set_ylabel('Silhouette Score', color = 'green')
    line2 = ax_silhouette.plot(k_range, silhouettes, 'go-', linewidth = 2, markersize = 8, label = 'Silhouette Score')
    ax_silhouette.tick_params(axis = 'y', labelcolor = 'green')
    
    #   Selected K vertical line
    ax_elbow.axvline(x = k_value, color = 'red', linestyle = '--', linewidth = 2, label = f'Το K που επέλεξες = {k_value}')
    
    #   Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_elbow.legend(lines + [ax_elbow.lines[-1]], labels + [f'Το K που επέλεξες = {k_value}'], loc = 'upper right')
    
    ax_elbow.set_title('Elbow Method & Silhouette Score')
    plt.tight_layout()
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)
    
    #   Find best K by silhouette
    st.success(f"🏆 Βέλτιστο K βάσει Silhouette Score: **{list(k_range)[np.argmax(silhouettes)]}**")
    
#   K COMPARISON
    st.header("🔬 Σύγκριση Διαφορετικών K")
    
    k_values_compare = [2, 3, 4, 5]
    
    fig_compare, axes_compare = plt.subplots(1, 4, figsize = (16, 4))
    
    for idx, k_comp in enumerate(k_values_compare):
        km_comp = KMeans(n_clusters = k_comp, init = init_method, n_init = n_init, random_state = RANDOM_SEED)
        labels_comp = km_comp.fit_predict(X)
        
        for k in range(k_comp):
            mask = labels_comp == k
            axes_compare[idx].scatter(X[mask, 0], X[mask, 1], c = [plt.cm.Set1(k/k_comp)], s = 30, alpha = 0.6)
        
        axes_compare[idx].scatter(km_comp.cluster_centers_[:, 0], km_comp.cluster_centers_[:, 1], c = 'black', s = 150, marker = 'X', edgecolors = 'white', linewidths = 1.5)
        axes_compare[idx].set_title(f'K = {k_comp}\nInertia: {km_comp.inertia_:.0f}')
        axes_compare[idx].set_xlabel('Feature 1')
        axes_compare[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    st.pyplot(fig_compare)
    plt.close(fig_compare)
    
#   INITIALIZATION COMPARISON
    with st.expander("🔬 Random vs K-Means++ Initialization", expanded = False):
        st.markdown("""
                    ### Η σημασία της αρχικοποίησης
                    
                    Η αρχική θέση των centroids επηρεάζει τεράστια το τελικό αποτέλεσμα. Δες πώς 
                    συμπεριφέρονται οι δύο μέθοδοι με ακριβώς τον ίδιο αριθμό επαναλήψεων (n_init = 1):
                    """)
        
        fig_init, axes_init = plt.subplots(1, 2, figsize = (14, 5))
        
        #   Random Initialization
        km_random = KMeans(n_clusters = k_value, init = 'random', n_init = 1, random_state = RANDOM_SEED)
        labels_r = km_random.fit_predict(X)
        
        for k in range(k_value):
            mask = labels_r == k
            axes_init[0].scatter(X[mask, 0], X[mask, 1], c = [plt.cm.Set1(k/max(k_value, 1))], s = 40, alpha = 0.6)
        
        axes_init[0].scatter(km_random.cluster_centers_[:, 0], km_random.cluster_centers_[:, 1], c = 'black', s = 200, marker = 'X', edgecolors = 'white', linewidths = 2)
        axes_init[0].set_title(f'Random Initialization\nInertia: {km_random.inertia_:.0f}')
        axes_init[0].set_xlabel('Feature 1')
        axes_init[0].set_ylabel('Feature 2')
        axes_init[0].grid(True, alpha = 0.3)
        
        # 2. K-Means++ Initialization
        km_plus = KMeans(n_clusters = k_value, init = 'k-means++', n_init = 1, random_state = RANDOM_SEED)
        labels_p = km_plus.fit_predict(X)
        
        for k in range(k_value):
            mask = labels_p == k
            axes_init[1].scatter(X[mask, 0], X[mask, 1], c = [plt.cm.Set1(k/max(k_value, 1))], s = 40, alpha = 0.6)
        
        axes_init[1].scatter(km_plus.cluster_centers_[:, 0], km_plus.cluster_centers_[:, 1], c = 'black', s = 200, marker = 'X', edgecolors = 'white', linewidths = 2)
        axes_init[1].set_title(f'K-Means++ Initialization\nInertia: {km_plus.inertia_:.0f}')
        axes_init[1].set_xlabel('Feature 1')
        axes_init[1].set_ylabel('Feature 2')
        axes_init[1].grid(True, alpha = 0.3)
        
        plt.tight_layout()
        st.pyplot(fig_init)
        plt.close(fig_init)
        
        st.info("""
                **Γιατί συμβαίνει αυτό;**
                * Στη **Random**, τα αρχικά κέντρα μπορεί να πέσουν πολύ κοντά το ένα στο άλλο, κάνοντας τον αλγόριθμο να "κολλήσει" σε τοπικό ελάχιστο (υψηλότερη Inertia).
                * Ο **K-Means++** τοποθετεί έξυπνα το πρώτο κέντρο, και μετά διαλέγει τα επόμενα έτσι ώστε να έχουν τη μέγιστη δυνατή απόσταση μεταξύ τους, εξασφαλίζοντας σχεδόν πάντα τη βέλτιστη λύση!
                """)
    
#   QUIZ
    render_quiz("kmeans")

#   CONCLUSION
    render_conclusion("kmeans")