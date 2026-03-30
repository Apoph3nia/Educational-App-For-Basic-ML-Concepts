import streamlit as st, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from utils import apply_global_styles
from quizzes import render_quiz
from conclusions import render_conclusion

def show_knn():
    apply_global_styles()
    RANDOM_SEED = 11
    
#   TITLE & HEADER
    st.title("🔍 K-Nearest Neighbors (KNN)", anchor = False)
    
    st.markdown("""
                Ο αλγόριθμος **K-Nearest Neighbors (KNN)** είναι ένας απλός αλλά ισχυρός αλγόριθμος ταξινόμησης.
                
                ### 🎯 Κεντρική Ιδέα: "Δημοκρατία στη Γειτονιά"
                
                Για να ταξινομήσουμε ένα νέο σημείο, βρίσκουμε τους **K πιο κοντινούς γείτονες** 
                και τους ζητάμε να "ψηφίσουν" για την κατηγορία του!
                """)
    
    st.latex(r"\text{Πρόβλεψη} = \text{Majority Vote}(k \text{ nearest neighbors})")
    
    st.markdown("""
                **Βασικές Έννοιες:**
                * **K:** Ο αριθμός των γειτόνων που εξετάζουμε
                * **Distance Metric:** Συνήθως η Ευκλείδεια απόσταση
                * **Majority Voting:** Η κατηγορία με τις περισσότερες ψήφους κερδίζει
                """)
    
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
    st.divider()
    
#   THEORY: Hyperparameter K
    with st.expander("📚 Η Σημασία του K: Overfitting vs Underfitting", expanded = False):
        st.markdown("""
                    ### ❓ Πώς το K επηρεάζει το μοντέλο;
                    
                    Η επιλογή του **K** είναι κρίσιμη για την απόδοση του KNN:
                    """)
        
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.error("""
                    **K = 1 (Overfitting)**
                    * Πολύ ευαίσθητο στο θόρυβο
                    * "Νησάκια" στο decision boundary
                    * Αναγνωρίζει κάθε σημείο
                    * Χαμηλή γενίκευση
                    """)
        
        with col_k2:
            st.success("""
                    **K = Μεσαία τιμή (Good Fit)**
                    * Ισορροπία μεταξύ ευαισθησίας και σταθερότητας
                    * Λεία καμπύλη διαχωρισμού
                    * Καλή γενίκευση
                    """)
        
        with col_k3:
            st.warning("""
                    **K = Μεγάλο (Underfitting)**
                    * Πολύ "απλό" μοντέλο
                    * Αγνοεί τοπικά μοτίβα
                    * Σχεδόν ευθεία γραμμή
                    * Χαμηλή ακρίβεια
                    """)
        
        st.info("""
        💡 **TIP:** Συχνά χρησιμοποιούμε $K = \sqrt{n}$ όπου $n$ είναι ο συνολικός αριθμός των δεδομένων, και προτιμάμε περιττούς αριθμούς για να αποφύγουμε ισοπαλίες σε binary classification.
            """)
        
        st.markdown("""
                    ### ⚖️ Βάρη Γειτόνων (Weighting)
                    * **Uniform:** Όλοι οι $K$ γείτονες έχουν ακριβώς την ίδια "δύναμη" ψήφου, ανεξάρτητα από το πόσο κοντά ή μακριά βρίσκονται από το νέο σημείο.
                    * **Distance:** Οι γείτονες που βρίσκονται πιο κοντά στο σημείο έχουν μεγαλύτερο "βάρος" στην τελική απόφαση από τους πιο απομακρυσμένους.
                    """)
    st.divider()
    
#   SIDEBAR: Dataset & KNN Parameters
    st.sidebar.header("1. Δημιουργία Δεδομένων")
    
    #   Dataset selection
    dataset_type = st.sidebar.selectbox(
                                        "Τύπος Δεδομένων:",
                                        ["Linear Separable", "Moons (Non-linear)", "Circles"],
                                        key = "knn_dataset"
                                        )
    
    n_samples = st.sidebar.slider("Αριθμός Σημείων", 50, 300, 150, key = "knn_samples")
    noise_level = st.sidebar.slider("Θόρυβος", 0.0, 0.5, 0.2, 0.05, key = "knn_noise")
    
    #   Generate data based on selection
    if dataset_type == "Linear Separable":
        X, y = make_classification(
            n_samples = n_samples, n_features = 2, n_redundant = 0, 
            n_informative = 2, n_clusters_per_class = 1,
            class_sep = 1.5, flip_y = noise_level, random_state = RANDOM_SEED
        )
    elif dataset_type == "Moons (Non-linear)":
        X, y = make_moons(n_samples = n_samples, noise = noise_level, random_state = RANDOM_SEED)
    else:
        X, y = make_circles(n_samples = n_samples, noise = noise_level, factor = 0.5, random_state = RANDOM_SEED)
    
    #   Train/test split
    test_size = st.sidebar.slider("Test Set %", 10, 50, 20, 5) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = RANDOM_SEED)
    st.sidebar.divider()
    
    # KKN Parameters
    st.sidebar.header("2. Παράμετροι KNN")
    
    k_value = st.sidebar.slider("K (Γείτονες)", 1, 25, 5, 2, key = "knn_k")
    
    #   Check if K is odd for binary classification
    if k_value % 2 == 0:
        st.sidebar.info("💡 Καλό είναι το K να είναι περιττός αριθμός!")
    
    weights = st.sidebar.selectbox("Weighting", ["uniform", "distance"], key = "knn_weights")
    
#   MAIN VISUALIZATION
    st.header("📊 Decision Boundary Visualization")
    
    #   Train KNN
    knn = KNeighborsClassifier(n_neighbors = k_value, weights = weights)
    knn.fit(X_train, y_train)
    
    #   Create meshgrid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    #   Predict on meshgrid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    #   Calculate accuracy
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    #   Plot
    fig, ax = plt.subplots(figsize = (10, 8))
    
    #   Decision boundary
    ax.contourf(xx, yy, Z, alpha = 0.3, cmap = 'coolwarm')
    
    #   Decision boundary line
    ax.contour(xx, yy, Z, levels = [0.5], colors = 'black')
    
    #   Training data
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c = 'blue', alpha = 0.7, label = 'Class 0 (Train)', edgecolors = 'black')
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c = 'red', alpha = 0.7, label = 'Class 1 (Train)', edgecolors = 'black')
    
    #   Test data
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c = 'blue', alpha = 0.9, label = 'Class 0 (Test)', marker = 's', edgecolors = 'black', linewidths = 2)
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c = 'red', alpha = 0.9, label = 'Class 1 (Test)', marker = 's', edgecolors = 'black', linewidths = 2)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'KNN Decision Boundary (K = {k_value}, {weights} weights)')
    ax.legend(loc = 'upper right')
    ax.grid(True, alpha = 0.3)
    
    st.pyplot(fig)
    plt.close(fig)
    
#   METRICS
    st.divider()
    st.header("📈 Αποτελέσματα")
    
    met_col1, met_col2, met_col3 = st.columns(3)
    
    with met_col1:
        st.metric("Training Accuracy", f"{train_acc:.2%}")
    
    with met_col2:
        st.metric("Test Accuracy", f"{test_acc:.2%}")
    
    with met_col3:
        gap = train_acc - test_acc
        if gap > 0.1:
            st.metric("Overfitting Gap", f"{gap:.2%}", delta_color = "inverse")
        else:
            st.metric("Generalization Gap", f"{gap:.2%}")
    
    #   Confusion Matrix
    st.subheader("🎯 Confusion Matrix (Test Set)")
    
    fig_cm, ax_cm = plt.subplots(figsize = (6, 5))
    
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred_test, 
        display_labels = ['Class 0', 'Class 1'], 
        cmap = 'Blues', 
        ax = ax_cm
    )
    
    ax_cm.set_title('Confusion Matrix')
    
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)
    
    st.divider()
    
#   K VALUE COMPARISON
    st.header("🔬 Σύγκριση Διαφορετικών K")
    
    st.markdown("""Δες πώς αλλάζει το decision boundary για διαφορετικές τιμές του K:""")
    
    k_values_to_compare = [1, 3, 5, 15]
    
    fig_compare, axes_compare = plt.subplots(1, 4, figsize = (16, 4))
    
    for idx, k_comp in enumerate(k_values_to_compare):
        #   Train KNN with different K
        knn_comp = KNeighborsClassifier(n_neighbors = k_comp, weights = weights)
        knn_comp.fit(X_train, y_train)
        
        #   Predict on meshgrid
        Z_comp = knn_comp.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_comp = Z_comp.reshape(xx.shape)
        
        #   Accuracy
        acc_comp = accuracy_score(y_test, knn_comp.predict(X_test))
        
        #   Plot
        axes_compare[idx].contourf(xx, yy, Z_comp, alpha = 0.3, cmap = 'coolwarm')
        axes_compare[idx].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c = 'blue', s = 20, alpha = 0.9, marker='s', edgecolors='black')
        axes_compare[idx].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c = 'red', s = 20, alpha = 0.9, marker='s', edgecolors='black')
        axes_compare[idx].set_title(f'K = {k_comp}\nAcc: {acc_comp:.2%}')
        axes_compare[idx].set_xlabel('Feature 1')
        axes_compare[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    st.pyplot(fig_compare)
    plt.close(fig_compare)
    
#   K vs ACCURACY CURVE
    st.header("📉 Accuracy vs K Curve")
    
    k_range = range(1, min(25, len(X_train)))
    train_scores = []
    test_scores = []
    
    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors = k, weights = weights)
        knn_temp.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, knn_temp.predict(X_train)))
        test_scores.append(accuracy_score(y_test, knn_temp.predict(X_test)))
    
    fig_curve, ax_curve = plt.subplots(figsize = (10, 5))
    ax_curve.plot(k_range, train_scores, 'b-', label = 'Training Accuracy', linewidth = 2)
    ax_curve.plot(k_range, test_scores, 'r-', label = 'Test Accuracy', linewidth = 2)
    ax_curve.axvline(x = k_value, color = 'green', linestyle = '--', label = f'Selected K = {k_value}')
    ax_curve.set_xlabel('K Value')
    ax_curve.set_ylabel('Accuracy')
    ax_curve.set_title('Effect of K on Model Performance')
    ax_curve.legend()
    ax_curve.grid(True, alpha = 0.3)
    ax_curve.set_xticks(list(k_range)[::2])
    
    st.pyplot(fig_curve)
    plt.close(fig_curve)
    
    #   Find optimal K
    optimal_k = list(k_range)[np.argmax(test_scores)]
    st.success(f"🏆 Βέλτιστο K (βάσει Test Accuracy): **{optimal_k}** με accuracy {max(test_scores):.2%}")
    
#   INTERACTIVE NEAREST NEIGHBORS
    st.divider()
    st.header("🎯 Εξερεύνησε τους Γείτονες")
    
    st.markdown("""Διάλεξε ένα σημείο για να δεις ποιοι είναι οι **K κοντινότεροι γείτονές** του!""")
    
    #   Layout for plot and controls
    col_plot, col_controls = st.columns([3, 1])
    
    with col_controls:
        st.markdown("#### ⚙️ Συντεταγμένες")
        point_x = st.slider("X θέση", float(x_min), float(x_max), 0.0, 0.1, key = "point_x")
        point_y = st.slider("Y θέση", float(y_min), float(y_max), 0.0, 0.1, key = "point_y")
        st.divider()
    
    #   Find nearest neighbors
    query_point = np.array([[point_x, point_y]])
    distances, indices = knn.kneighbors(query_point)
    
    #   Predict class
    predicted_class = knn.predict(query_point)[0]
    proba = knn.predict_proba(query_point)[0]
    
    #   Display prediction and probabilities
    with col_controls:
        st.markdown("#### 📊 Αποτέλεσμα")
        st.metric("Πρόβλεψη", f"Class {predicted_class}")
        st.metric("Πιθανότητα Class 0", f"{proba[0]:.2%}")
        st.metric("Πιθανότητα Class 1", f"{proba[1]:.2%}")
    
    #   Plot with neighbors
    fig_nn, ax_nn = plt.subplots(figsize = (10, 8))
    
    #   Decision boundary
    ax_nn.contourf(xx, yy, Z, alpha = 0.2, cmap = 'coolwarm')
    
    #   Training data
    ax_nn.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c = 'blue', alpha = 0.5, label = 'Class 0')
    ax_nn.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c = 'red', alpha = 0.5, label = 'Class 1')
    
    #   Highlight nearest neighbors
    for i, idx in enumerate(indices[0]):
        ax_nn.scatter(X_train[idx, 0], X_train[idx, 1], s = 30, c = 'yellow', edgecolors = 'black', linewidths = 2, marker = 'o')
        ax_nn.plot([point_x, X_train[idx, 0]], [point_y, X_train[idx, 1]], 'g--', linewidth = 1.5, alpha = 0.7)
    
    #   Query point
    ax_nn.scatter(point_x, point_y, c = 'red' if predicted_class == 1 else 'blue', s = 200, marker = '*', edgecolors = 'black', linewidths = 2, label = f'Query Point (Class {predicted_class})', zorder = 5)
    
    ax_nn.set_xlabel('Feature 1')
    ax_nn.set_ylabel('Feature 2')
    ax_nn.set_title(f'K = {k_value} Nearest Neighbors')
    ax_nn.legend(loc = 'upper right')
    ax_nn.grid(True, alpha = 0.3)
    
    #   Display plot
    with col_plot:
        st.pyplot(fig_nn)
        plt.close(fig_nn)
    
    #   Show neighbor info table
    st.markdown("### 📋 Αναλυτικές Πληροφορίες Γειτόνων")
    
    neighbor_data = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        neighbor_data.append({
            "Γείτονας #": i + 1,
            "X": f"{X_train[idx, 0]:.3f}",
            "Y": f"{X_train[idx, 1]:.3f}",
            "Κατηγορία": f"Class {y_train[idx]}",
            "Απόσταση": f"{dist:.3f}"
        })
    
    st.table(neighbor_data)
    
#   QUIZ
    render_quiz("knn")
    
#   CONCLUSION
    render_conclusion("knn")