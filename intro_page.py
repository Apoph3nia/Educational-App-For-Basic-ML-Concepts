import streamlit as st, matplotlib.pyplot as plt, numpy as np, pandas as pd
from quizzes import render_quiz
from utils import apply_global_styles

def show_intro_page():
    apply_global_styles()
    RANDOM_SEED = 11

#   TITLE AND HEADER
    st.title("🎓 Εισαγωγή στο Machine Learning", anchor = False)
    st.markdown("### Μια διαδραστική εκπαιδευτική εφαρμογή για την κατανόηση αλγορίθμων")
    st.divider()


#   WHAT IS ML?
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Τι είναι η Μηχανική Μάθηση;", anchor = False)
        st.write("""
                Η **Μηχανική Μάθηση (Machine Learning)** είναι ένα υποπεδίο της Τεχνητής Νοημοσύνης (AI) 
                που εστιάζει στη δημιουργία συστημάτων τα οποία **μαθαίνουν από τα δεδομένα**, αντί να 
                προγραμματίζονται ρητά για κάθε πιθανή κατάσταση.
                
                Ο στόχος είναι να επιτρέψουμε στους υπολογιστές να:
                * 🔍 **Ανακαλύπτουν** κρυμμένα μοτίβα (patterns) στα δεδομένα
                * 🎯 **Προβλέπουν** μελλοντικά αποτελέσματα
                * 🧠 **Λαμβάνουν αποφάσεις** χωρίς ρητό προγραμματισμό
                 """)
    
    with col2:
        st.image("./src/ai-img.jpg", caption = "AI", use_column_width = True)
    st.divider()


#   TRADITIONAL vs ML PROGRAMMING
    st.header("💻 Παραδοσιακός Προγραμματισμός vs Machine Learning", anchor = False)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 📝 Παραδοσιακός Προγραμματισμός")
        st.info("""
                **Είσοδος:** Δεδομένα + Κανόνες (Κώδικας)  
                **Έξοδος:** Αποτελέσματα
                
                Ο προγραμματιστής γράφει ρητούς κανόνες.
                """)
        st.code("""
                # Παράδειγμα: Ταξινόμηση email
                if "αγορά" in email:
                    return "διαφήμιση"
                elif "φίλος" in email:
                    return "προσωπικό"
                else:
                    return "άγνωστο"
                """, language = "python")

    with col_b:
        st.markdown("#### 🧠 Machine Learning")
        st.success("""
                **Είσοδος:** Δεδομένα + Αποτελέσματα (Ετικέτες)  
                **Έξοδος:** Κανόνες (Μοντέλο)
                
                Ο αλγόριθμος ανακαλύπτει τους κανόνες μόνος του!
                """)
        st.code("""
                # Παράδειγμα: ML ταξινόμηση
                model = Algorithm()
                model.train(emails, labels)
                prediction = model.predict(new_email)
                # Το μοντέλο έμαθε τους κανόνες!
                """, language = "python")

    st.warning("💡 **Βασική Διαφορά:**\n Στο ML, δίνουμε παραδείγματα και η μηχανή βρίσκει τους κανόνες. Αυτό επιτρέπει την επίλυση προβλημάτων πολύπλοκων για να περιγραφούν με χειροκίνητους κανόνες.")
    st.divider()


#   THE 3 MAIN CATEGORIES
    st.header("📌 Οι 3 Βασικές Κατηγορίες της Μηχανικής Μάθησης", anchor = False)
    st.write("Αυτή η εφαρμογή καλύπτει τις τρεις κύριες κατηγορίες αλγορίθμων:")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("### 🎯 Supervised Learning")
        st.markdown("**Επιβλεπόμενη Μάθηση**")
        st.write("""
                Το μοντέλο εκπαιδεύεται με **δεδομένα που έχουν ετικέτες** (labels). 
                Ξέρουμε ήδη τη σωστή απάντηση και διδάσκουμε τη μηχανή.
                
                **Ανάγκη:** Δεδομένα με σωστές απαντήσεις
                
                *Παραδείγματα στην εφαρμογή:*
                * **Linear Regression:** Πρόβλεψη αριθμών (π.χ. τιμή σπιτιού)
                * **KNN:** Ταξινόμηση σε κατηγορίες (π.χ. spam/όχι spam)
                """)
        st.markdown("**Τύποι:** Regression, Classification")

    with c2:
        st.warning("### 🔍 Unsupervised Learning")
        st.markdown("**Μη Επιβλεπόμενη Μάθηση**")
        st.write("""
                Τα δεδομένα **δεν έχουν ετικέτες**. Το μοντέλο προσπαθεί 
                να βρει δομή και ομοιότητες μόνο του.
                
                **Ανάγκη:** Μόνο δεδομένα, χωρίς ετικέτες
                
                *Παραδείγματα στην εφαρμογή:*
                * **K-Means:** Ομαδοποίηση (Clustering)
                * **DBSCAN:** Εύρεση πυκνότητας και θορύβου
                """)
        st.markdown("**Τύποι:** Clustering, Dimensionality Reduction")

    with c3:
        st.success("### 🎮 Reinforcement Learning")
        st.markdown("**Ενισχυτική Μάθηση**")
        st.write("""
                Ένας **πράκτορας (agent)** μαθαίνει να παίρνει αποφάσεις 
                αλληλεπιδρώντας με ένα περιβάλλον.
                
                **Ανάγκη:** Περιβάλλον + Σύστημα επιβράβευσης/ποινής
                
                *Παράδειγμα στην εφαρμογή:*
                * **Grid World (Q-Learning):** Εκμάθηση διαδρομής
                """)
        st.markdown("**Τύποι:** Model-free, Model-based")

    st.divider()


#   HOW ML WORKS - TRAINING PROCESS
    st.header("⚙️ Πώς Λειτουργεί η Εκπαίδευση ενός Μοντέλου;", anchor = False)
    
    st.write("""
            Η διαδικασία εκπαίδευσης ενός μοντέλου Machine Learning μπορεί να συνοψιστεί στα εξής βήματα:
            """)
    
    #   Expandable sections for each step
    with st.expander("📊 Βήμα 1: Συλλογή και Προετοιμασία Δεδομένων", expanded = True):
        st.write("""
                Η ποιότητα των δεδομένων είναι το πιο κρίσιμο στοιχείο. Τα δεδομένα πρέπει να:
                * Είναι αντιπροσωπευτικά του προβλήματος
                * Να έχουν καθαριστεί από λάθη και ελλείπουσες τιμές
                * Να έχουν μετασχηματιστεί σε μορφή κατανοητή από τον αλγόριθμο
                """)
        st.code("""
                # Παράδειγμα προετοιμασίας δεδομένων
                import pandas as pd
                from sklearn.preprocessing import StandardScaler

                # Φόρτωση δεδομένων
                data = pd.read_csv("data.csv")

                # Καθαρισμός
                data = data.dropna()  # Αφαίρεση κενών τιμών

                # Κανονικοποίηση (Scaling)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(data)
                """, language = "python")

    with st.expander("🎯 Βήμα 2: Επιλογή Αλγορίθμου"):
        st.write("""
        Η επιλογή του κατάλληλου αλγορίθμου εξαρτάται από:
        * **Τύπος προβλήματος:** Regression, Classification, Clustering
        * **Μέγεθος δεδομένων:** Μικρά ή μεγάλα datasets
        * **Ερμηνευσιμότητα:** Χρειαζόμαστε να κατανοήσουμε τις αποφάσεις;
        * **Ακρίβεια:** Πόσο σημαντική είναι η μέγιστη απόδοση;
        """)

    with st.expander("🏋️ Βήμα 3: Εκπαίδευση (Training)"):
        st.write("""
                Κατά την εκπαίδευση, ο αλγόριθμος:
                * Δέχεται τα δεδομένα εισόδου (X) και τις ετικέτες (y) - για supervised learning
                * Προσαρμόζει τα βάρη (weights) του για να ελαχιστοποιήσει το σφάλμα
                * Χρησιμοποιεί μαθηματικές μεθόδους όπως ο **Gradient Descent**
                """)
        st.latex(r"\text{Loss Function: } L = \frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2")
        st.write("Στόχος: Ελαχιστοποίηση του Loss")

    with st.expander("📈 Βήμα 4: Αξιολόγηση (Evaluation)"):
        st.write("""
                Μετά την εκπαίδευση, πρέπει να αξιολογήσουμε το μοντέλο:
                * **Training Set:** Δεδομένα που χρησιμοποιήθηκαν για εκπαίδευση
                * **Test Set:** Δεδομένα που δεν έχει δει το μοντέλο (γενίκευση)
                
                **Βασικά Metrics:**
                * **Accuracy:** Ποσοστό σωστών προβλέψεων
                * **MSE (Mean Squared Error):** Μέσο τετραγωνικό σφάλμα
                * **Precision/Recall:** Για προβλήματα ταξινόμησης
                """)

    with st.expander("🔄 Βήμα 5: Βελτιστοποίηση (Optimization)"):
        st.write("""
                Αν το μοντέλο δεν αποδίδει καλά:
                * **Underfitting:** Το μοντέλο είναι πολύ απλό → Χρειάζεται πιο σύνθετο μοντέλο
                * **Overfitting:** Το μοντέλο έμαθε τα δεδομένα απ' έξω → Χρειάζεται απλούστερο μοντέλο ή περισσότερα δεδομένα
                """)
        
        #   Visualization of Overfitting/Underfitting
        np.random.seed(RANDOM_SEED)
        X_fit = np.linspace(0, 10, 30)
        y_true = 0.5 * X_fit + np.sin(X_fit) + np.random.normal(0, 0.3, 30)
        
        fig_fit, axes = plt.subplots(1, 3, figsize = (12, 3))
        
        #   Underfitting - linear model
        axes[0].scatter(X_fit, y_true, c = 'blue', alpha = 0.6, s = 30)
        axes[0].plot(X_fit, np.polyval(np.polyfit(X_fit, y_true, 1), X_fit), 'r-', linewidth = 2)
        axes[0].set_title('Underfitting\n(Πολύ απλό μοντέλο)', fontsize = 10)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        
        #   Good fit
        axes[1].scatter(X_fit, y_true, c = 'blue', alpha = 0.6, s = 30)
        axes[1].plot(X_fit, np.polyval(np.polyfit(X_fit, y_true, 3), X_fit), 'g-', linewidth = 2)
        axes[1].set_title('Good Fit ✓\n(Ισορροπία)', fontsize = 10)
        axes[1].set_xlabel('X')
        
        #   Overfitting - high degree polynomial
        axes[2].scatter(X_fit, y_true, c = 'blue', alpha = 0.6, s = 30)
        axes[2].plot(X_fit, np.polyval(np.polyfit(X_fit, y_true, 15), X_fit), 'r-', linewidth = 2)
        axes[2].set_title('Overfitting\n(Πολύ σύνθετο μοντέλο)', fontsize = 10)
        axes[2].set_xlabel('X')
        
        plt.tight_layout()
        st.pyplot(fig_fit)
        plt.close(fig_fit)
    st.divider()

#   HISTORY OF ML
    st.header("📜 Σύντομη Ιστορία της Μηχανικής Μάθησης", anchor = False)
    
    col_hist1, col_hist2 = st.columns([2, 1])
    
    with col_hist1:
        st.markdown("""
                    **1950s - Οι Αρχές**
                    * **1950:** Alan Turing προτείνει το "Turing Test" για τεχνητή νοημοσύνη
                    * **1956:** Ο όρος "Artificial Intelligence" στο Dartmouth Conference
                    * **1957:** Frank Rosenblatt εφευρίσκει το **Perceptron** (πρώτο νευρωνικό δίκτυο)
                    
                    **1960s-1980s - Η Εποχή των Συμβολικών Συστήματων**
                    * Εστιάστηκαν σε κανόνες και λογική αντί για μάθηση από δεδομένα
                    * "AI Winter" - Περίοδοι απογοήτευσης και μειωμένης χρηματοδότησης
                    
                    **1990s - Η Επανάσταση της Στατιστικής Μάθησης**
                    * Εμφάνιση αλγορίθμων όπως SVM, Random Forests
                    * Εστίαση σε πρακτικές εφαρμοσμένες λύσεις
                    
                    **2000s-2010s - Big Data & Deep Learning**
                    * **2012:** AlexNet κερδίζει ImageNet - Έκρηξη Deep Learning
                    * Διαθεσιμότητα μεγάλων datasets και GPU υπολογιστικής ισχύος
                    
                    **2020s - Η Εποχή των Μεγάλων Γλωσσικών Μοντέλων**
                    * GPT, BERT, Transformers
                    * Generative AI (ChatGPT, DALL-E, κλπ.)
                    """)
    
    with col_hist2:
        st.info("""
                ### 🎯 ML στην Καθημερινή μας Ζωή
                
                Το ML χρησιμοποιείται πλέον παντού:
                * 📱 Smartphones
                * 🚗 Αυτόνομη οδήγηση
                * 🏥 Ιατρική διάγνωση
                * 💰 Χρηματοοικονομικά
                * 🎮 Παιχνίδια
                """)
    st.divider()

#   REAL-WORLD APPLICATIONS
    st.header("🌐 Πρακτικές Εφαρμογές του Machine Learning", anchor = False)
    
    app_col1, app_col2, app_col3, app_col4 = st.columns(4)
    
    with app_col1:
        st.markdown("#### 🏥 Υγεία")
        st.write("""
                * Διάγνωση ασθενειών
                * Ανακάλυψη φαρμάκων
                * Προσωποποιημένη ιατρική
                """)
    
    with app_col2:
        st.markdown("#### 💳 Οικονομία")
        st.write("""
                * Ανίχνευση απάτης
                * Αξιολόγηση κινδύνου
                * Αλγοριθμικό trading
                """)
    
    with app_col3:
        st.markdown("#### 🛒 Εμπορίου")
        st.write("""
                * Συστάσεις προϊόντων
                * Πρόβλεψη ζήτησης
                * Customer segmentation
                """)
    
    with app_col4:
        st.markdown("#### 🚗 Μεταφορές")
        st.write("""
                * Αυτόνομη οδήγηση
                * Βελτιστοποίηση διαδρομών
                * Πρόβλεψη καθυστερήσεων
                """)
    st.divider()

#   ALGORITHMS OVERVIEW
    st.header("📊 Επισκόπηση Αλγορίθμων της Εφαρμογής", anchor = False)
    
    algo_data = {
                "Αλγόριθμος": ["Linear Regression", "KNN", "K-Means", "DBSCAN", "Q-Learning"],
                "Κατηγορία": ["Supervised", "Supervised", "Unsupervised", "Unsupervised", "Reinforcement"],
                "Είδος Προβλήματος": ["Regression", "Classification", "Clustering", "Clustering", "Control"],
                "Εφαρμογή": ["Πρόβλεψη τιμών", "Ταξινόμηση εικόνων", "Customer segments", "Ανίχνευση ανωμαλιών", "Robotics"]
                }
    
    df_algos = pd.DataFrame(algo_data)
    st.dataframe(df_algos, use_container_width = True, hide_index = True)
    st.divider()

#   ETHICS & CHALLENGES
    st.header("⚖️ Ηθικά Ζητήματα & Προκλήσεις στο ML", anchor = False)
    
    st.write("""
            Καθώς τα συστήματα Machine Learning γίνονται όλο και πιο διαδεδομένα, 
            προκύπτουν σημαντικά ηθικά και πρακτικά ζητήματα που πρέπει να αντιμετωπιστούν:
            """)
    
    ethics_col1, ethics_col2 = st.columns(2)
    
    with ethics_col1:
        st.markdown("#### 🎭 Bias & Fairness (Προκατάληψη)")
        st.warning("""
                    Τα μοντέλα μαθαίνουν από δεδομένα που μπορεί να περιέχουν προκαταλήψεις.
                    
                    **Παράδειγμα:** Ένα σύστημα πρόσληψης που προτιμά άνδρες επειδή τα ιστορικά δεδομένα δείχνουν περισσότερους άνδρες προσληφθέντες.
                    
                    **Λύση:** Ελέγχος δεδομένων, fair algorithms, διαφάνεια.
                    """)
        
        st.markdown("#### 🔒 Privacy (Ιδιωτικότητα)")
        st.info("""
                Τα ML μοντέλα συχνά απαιτούν μεγάλα ποσά προσωπικών δεδομένων.
                
                **Παράδειγμα:** Ιατρικά δεδομένα, συμπεριφορά χρηστών.
                
                **Λύση:** Anonymization, Federated Learning, Differential Privacy.
                """)
    
    with ethics_col2:
        st.markdown("#### 🔮 Interpretability (Ερμηνευσιμότητα)")
        st.error("""
                Πολλά μοντέλα (π.χ. Deep Learning) λειτουργούν ως "μαύρα κουτιά".
                
                **Πρόβλημα:** Δεν μπορούμε να εξηγήσουμε γιατί το μοντέλο πήρε μια απόφαση.
                
                **Λύση:** Explainable AI (XAI), simpler models για κρίσιμες εφαρμογές.
                """)
        
        st.markdown("#### ⚠️ Safety & Security")
        st.success("""
                Τα ML συστήματα μπορεί να είναι ευάλωτα σε επιθέσεις.
                
                **Παράδειγμα:** Adversarial attacks - μικρές αλλαγές σε εικόνα που ξεγελούν το μοντέλο.
                
                **Λύση:** Robust training, security testing.
                """)
    st.divider()

#   QUIZ
    render_quiz("intro")
    st.divider()

#   THESIS INFO
    st.subheader("ℹ️ Πληροφορίες Εργασίας", anchor = False)
    
    st.markdown("""
                    **Τίτλος:** Διερεύνηση και Υλοποίηση Εκπαιδευτικής Εφαρμογής για Βασικές Έννοιες Machine Learning σε Python

                    **Τεχνολογίες:**
                    * 🐍 Python 3.11.5
                    * 💠 Streamlit
                    * 🧮 Scikit-Learn
                    * 📊 Matplotlib / Plotly
                    * 🔢 NumPy
                    """)
    st.divider()
    
#   HINT
    st.success("👈 Χρησιμοποίησε το μενού στην αριστερή πλευρά για να επιλέξεις έναν αλγόριθμο και να πειραματιστείς με τις παραμέτρους του.")