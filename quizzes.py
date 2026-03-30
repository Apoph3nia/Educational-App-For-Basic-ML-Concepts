import streamlit as st

QUIZ_DATA = {
    "linear_regression": {
        "title": "🎯 Quiz: Τι έμαθες για τη Γραμμική Παλινδρόμηση;",
        "questions": [
            {
                "q": "1. Ποιος είναι ο στόχος της Γραμμικής Παλινδρόμησης;",
                "options": ["Να μεγιστοποιήσει το MSE", "Να ελαχιστοποιήσει το MSE", "Να βρει τη μέση τιμή"],
                "correct": "Να ελαχιστοποιήσει το MSE"
            },
            {
                "q": "2. Τι μετράει το R² Score;",
                "options": ["Το σφάλμα σε €", "Το ποσοστό μεταβλητότητας που εξηγείται", "Τον αριθμό των δεδομένων"],
                "correct": "Το ποσοστό μεταβλητότητας που εξηγείται"
            },
            {
                "q": "3. Τι συμβαίνει αν το Learning Rate είναι πολύ μεγάλο;",
                "options": ["Η εκπαίδευση είναι πολύ αργή", "Ο αλγόριθμος μπορεί να αποκλίνει", "Τίποτα δεν αλλάζει"],
                "correct": "Ο αλγόριθμος μπορεί να αποκλίνει"
            },
            {
                "q": "4. Πώς επηρεάζει ένα outlier τη γραμμή παλινδρόμησης;",
                "options": ["Δεν την επηρεάζει", "Την τραβάει προς το σημείο του", "Την κάνει πιο επίπεδη"],
                "correct": "Την τραβάει προς το σημείο του"
            },
            {
                "q": "5. Ποια είναι η εξίσωση της ευθείας;",
                "options": ["y = w + b", "y = w · x + b", "y = x + w · b"],
                "correct": "y = w · x + b"
            }
        ],
        "success_msg": "Κατανόησες πλήρως τη Γραμμική Παλινδρόμηση!"
    },
    "knn": {
        "title": "🎯 Quiz: Τι έμαθες για το KNN;",
        "questions": [
            {
                "q": "1. Τι συμβαίνει όταν το K είναι πολύ μικρό (π.χ. K=1);",
                "options": ["Overfitting", "Underfitting", "Δεν επηρεάζει"],
                "correct": "Overfitting"
            },
            {
                "q": "2. Ποια είναι η πιο συνηθισμένη απόσταση στο KNN;",
                "options": ["Euclidean Distance", "Manhattan Distance", "Cosine Distance"],
                "correct": "Euclidean Distance"
            },
            {
                "q": "3. Γιατί προτιμούμε περιττούς αριθμούς για το K σε binary classification;",
                "options": ["Για ταχύτητα", "Για να αποφύγουμε ισοπαλίες", "Δεν υπάρχει λόγος"],
                "correct": "Για να αποφύγουμε ισοπαλίες"
            },
            {
                "q": "4. Το KNN είναι επιβλεπόμενη ή μη επιβλεπόμενη μάθηση;",
                "options": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
                "correct": "Supervised Learning"
            },
            {
                "q": "5. Τι συμβαίνει όταν το K είναι πολύ μεγάλο;",
                "options": ["Overfitting", "Underfitting", "Καλύτερη ακρίβεια"],
                "correct": "Underfitting"
            }
        ],
        "success_msg": "Κατανόησες πλήρως το KNN!"
    },
    "kmeans": {
        "title": "🎯 Quiz: Τι έμαθες για το K-Means;",
        "questions": [
            {
                "q": "1. Πόσα βήματα έχει κάθε επανάληψη του K-Means;",
                "options": ["1", "2", "3"],
                "correct": "2"
            },
            {
                "q": "2. Τι μετράει η Inertia;",
                "options": ["Απόσταση μεταξύ centroids", "Άθροισμα τετραγωνικών αποστάσεων από centroids", "Αριθμό clusters"],
                "correct": "Άθροισμα τετραγωνικών αποστάσεων από centroids"
            },
            {
                "q": "3. Τι είναι το K-Means++;",
                "options": ["Μια βελτιωμένη αρχικοποίηση", "Ένας νέος αλγόριθμος", "Τρόπος υπολογισμού του K"],
                "correct": "Μια βελτιωμένη αρχικοποίηση"
            },
            {
                "q": "4. Το K-Means είναι supervised ή unsupervised;",
                "options": ["Supervised Learning", "Unsupervised Learning", "Semi-supervised Learning"],
                "correct": "Unsupervised Learning"
            },
            {
                "q": "5. Ποια μέθοδος βοηθά να βρούμε το βέλτιστο K;",
                "options": ["Cross-validation", "Elbow Method", "Gradient Descent"],
                "correct": "Elbow Method"
            }
        ],
        "success_msg": "Κατανόησες πλήρως το K-Means!"
    },
    "dbscan": {
        "title": "🎯 Quiz: Τι έμαθες για το DBSCAN;",
        "questions": [
            {
                "q": "1. Τι χρειάζεται να ορίσουμε εκ των προτέρων στο DBSCAN;",
                "options": ["Αριθμό clusters", "Ε και minPts", "Τίποτα"],
                "correct": "Ε και minPts"
            },
            {
                "q": "2. Τι είναι ένα core point;",
                "options": ["Ένα σημείο με τουλάχιστον minPts γείτονες εντός ε", "Το κέντρο ενός cluster", "Ένα outlier"],
                "correct": "Ένα σημείο με τουλάχιστον minPts γείτονες εντός ε"
            },
            {
                "q": "3. Πώς σημειώνονται τα noise points στο DBSCAN;",
                "options": ["Με label 0", "Με label -1", "Με label 'noise'"],
                "correct": "Με label -1"
            },
            {
                "q": "4. Ποιο είναι το βασικό πλεονέκτημα του DBSCAN έναντι του K-Means;",
                "options": ["Είναι πιο γρήγορος", "Δεν χρειάζεται να ορίσουμε τον αριθμό clusters", "Βρίσκει πάντα σφαιρικά clusters"],
                "correct": "Δεν χρειάζεται να ορίσουμε τον αριθμό clusters"
            },
            {
                "q": "5. Τι δείχνει το K-Distance Graph;",
                "options": ["Τον αριθμό των clusters", "Κατάλληλη τιμή για το ε", "Τον αριθμό των outliers"],
                "correct": "Κατάλληλη τιμή για το ε"
            }
        ],
        "success_msg": "Κατανόησες πλήρως το DBSCAN!"
    },
    "rl": {
        "title": "🎯 Quiz: Τι έμαθες για το Q-Learning;",
        "questions": [
            {
                "q": "1. Τι αναπαριστά το Q(s, a);",
                "options": ["Την κατάσταση s", "Την αναμενόμενη ανταμοιβή από την ενέργεια a στην κατάσταση s", "Την άμεση ανταμοιβή"],
                "correct": "Την αναμενόμενη ανταμοιβή από την ενέργεια a στην κατάσταση s"
            },
            {
                "q": "2. Τι είναι το discount factor (γ);",
                "options": ["Πόσο γρήγορα μαθαίνουμε", "Πόσο εκτιμούμε μελλοντικές ανταμοιβές", "Την πιθανότητα exploration"],
                "correct": "Πόσο εκτιμούμε μελλοντικές ανταμοιβές"
            },
            {
                "q": "3. Σε τι χρησιμεύει το epsilon;",
                "options": ["Να αυξήσει το learning rate", "Να ισορροπήσει exploration και exploitation", "Να μειώσει τον αριθμό επεισοδίων"],
                "correct": "Να ισορροπήσει exploration και exploitation"
            },
            {
                "q": "4. Το Q-Learning είναι:",
                "options": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
                "correct": "Reinforcement Learning"
            },
            {
                "q": "5. Τι συμβαίνει αν γ = 0;",
                "options": ["Ο agent νοιάζεται μόνο για άμεσες ανταμοιβές", "Ο agent αγνοεί όλες τις ανταμοιβές", "Ο agent μαθαίνει αργά"],
                "correct": "Ο agent νοιάζεται μόνο για άμεσες ανταμοιβές"
            }
        ],
        "success_msg": "Κατανόησες πλήρως το Q-Learning!"
    },
    "intro": {
        "title": "🎯 Quiz: Τι έμαθες;",
        "questions": [
            {
                "q": "Ερώτηση 1: Ποια κατηγορία ML χρησιμοποιεί δεδομένα με ετικέτες;",
                "options": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
                "correct": "Supervised Learning"
            },
            {
                "q": "Ερώτηση 2: Τι συμβαίνει όταν ένα μοντέλο κάνει overfitting;",
                "options": [
                    "Είναι πολύ απλό και δεν μαθαίνει καλά", 
                    "Έμαθε τα δεδομένα εκπαίδευσης απ' έξω", 
                    "Δεν χρειάζεται περισσότερα δεδομένα"
                ],
                "correct": "Έμαθε τα δεδομένα εκπαίδευσης απ' έξω"
            },
            {
                "q": "Ερώτηση 3: Ποιος αλγόριθμος χρησιμοποιείται για Clustering;",
                "options": ["Linear Regression", "KNN", "K-Means"],
                "correct": "K-Means"
            },
            {
                "q": "Ερώτηση 4: Τι είναι το MSE;",
                "options": [
                    "Ένα είδος ταξινόμησης", 
                    "Ένα metric για τη μέτρηση σφάλματος σε regression", 
                    "Ένας αλγόριθμος clustering"
                ],
                "correct": "Ένα metric για τη μέτρηση σφάλματος σε regression"
            },
            {
                "q": "Ερώτηση 5: Στο Reinforcement Learning, πώς μαθαίνει ο πράκτορας;",
                "options": [
                    "Από ετικετοποιημένα δεδομένα", 
                    "Μέσω επιβράβευσης και ποινής", 
                    "Από στατικά δεδομένα"
                ],
                "correct": "Μέσω επιβράβευσης και ποινής"
            }
        ],
        "success_msg": "Κατανόησες εξαιρετικά τις βασικές έννοιες!"
    }
}

def render_quiz(quiz_key):
    quiz_info = QUIZ_DATA.get(quiz_key)
    if not quiz_info:
        st.error(f"Quiz '{quiz_key}' not found.")
        return
        
    with st.expander(quiz_info["title"], expanded = False):
        st.markdown("Δοκίμασε το παρακάτω Quiz:")
        
        questions = quiz_info["questions"]
        
        with st.form(f"{quiz_key}_quiz_form"):
            for i, question in enumerate(questions):
                st.markdown(f"**{question['q']}**")
                    
                st.radio(
                    f"Επιλογή {i + 1}:", 
                    question['options'], 
                    key = f"{quiz_key}_quiz_{i}", 
                    label_visibility = "collapsed"
                )
                st.divider()
            
            submitted = st.form_submit_button("📊 Εμφάνιση Αποτελεσμάτων")
            
            if submitted:
                score = 0
                for i, question in enumerate(questions):
                    user_answer = st.session_state.get(f"{quiz_key}_quiz_{i}")
                    if user_answer == question['correct']:
                        score += 1
                
                total = len(questions)
                if score == total:
                    msg = quiz_info["success_msg"]
                    st.success(f"🎉 Τέλειο! Σκορ: {score}/{total}! {msg}")
                elif score >= total // 2 + 1:
                    st.success(f"✅ Καλό! Σκορ: {score}/{total}. Συνέχισε την εξερεύνηση!")
                else:
                    st.warning(f"📖 Σκορ: {score}/{total}. Ίσως χρειαστεί να ξαναδιαβάσεις την ενότητα.")
