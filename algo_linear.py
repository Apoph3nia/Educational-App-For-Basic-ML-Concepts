import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import apply_global_styles
from quizzes import render_quiz
from conclusions import render_conclusion

def show_linear_regression():
    apply_global_styles()
    
#   TITLE AND THEORY
    st.title("📏 Linear Regression (Γραμμική Παλινδρόμηση)", anchor = False)
    
    st.markdown("""
                Η **Γραμμική Παλινδρόμηση** είναι η προσπάθεια να βρούμε μια ευθεία γραμμή που περιγράφει καλύτερα τη σχέση μεταξύ δύο μεταβλητών ($x$ και $y$).
                
                Η εξίσωση της ευθείας είναι:
                """)
    
    st.latex(r"y = w \cdot x + b")
    
    st.markdown("""
                Όπου:
                * $w$ (**Weight/Slope**): Η κλίση της γραμμής.
                * $b$ (**Bias/Intercept**): Το σημείο που η γραμμή κόβει τον άξονα $y$.
                
                **Στόχος μας:** Να βρούμε τα $w$ και $b$ που ελαχιστοποιούν το **Μέσο Τετραγωνικό Σφάλμα (MSE)**.
                """)
    
    st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{true}^{(i)} - y_{pred}^{(i)})^2")
    st.divider()

#   THEORY: GRADIENT DESCENT
    with st.expander("📚 Πώς βρίσκει ο αλγόριθμος τη βέλτιστη λύση; (Gradient Descent)", expanded = False):
        st.markdown("""
                    ### 🎢 Gradient Descent - Η "κατάβαση" προς το ελάχιστο
                    
                    Ο αλγόριθμος δεν μαντεύει τη λύση, αλλά τη **ψάχνει συστηματικά**. Φαντάσου ότι είσαι σε ένα βουνό και θέλεις να βρεις το χαμηλότερο σημείο (την κοιλάδα). Είναι σκοτεινή νύχτα και βλέπεις μόνο τα πόδια σου.
                    
                    **Τι κάνεις;** Πατάς προς την κατεύθυνση που κατεβαίνει πιο απότομα!
                    
                    #### Μαθηματικά:
                    """)
        
        st.latex(r"w_{new} = w_{old} - \alpha \cdot \frac{\partial MSE}{\partial w}")
        st.latex(r"b_{new} = b_{old} - \alpha \cdot \frac{\partial MSE}{\partial b}")
        
        st.markdown("""
                    Όπου:
                    * $\\alpha$ (**Learning Rate**): Το μέγεθος του βήματος (πόσο γρήγορα κατεβαίνουμε)
                    * $\\frac{\\partial MSE}{\\partial w}$: Η κλίση (gradient) - δείχνει την κατεύθυνση της ανόδου
                    
                    **Προσοχή στο Learning Rate:**
                    * Πολύ μικρό → Αργή σύγκλιση (πολλά βήματα)
                    * Πολύ μεγάλο → Μπορεί να "πηδήξουμε" πάνω από το ελάχιστο!
                    """)
        
        #   Visualization of learning rate
        fig_lr, ax_lr = plt.subplots(figsize = (8, 3))
        x_lr = np.linspace(-3, 3, 100)
        y_lr = x_lr**2
        
        ax_lr.plot(x_lr, y_lr, 'b-', linewidth = 2, label = 'Loss Function')
        
        #   Small LR
        ax_lr.annotate('', xy = (-2.2, 4.84), xytext = (-2.5, 6.25), arrowprops = dict(arrowstyle = '->', color = 'green', lw = 2))
        ax_lr.annotate('', xy = (-1.9, 3.61), xytext = (-2.2, 4.84), arrowprops = dict(arrowstyle = '->', color = 'green', lw = 2))
        ax_lr.text(-2.5, 7, 'Μικρό LR\n(αργό)', fontsize = 9, color = 'green', ha = 'center')
        
        #   Large LR
        ax_lr.annotate('', xy = (2.5, 6.25), xytext = (-0.5, 0.25), arrowprops = dict(arrowstyle = '->', color = 'red', lw = 2))
        ax_lr.text(2.5, 7, 'Μεγάλο LR\n(ασταθές)', fontsize = 9, color = 'red', ha = 'center')
        
        ax_lr.scatter([0], [0], color = 'purple', s = 100, zorder = 5, label = 'Ελάχιστο')
        ax_lr.set_xlabel('w')
        ax_lr.set_ylabel('Loss')
        ax_lr.legend(loc = 'upper right')
        ax_lr.set_title('Επίδραση του Learning Rate')
        plt.tight_layout()
        st.pyplot(fig_lr)
        plt.close(fig_lr)
    st.divider()


#   ΡΥΘΜΙΣΕΙΣ ΔΕΔΟΜΕΝΩΝ (SIDEBAR)
    st.sidebar.header("1. Δημιουργία Δεδομένων")
    
    #   Initialize session state
    if 'X' not in st.session_state:
        st.session_state.X = np.random.rand(50, 1) * 10
        st.session_state.noise = np.random.randn(50, 1)
        st.session_state.y = 2.5 * st.session_state.X + 5.0 + st.session_state.noise
        st.session_state.has_outlier = False
        st.session_state.outlier_X = None
        st.session_state.outlier_y = None

    #   Noise level slider
    noise_level = st.sidebar.slider("🔊 Επίπεδο Θορύβου", 0.0, 5.0, 2.0, 0.5)
    
    #   Generate new data button
    if st.sidebar.button("🎲 Γέννηση Νέων Δεδομένων"):
        st.session_state.X = np.random.rand(50, 1) * 10
        st.session_state.noise = np.random.randn(50, 1) * noise_level
        st.session_state.y = 2.5 * st.session_state.X + 5.0 + st.session_state.noise
        st.session_state.has_outlier = False
        st.session_state.outlier_X = None
        st.session_state.outlier_y = None
        st.rerun()

    X = st.session_state.X.copy()
    y = st.session_state.y.copy()


#   OUTLIER FEATURE
    st.sidebar.header("2. Outlier")
    
    st.sidebar.markdown("""Πρόσθεσε ένα "ακραίο σημείο" για να δεις πώς επηρεάζει τη γραμμή!""")
    
    outlier_col1, outlier_col2 = st.sidebar.columns(2)
    
    with outlier_col1:
        outlier_x = st.number_input("X θέση", min_value = 0.0, max_value = 15.0, value = 12.0, step = 0.5)
    with outlier_col2:
        outlier_y = st.number_input("Y θέση", min_value = -10.0, max_value = 50.0, value = 40.0, step = 1.0)
    
    if st.sidebar.button("➕ Προσθήκη Outlier"):
        st.session_state.has_outlier = True
        st.session_state.outlier_X = outlier_x
        st.session_state.outlier_y = outlier_y
        st.rerun()
    
    if st.session_state.has_outlier:
        if st.sidebar.button("➖ Αφαίρεση Outlier"):
            st.session_state.has_outlier = False
            st.session_state.outlier_X = None
            st.session_state.outlier_y = None
            st.rerun()
        
        #   Add outlier to data
        X = np.vstack([X, [[st.session_state.outlier_X]]])
        y = np.vstack([y, [[st.session_state.outlier_y]]])
        st.sidebar.warning(f"⚠️ Outlier ενεργό στο ({st.session_state.outlier_X}, {st.session_state.outlier_y})")

#   TABS FOR DIFFERENT MODES
    tab1, tab2, tab3 = st.tabs(["🎮 Manual Mode", "🎢 Gradient Descent", "📊 Cost Surface"])
    
    #   TAB 1: MANUAL MODE
    with tab1:
        st.subheader("🎯 Προσπάθησε να βρεις τη βέλτιστη γραμμή!", anchor = False)
        
        col1, col2 = st.columns([3, 1])

        with col2:
            st.markdown("#### ⚙️ Ρυθμίσεις")
            st.info("Προσάρμοσε τα sliders για να μειώσεις το MSE!")
            
            w_guess = st.slider(
                "Κλίση (w)", 
                min_value = -5.0, 
                max_value = 10.0, 
                value = 1.0, 
                step = 0.01,
                format = "%.2f",
                key = "w_manual"
            )
            
            b_guess = st.slider(
                "Σταθερά (b)", 
                min_value = -10.0, 
                max_value = 20.0, 
                value = 0.0, 
                step = 0.01,
                format = "%.2f",
                key = "b_manual"
            )
            
            show_residuals = st.checkbox("Εμφάνιση Residuals", value = True, key = "res_manual")
            show_best_fit = st.checkbox("Δείξε Βέλτιστη Λύση", value = False, key = "best_manual")

        with col1:
            y_pred_user = w_guess * X + b_guess
            mse_user = mean_squared_error(y, y_pred_user)

            fig, ax = plt.subplots(figsize = (10, 6))
            
            #   Plot data
            ax.scatter(st.session_state.X, 
                      st.session_state.y, 
                      color = 'blue', label = 'Δεδομένα', alpha = 0.6, s = 50)
            
            #   Highlight outlier
            if st.session_state.has_outlier:
                ax.scatter([st.session_state.outlier_X], [st.session_state.outlier_y], color = 'orange', s = 150, marker = '*', label = 'Outlier', zorder = 5)
            
            #   User's line
            ax.plot(X, y_pred_user, color = 'red', linewidth = 2, label = f'Η Γραμμή σου (MSE: {mse_user:.2f})')
            
            #   Residuals
            if show_residuals:
                for i in range(len(X)):
                    ax.plot([X[i], X[i]], [y[i], y_pred_user[i]], color = 'gray', linestyle = '--', alpha = 0.3, linewidth = 0.5)

            #   Best fit
            if show_best_fit:
                model = LinearRegression()
                model.fit(X, y)
                y_best = model.predict(X)
                mse_best = mean_squared_error(y, y_best)
                
                ax.plot(X, y_best, color = 'green', linestyle = '--', linewidth = 3, label = f'Βέλτιστη (MSE: {mse_best:.2f})')
                
                #   Show equation
                st.success(f"✅ Βέλτιστη: w = {model.coef_[0][0]:.2f}, b = {model.intercept_[0]:.2f}")

            ax.set_xlabel("X (Χαρακτηριστικό)")
            ax.set_ylabel("y (Στόχος)")
            ax.legend(loc = 'upper left')
            ax.grid(True, alpha = 0.3)
            
            st.pyplot(fig)
            plt.close(fig)

        #   Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Το MSE σου", f"{mse_user:.2f}")
        
        if show_best_fit:
            model = LinearRegression()
            model.fit(X, y)
            y_best = model.predict(X)
            mse_best = mean_squared_error(y, y_best)
            m2.metric("Ελάχιστο MSE", f"{mse_best:.2f}", delta = f"{mse_user - mse_best:.2f}")
            
            if st.session_state.has_outlier:
                #   Compare with outlier removed
                model_clean = LinearRegression()
                model_clean.fit(st.session_state.X, st.session_state.y)
                mse_clean = mean_squared_error(st.session_state.y, model_clean.predict(st.session_state.X))
                m3.metric("MSE χωρίς Outlier", f"{mse_clean:.2f}")
        
        if st.session_state.has_outlier:
            st.error("""
                    ⚠️ **Παρατήρησε:** Το outlier "τραβάει" τη γραμμή προς τα εκεί! 
                    Αυτό δείχνει πόσο ευαίσθητη είναι η Γραμμική Παλινδρόμηση σε ακραίες τιμές.
                    """)

    #   TAB 2: GRADIENT DESCENT VISUALIZATION
    with tab2:
        st.subheader("🎢 Δες τον Gradient Descent σε δράση!", anchor = False)
        
        col_gd1, col_gd2 = st.columns([1, 2])
        
        with col_gd1:
            st.markdown("#### ⚙️ Παράμετροι")
            
            learning_rate = st.slider(
                "Learning Rate (α)", 
                min_value = 0.001, 
                max_value = 0.1, 
                value = 0.01, 
                step = 0.001,
                format = "%.3f",
                key = "lr_gd"
            )
            
            n_iterations = st.slider(
                "Αριθμός Επαναλήψεων", 
                min_value = 10, 
                max_value = 200, 
                value = 50, 
                step = 10,
                key = "iter_gd"
            )
            
            start_w = st.slider(
                "Αρχική Κλίση (w)", 
                min_value = -2.0, 
                max_value = 8.0, 
                value = 0.0, 
                step = 0.5,
                key = "start_w"
            )
            
            start_b = st.slider(
                "Αρχική Σταθερά (b)", 
                min_value = -5.0, 
                max_value = 15.0, 
                value = 0.0, 
                step = 0.5,
                key = "start_b"
            )
            
            run_gd = st.button("🚀 Τρέξε Gradient Descent", type = "primary")
        
        with col_gd2:
            if run_gd:
                #   Gradient Descent Implementation
                w, b = start_w, start_b
                history_w, history_b = [w], [b]
                history_mse = [mean_squared_error(y, w * X + b)]
                
                X_flat = X.flatten()
                y_flat = y.flatten()
                n = len(X_flat)
                
                for i in range(n_iterations):
                    y_pred = w * X_flat + b
                    
                    #   Gradients
                    dw = (-2/n) * np.sum(X_flat * (y_flat - y_pred))
                    db = (-2/n) * np.sum(y_flat - y_pred)
                    
                    #   Update
                    w = w - learning_rate * dw
                    b = b - learning_rate * db
                    
                    history_w.append(w)
                    history_b.append(b)
                    history_mse.append(mean_squared_error(y, w * X + b))
                
                #   Plot the progress
                fig_gd, axes_gd = plt.subplots(1, 2, figsize = (12, 5))
                
                #   Left: Data with evolving line
                ax1 = axes_gd[0]
                ax1.scatter(X_flat, y_flat, color = 'blue', alpha = 0.5, label = 'Δεδομένα')
                
                #   Show final line
                final_w = history_w[-1]
                final_b = history_b[-1]
                X_line = np.linspace(X.min(), X.max(), 100)
                ax1.plot(X_line, final_w * X_line + final_b, 'r-', linewidth = 2, label=f'Τελική: w = {final_w:.2f}, b = {final_b:.2f}')
                
                #   Show starting line
                ax1.plot(X_line, start_w * X_line + start_b, 'g--', linewidth = 1, alpha = 0.5, label = f'Αρχική: w = {start_w:.2f}, b = {start_b:.2f}')
                
                ax1.set_xlabel('X')
                ax1.set_ylabel('y')
                ax1.legend()
                ax1.set_title('Εξέλιξη της Γραμμής')
                ax1.grid(True, alpha = 0.3)
                
                #   Right: MSE over iterations
                ax2 = axes_gd[1]
                ax2.plot(history_mse, 'b-', linewidth = 2)
                ax2.set_xlabel('Επανάληψη')
                ax2.set_ylabel('MSE')
                ax2.set_title('MSE vs Επαναλήψεις')
                ax2.grid(True, alpha = 0.3)
                ax2.set_yscale('log')
                
                plt.tight_layout()
                st.pyplot(fig_gd)
                plt.close(fig_gd)
                
                #   Show final results
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Αρχικό MSE", f"{history_mse[0]:.2f}")
                col_r2.metric("Τελικό MSE", f"{history_mse[-1]:.2f}", delta = f"-{history_mse[0] - history_mse[-1]:.2f}")
                
                #   Compare with sklearn
                model = LinearRegression()
                model.fit(X, y)
                optimal_mse = mean_squared_error(y, model.predict(X))
                col_r3.metric("Βέλτιστο MSE", f"{optimal_mse:.2f}")
                
                st.success(f"✅ Gradient Descent ολοκληρώθηκε! w = {final_w:.2f}, b = {final_b:.2f}")
            else:
                #   Show explanation
                fig_demo, ax_demo = plt.subplots(figsize = (8, 4))
                x_demo = np.linspace(0, 10, 100)
                y_demo = 2.5 * x_demo + 5 + np.random.randn(100) * 0.5
                
                ax_demo.scatter(x_demo, y_demo, alpha = 0.5)
                ax_demo.set_xlabel('X')
                ax_demo.set_ylabel('y')
                ax_demo.set_title('Παράδειγμα Δεδομένων')
                ax_demo.grid(True, alpha=0.3)
                st.pyplot(fig_demo)
                plt.close(fig_demo)

    #   TAB 3: COST SURFACE VISUALIZATION
    with tab3:
        st.subheader("📊 Η Επιφάνεια Κόστους (Cost Surface)", anchor = False)
        
        st.markdown("""
                    Το **Cost Surface** δείχνει το MSE για όλους τους πιθανούς συνδυασμούς των $w$ και $b$.
                    Ο στόχος είναι να βρούμε το "βάθος" της κοιλάδας - το ελάχιστο MSE.
                    """)
        
        #   Calculate optimal values first
        model = LinearRegression()
        model.fit(X, y)
        optimal_w = model.coef_[0][0]
        optimal_b = model.intercept_[0]
        
        col_cs1, col_cs2 = st.columns([1, 2])
        
        with col_cs1:
            st.markdown("#### ⚙️ Ρυθμίσεις")
            
            w_range = st.slider(
                "Εύρος w γύρω από το βέλτιστο", 
                min_value = 1.0, 
                max_value = 5.0, 
                value = 3.0, 
                step = 0.5,
                key = "w_range"
            )
            
            b_range = st.slider(
                "Εύρος b γύρω από το βέλτιστο", 
                min_value = 2.0, 
                max_value = 10.0, 
                value = 5.0, 
                step = 0.5,
                key = "b_range"
            )
            
            show_path = st.checkbox("Δείξε διαδρομή GD", value = True, key = "show_path")
            
            if show_path:
                path_lr = st.slider("Learning Rate για διαδρομή", 0.001, 0.05, 0.01, 0.001, key = "path_lr")
        
        with col_cs2:
            #   Create meshgrid for cost surface
            w_vals = np.linspace(optimal_w - w_range, optimal_w + w_range, 50)
            b_vals = np.linspace(optimal_b - b_range, optimal_b + b_range, 50)
            W, B = np.meshgrid(w_vals, b_vals)
            
            X_flat = X.flatten()
            y_flat = y.flatten()
            
            #   Calculate MSE for each (w, b)
            MSE = np.zeros_like(W)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    y_pred = W[i, j] * X_flat + B[i, j]
                    MSE[i, j] = mean_squared_error(y, y_pred)
            
            #   Create 3D plot
            fig_3d = plt.figure(figsize = (10, 7))
            ax_3d = fig_3d.add_subplot(111, projection = '3d')
            
            #   Surface
            surf = ax_3d.plot_surface(W, B, MSE, cmap = 'viridis', alpha = 0.8)
            
            #   Mark optimal point
            ax_3d.scatter([optimal_w], [optimal_b], [MSE.min()], color = 'red', s = 100, label = 'Βέλτιστο', zorder = 5)
            
            #   Show GD path
            if show_path:
                w_path = optimal_w - w_range * 0.8
                b_path = optimal_b - b_range * 0.8
                path_w = [w_path]
                path_b = [b_path]
                path_mse = [mean_squared_error(y, w_path * X_flat + b_path)]
                
                n = len(X_flat)
                for _ in range(100):
                    y_pred = w_path * X_flat + b_path
                    dw = (-2/n) * np.sum(X_flat * (y_flat - y_pred))
                    db = (-2/n) * np.sum(y_flat - y_pred)
                    w_path = w_path - path_lr * dw
                    b_path = b_path - path_lr * db
                    path_w.append(w_path)
                    path_b.append(b_path)
                    path_mse.append(mean_squared_error(y, w_path * X_flat + b_path))
                
                ax_3d.plot(path_w, path_b, path_mse, 'r.-', linewidth = 2, markersize = 3, label = 'Διαδρομή GD')
            
            ax_3d.set_xlabel('w (Κλίση)')
            ax_3d.set_ylabel('b (Σταθερά)')
            ax_3d.set_zlabel('MSE')
            ax_3d.set_title('Cost Surface (3D)')
            ax_3d.legend()
            
            fig_3d.colorbar(surf, shrink=0.5, aspect=5, label='MSE')
            
            st.pyplot(fig_3d)
            plt.close(fig_3d)
            
            st.success(f"🔴 Κόκκινο σημείο: Βέλτιστο w = {optimal_w:.2f}, b = {optimal_b:.2f}")
        
        #   2D Contour plot
        st.markdown("### 🗺️ Contour Plot (Προβολή από πάνω)")
        
        fig_contour, ax_contour = plt.subplots(figsize=(10, 6))
        
        contour = ax_contour.contour(W, B, MSE, levels = 20, cmap = 'viridis')
        ax_contour.clabel(contour, inline = True, fontsize = 8)
        
        #   Mark optimal
        ax_contour.scatter([optimal_w], [optimal_b], color = 'red', s = 100, marker = '*', label = 'Βέλτιστο', zorder = 5)
        
        #   Show path
        if show_path:
            ax_contour.plot(path_w, path_b, 'r.-', linewidth = 2, markersize = 4, label = 'Διαδρομή GD')
        
        ax_contour.set_xlabel('w (Κλίση)')
        ax_contour.set_ylabel('b (Σταθερά)')
        ax_contour.set_title('Contour Plot του MSE')
        ax_contour.legend()
        ax_contour.grid(True, alpha = 0.3)
        
        st.pyplot(fig_contour)
        plt.close(fig_contour)

    #   TAB 4: ADVANCED METRICS & ASSUMPTIONS
    with st.expander("📈 Προηγμένα Metrics & Παραδοχές", expanded = False):
        #   Fit the best model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        #   Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        #   Metrics display
        st.markdown("### 📊 Μετρικές Απόδοσης")
        
        met_col1, met_col2, met_col3 = st.columns(3)
        
        with met_col1:
            st.metric("MSE", f"{mse:.2f}", help = "Mean Squared Error - Μέσο Τετραγωνικό Σφάλμα")
        
        with met_col2:
            st.metric("RMSE", f"{rmse:.2f}", help = "Root Mean Squared Error - Στην ίδια μονάδα με y")
        
        with met_col3:
            st.metric("R² Score", f"{r2:.4f}", help = "Coefficient of Determination - Ποσοστό μεταβλητότητας που εξηγείται")
        
        st.markdown("""
                    **Τι σημαίνει το R²;**
                    * $R^2 = 1.0$: Τέλειο ταίριασμα (σπάνιο)
                    * $R^2 = 0.0$: Το μοντέλο δεν εξηγεί τίποτα
                    * $R^2 < 0$: Το μοντέλο είναι χειρότερο από τη μέση τιμή!
                    """)
        st.divider()
        
        #   Assumptions
        st.markdown("### 📋 Παραδοχές της Γραμμικής Παλινδρόμησης")
        
        assumptions_col1, assumptions_col2 = st.columns(2)
        
        with assumptions_col1:
            st.info("""
                    **1. Linearity (Γραμμικότητα):**\n
                    Η σχέση μεταξύ X και y πρέπει να είναι γραμμική.
                    
                    **2. Independence (Ανεξαρτησία):**\n
                    Τα residuals δεν πρέπει να έχουν autocorrelation.
                    """)
            
            st.warning("""
                    **3. Homoscedasticity (Ομοσκεδαστικότητα):**\n
                    Η διακύμανση των residuals πρέπει να είναι σταθερή για όλα τα X.
                    """)
        
        with assumptions_col2:
            st.success("""
                    **4. Normality (Κανονικότητα):**\n
                    Τα residuals πρέπει να ακολουθούν κανονική κατανομή.
                    """)
            
            st.error("""
                    **⚠️ Αν παραβιάζονται οι παραδοχές:**
                    * Μη γραμμική σχέση → Πολυωνυμική παλινδρόμηση
                    * Heteroscedasticity → Weighted regression
                    * Outliers → Robust regression (Ridge, Lasso)
                    """)
        st.divider()
        
        #   Residuals Histogram
        st.markdown("### 📊 Ανάλυση Residuals")
        
        residuals = y - y_pred
        
        fig_hist, axes_hist = plt.subplots(1, 2, figsize = (12, 4))
        
        #   Histogram of residuals
        axes_hist[0].hist(residuals, bins = 20, color = 'steelblue', edgecolor = 'black', alpha = 0.7)
        axes_hist[0].axvline(x = 0, color = 'red', linestyle = '--', linewidth = 2, label = 'Μέσος = 0')
        axes_hist[0].set_xlabel('Residuals')
        axes_hist[0].set_ylabel('Συχνότητα')
        axes_hist[0].set_title('Κατανομή Residuals')
        axes_hist[0].legend()
        axes_hist[0].grid(True, alpha = 0.3)
        
        #   Residuals vs Predicted
        axes_hist[1].scatter(y_pred, residuals, color = 'steelblue', alpha = 0.6)
        axes_hist[1].axhline(y = 0, color = 'red', linestyle = '--', linewidth = 2)
        axes_hist[1].set_xlabel('Predicted Values')
        axes_hist[1].set_ylabel('Residuals')
        axes_hist[1].set_title('Residuals vs Predicted')
        axes_hist[1].grid(True, alpha = 0.3)
        
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)
        
        #   Interpretation
        residuals_mean = np.mean(residuals)
        residuals_std = np.std(residuals)
        
        if abs(residuals_mean) < 0.5:
            st.success(f"✅ Το μέσο των residuals είναι κοντά στο μηδέν (MEAN:{residuals_mean:.3f} | STD: {residuals_std:.3f})")
        else:
            st.warning(f"⚠️ Το μέσο των residuals αποκλίνει από το μηδέν (MEAN:{residuals_mean:.3f} | STD: {residuals_std:.3f})")

#   REAL-WORLD EXAMPLE (BOSTON DATASET)
    with st.expander("🏠 Πραγματικό Παράδειγμα: Πρόβλεψη Αξίας Σπιτιού", expanded = False):
        st.markdown("""
                    ### Πραγματικά Δεδομένα (Boston Housing Dataset)
                    
                    Σε αυτό το παράδειγμα θα προσπαθήσουμε να προβλέψουμε την αξία ενός σπιτιού στη Βοστώνη (σε δολάρια) 
                    ανάλογα με τον **αριθμό των δωματίων (RM)** του, χρησιμοποιώντας ένα πραγματικό και ιστορικό dataset (CMU StatLib).
                    """)
        
        @st.cache_data
        def load_boston_data():
            #   Load boston dataset
            data_url = "./src/BostonHousing.csv"
            df = pd.read_csv(data_url)
            
            #   Choose the average number of rooms (RM - Index 5) as X
            X_rooms = df['rm'].values.reshape(-1, 1)
            
            #   The price (target) is in $1000, so we multiply by 1000
            y_price = df['medv'].values.reshape(-1, 1) * 1000
            
            return X_rooms, y_price
        
        try:
            rooms, price = load_boston_data()
            
            #   Train model
            house_model = LinearRegression()
            house_model.fit(rooms, price)
            
            #   Layout for prediction and plot
            col_plot, col_controls = st.columns([3, 1])
            
            with col_controls:
                st.markdown("### 🎯 Πρόβλεψη")
                user_rooms = st.slider("Αριθμός Δωματίων:", min_value = 1.0, max_value = 10.0, value = 6.0, step = 1.0)
                
                #   Predict price based on user input
                predicted_price = house_model.predict([[user_rooms]])[0][0]
                predicted_price = max(0, predicted_price)
                
                st.success(f"💰 Προβλεπόμενη αξία:\n\n**${predicted_price:,.0f}**")
                
                st.info(f"""
                        **Εξίσωση Μοντέλου:** \n\nΑξία = {house_model.coef_[0][0]:.0f}$ × (Δωμάτια) + {house_model.intercept_[0]:,.0f}$
                        \n\n*Κάθε επιπλέον δωμάτιο προσθέτει κατά μέσο όρο **${house_model.coef_[0][0]:,.0f}** στην αξία.*
                        """)

            #   Predict values for plotting
            rooms_line = np.linspace(rooms.min(), rooms.max(), 100).reshape(-1, 1)
            price_line = house_model.predict(rooms_line)
            
            fig_house, ax_house = plt.subplots(figsize = (10, 6))
            ax_house.scatter(rooms, price, color = 'blue', alpha = 0.4, label = 'Δεδομένα Σπιτιών (Βοστώνη)')
            ax_house.plot(rooms_line, price_line, 'r-', linewidth = 2, label = 'Γραμμή Παλινδρόμησης')
            
            #   Highlight user's prediction
            ax_house.scatter([user_rooms], [predicted_price], color = 'gold', s = 300, marker = '*', edgecolors = 'black', linewidths = 1.5, zorder = 5, label = 'Η Πρόβλεψή σου')
            ax_house.plot([user_rooms, user_rooms], [0, predicted_price], color = 'gray', linestyle = '--', alpha = 0.7)
            ax_house.plot([0, user_rooms], [predicted_price, predicted_price], color = 'gray', linestyle = '--', alpha = 0.7)
            
            ax_house.set_xlim(left = 0)
            ax_house.set_ylim(bottom = 0)
            
            ax_house.set_xlabel('Αριθμός Δωματίων (RM)')
            ax_house.set_ylabel('Αξία ($)')
            ax_house.set_title('Γραμμική Παλινδρόμηση: Αξία Σπιτιού vs Αριθμός Δωματίων')
            ax_house.legend()
            ax_house.grid(True, alpha = 0.3)
            
            with col_plot:
                st.pyplot(fig_house)
            plt.close(fig_house)
            
        except Exception as e:
            st.error(f"⚠️ Σφάλμα κατά τη φόρτωση των δεδομένων: {e}")

#   QUIZ
    render_quiz("linear_regression")

#   CONCLUSION
    render_conclusion("linear_regression")