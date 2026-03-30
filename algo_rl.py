import streamlit as st, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from utils import apply_global_styles, draw_base_grid
from quizzes import render_quiz
from conclusions import render_conclusion

def show_rl():
    apply_global_styles()
    
#   TITLE & THEORY
    st.title("🤖 Q-Learning (Reinforcement Learning)", anchor = False)
    
    st.markdown("""
                Το **Q-Learning** είναι ένας αλγόριθμος **Reinforcement Learning** όπου ένας agent 
                μαθαίνει να παίρνει βέλτιστες αποφάσεις μέσω πειραματισμού!
                
                ### 🎯 Κεντρική Ιδέα: "Μάθηση από την Εμπειρία"
                
                Ο agent εκπαιδεύεται μέσω trial-and-error, μαθαίνοντας ποιες ενέργειες 
                οδηγούν σε καλύτερες ανταμοιβές.
                """)
    st.divider()
    
#   THEORY: Q-LEARNING CONCEPTS
    with st.expander("📚 Βασικές Έννοιες του Q-Learning", expanded = False):
        st.markdown("""### 🔑 Βασικά Στοιχεία:""")
        
        col_rl1, col_rl2, col_rl3 = st.columns(3)
        
        with col_rl1:
            st.info("""
                    **State (Κατάσταση)**
                    
                    Η τρέχουσα θέση του agent.
                    
                    *Παράδειγμα:* Θέση σε ένα grid world
                    """)
        
        with col_rl2:
            st.success("""
                    **Action (Ενέργεια)**
                    
                    Κίνηση που μπορεί να κάνει ο agent.
                    
                    *Παράδειγμα:* Πάνω, Κάτω, Αριστερά, Δεξιά
                    """)
        
        with col_rl3:
            st.warning("""
                    **Reward (Ανταμοιβή)**
                    
                    Feedback για την ενέργεια.
                    
                    *Παράδειγμα:* +1 για στόχο, -1 για εμπόδιο
                    """)
        st.divider()
        
        st.markdown("### 📐 Q-Function και Bellman Equation:")
        st.latex(r"Q(s, a) = Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]")
        
        st.markdown("""
                    Όπου:
                    * $Q(s, a)$ = Η αναμενόμενη ανταμοιβή από την κατάσταση $s$ με ενέργεια $a$
                    * $\\alpha$ = Learning Rate (πόσο γρήγορα μαθαίνουμε)
                    * $\\gamma$ = Discount Factor (πόσο εκτιμούμε μελλοντικές ανταμοιβές)
                    * $r$ = Άμεση ανταμοιβή
                    * $s'$ = Νέα κατάσταση
                    """)
        
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            st.markdown("""
                        **Learning Rate (α):**
                        * α = 0: Δεν μαθαίνει τίποτα καινούργιο
                        * α = 1: Αντικαθιστά πλήρως την παλιά γνώση
                        * Τυπική τιμή: 0.1 - 0.5
                        """)
        
        with col_param2:
            st.markdown("""
                        **Discount Factor (γ):**
                        * γ = 0: Μόνο άμεσες ανταμοιβές
                        * γ = 1: Μελλοντικές ανταμοιβές πολύ σημαντικές
                        * Τυπική τιμή: 0.9 - 0.99
                        """)
    
    st.divider()
    
#   SIDEBAR: ENVIRONMENT & PARAMETERS
    st.sidebar.header("1. Grid World Environment")
    
    grid_size = st.sidebar.slider("Grid Size", 3, 8, 5, key = "rl_grid_size")
    
    goal_pos = (grid_size - 1, grid_size - 1)
    
    #   Starting position selector
    start_options = [(i, j) for i in range(grid_size) for j in range(grid_size) 
                     if (i, j) != goal_pos]
    
    default_start = (0, 0)
    selected_start = st.sidebar.selectbox(
        "Τοποθέτησε τον Agent",
        options = start_options,
        index = start_options.index(default_start) if default_start in start_options else 0,
        format_func = lambda x: f"Γραμμή {x[0]}, Στήλη {x[1]}",
        key = "rl_start"
    )
    
    #   List of all possible coordinates for obstacles (excluding start and goal)
    available_coords = [(i, j) for i in range(grid_size) for j in range(grid_size) 
                        if (i, j) != selected_start and (i, j) != goal_pos]
    
    #   Default obstacle in the center of the grid (if available)
    default_obstacle = (grid_size // 2, grid_size // 2)
    if default_obstacle not in available_coords:
        default_obstacle = available_coords[0] if available_coords else None

    #   Save selected obstacles in session state to persist across interactions
    if 'saved_obstacles' not in st.session_state:
        st.session_state.saved_obstacles = [default_obstacle] if default_obstacle else []
        
    #   Filter saved obstacles to only include those that are still valid (in case grid size or start position changed)
    valid_obstacles = [obs for obs in st.session_state.saved_obstacles if obs in available_coords]

    #   Multiselect for obstacles
    selected_obstacles = st.sidebar.multiselect(
        "Τοποθέτησε Εμπόδια",
        options = available_coords,
        default = valid_obstacles,
        format_func = lambda x: f"Γραμμή {x[0]}, Στήλη {x[1]}"
    )
    
    #   Update session state with the currently selected obstacles
    st.session_state.saved_obstacles = selected_obstacles
    
    st.sidebar.markdown("**Στόχος (🟢):** +1 reward")
    st.sidebar.markdown("**Εμπόδια (🔴):** -1 reward")
    st.sidebar.markdown("**Κενό:** -0.01 reward (να βρει γρήγορα)")
    
    st.sidebar.divider()
    
    #   Q-LEARNING PARAMETERS
    st.sidebar.header("2. Παράμετροι Q-Learning")
    alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01, key = "rl_alpha")
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.9, 0.01, key = "rl_gamma")
    epsilon_start = st.sidebar.slider("Initial Epsilon (Exploration)", 0.1, 1.0, 1.0, 0.05, key = "rl_epsilon")
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, 0.995, 0.001, key = "rl_decay")
    
    st.sidebar.markdown("""
                        **Epsilon-Greedy Policy:**
                        * ε πιθανότητα → τυχαία ενέργεια (exploration)
                        * 1-ε πιθανότητα → καλύτερη ενέργεια (exploitation)
                        """)
    
    #   ENVIRONMENT SETUP & RESET LOGIC
    current_env_config = {'grid': grid_size, 'obs': sorted(selected_obstacles), 'start': selected_start}
    
    #   If the environment configuration has changed, reset the environment and Q-table
    if 'rl_env_config' not in st.session_state or st.session_state.rl_env_config != current_env_config:
        st.session_state.rl_env_config = current_env_config
        st.session_state.rl_env = {
            'goal': goal_pos,
            'obstacles': selected_obstacles,
            'start': selected_start
        }
        st.session_state.agent_pos = selected_start
        st.session_state.q_table = np.zeros((grid_size, grid_size, 4))  #   4 actions
    
    #   Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
    actions, n_actions = ['↑', '↓', '←', '→'], 4
    
#   GRID WORLD VISUALIZATION
    st.header("🎮 Grid World Environment")
    
    col_env1, col_env2 = st.columns([2, 1])
    
    with col_env1:
        fig_grid, ax_grid = plt.subplots(figsize = (8, 8))
        
        #   Call the draw_base_grid function to render the grid, goal, and obstacle
        draw_base_grid(ax_grid, grid_size, st.session_state.rl_env)
        
        #   Draw agent
        if 'agent_pos' not in st.session_state:
            st.session_state.agent_pos = st.session_state.rl_env['start']
        
        agent_pos = st.session_state.agent_pos
        ax_grid.text(agent_pos[1] + 0.5, grid_size - 1 - agent_pos[0] + 0.5, 'A', fontsize = 30, ha = 'center', va = 'center', fontweight = 'bold', color = 'darkblue')
        
        ax_grid.set_title('Grid World Environment')
        st.pyplot(fig_grid)
        plt.close(fig_grid)
    
    with col_env2:
        st.markdown("### 📊 Legend")
        st.markdown("**A** | **Agent** (Πράκτορας)")
        st.markdown("**G** | **Goal** (+1 reward)")
        st.markdown("**X** | **Obstacle** (-1 reward)")
        
        st.divider()
        st.markdown("### 🎯 Actions")
        st.markdown("↑ Up | ↓ Down")
        st.markdown("← Left | → Right")
        
        st.divider()
        st.markdown("### 📍 Current State")
        st.info(f"Position: {st.session_state.agent_pos}")
    
#   TRAINING
    st.divider()
    st.header("🎓 Q-Learning Training")
    
    col_train1, col_train2 = st.columns([1, 2])
    
    with col_train1:
        n_episodes = st.slider("Επεισόδια Εκπαίδευσης", 10, 1000, 100, key = "rl_episodes")
        
        if st.button("🚀 Ξεκίνα Εκπαίδευση", type = "primary"):
            #   Reset Q-table
            st.session_state.q_table = np.zeros((grid_size, grid_size, n_actions))
            q_table = st.session_state.q_table
            
            #   Training loop
            epsilon, rewards_history, steps_history = epsilon_start, [], []
            
            progress_bar, status_text = st.progress(0), st.empty()
            
            for episode in range(n_episodes):
                state = st.session_state.rl_env['start']
                total_reward, steps, done = 0, 0, False
                
                while not done and steps < 100:
                    #   Epsilon-greedy action selection
                    if np.random.random() < epsilon:
                        action = np.random.randint(n_actions)
                    else:
                        action = np.argmax(q_table[state[0], state[1]])
                    
                    #   Take action
                    new_state = list(state)
                    hit_wall = False

                    if action == 0:     #   Up
                        if state[0] == 0: hit_wall = True
                        new_state[0] = max(0, state[0] - 1)
                    elif action == 1:   #   Down
                        if state[0] == grid_size - 1: hit_wall = True
                        new_state[0] = min(grid_size - 1, state[0] + 1)
                    elif action == 2:   #   Left
                        if state[1] == 0: hit_wall = True
                        new_state[1] = max(0, state[1] - 1)
                    elif action == 3:   #   Right
                        if state[1] == grid_size - 1: hit_wall = True
                        new_state[1] = min(grid_size - 1, state[1] + 1)
                    
                    new_state = tuple(new_state)
                    
                    #   Get reward
                    if new_state == st.session_state.rl_env['goal']:
                        reward = 1.0
                        done = True
                    elif new_state in st.session_state.rl_env['obstacles']:
                        reward = -1.0
                        done = True
                    elif hit_wall:
                        reward = -0.1   #   Penalty for hitting wall
                    else:
                        reward = -0.01  #   Small penalty for each step
                    
                    #   Q-learning update
                    current_q = q_table[state[0], state[1], action]
                    max_future_q = np.max(q_table[new_state[0], new_state[1]])
                    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
                    q_table[state[0], state[1], action] = new_q
                    
                    state = new_state
                    total_reward += reward
                    steps += 1
                
                rewards_history.append(total_reward)
                steps_history.append(steps)
                
                #   Decay epsilon
                epsilon = max(0.01, epsilon * epsilon_decay)
                
                #   Update progress
                progress_bar.progress((episode + 1) / n_episodes)
                status_text.text(f"Episode {episode + 1}/{n_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")
            
            st.session_state.q_table = q_table
            st.session_state.rewards_history = rewards_history
            st.session_state.steps_history = steps_history
            st.success(f"✅ Εκπαίδευση ολοκληρώθηκε! Τελικό ε: {epsilon:.4f}")
            st.info("📊 Κάνε scroll κάτω για να δεις το ενημερωμένο Q-Table και την Learned Policy!")
    
    with col_train2:
        #   Plot training progress
        if 'rewards_history' in st.session_state:
            fig_train, axes_train = plt.subplots(1, 2, figsize = (12, 4))
            
            #   Rewards
            axes_train[0].plot(st.session_state.rewards_history, 'b-', alpha = 0.6)

            #   Moving average
            window = min(20, len(st.session_state.rewards_history))
            if window > 1:
                moving_avg = np.convolve(st.session_state.rewards_history, np.ones(window)/window, mode = 'valid')
                axes_train[0].plot(range(window-1, len(st.session_state.rewards_history)), moving_avg, 'r-', linewidth = 2, label = 'Moving Avg')

            axes_train[0].set_xlabel('Episode')
            axes_train[0].set_ylabel('Total Reward')
            axes_train[0].set_title('Training Rewards')
            axes_train[0].grid(True, alpha = 0.3)
            axes_train[0].legend()
            
            #   Steps
            axes_train[1].plot(st.session_state.steps_history, 'g-', alpha = 0.6)
            if window > 1:
                moving_avg_steps = np.convolve(st.session_state.steps_history, np.ones(window)/window, mode = 'valid')
                axes_train[1].plot(range(window-1, len(st.session_state.steps_history)), moving_avg_steps, 'r-', linewidth = 2, label = 'Moving Avg')
            axes_train[1].set_xlabel('Episode')
            axes_train[1].set_ylabel('Steps to Goal')
            axes_train[1].set_title('Steps per Episode')
            axes_train[1].grid(True, alpha = 0.3)
            axes_train[1].legend()
            
            plt.tight_layout()
            st.pyplot(fig_train)
            plt.close(fig_train)
        else:
            st.info("Πάτησε το κουμπί για να ξεκινήσεις την εκπαίδευση!")
    
#   Q-TABLE VISUALIZATION
    st.divider()
    st.header("📊 Q-Table Visualization")
    
    st.markdown("""
                Το **Q-Table** αποθηκεύει την αναμενόμενη ανταμοιβή για κάθε (state, action) pair.
                
                Πράσινο = Υψηλή Q-value (καλή ενέργεια), Κόκκινο = Χαμηλή Q-value
                """)
    
    #   Create Q-table heatmap
    fig_q, axes_q = plt.subplots(2, 2, figsize = (12, 10))
    q_table = st.session_state.q_table
    
    #   Find global min/max for consistent colormap
    q_min, q_max = q_table.min(), q_table.max()
    if q_min == q_max:
        q_min, q_max = -1, 1
    
    #   Plot each action heatmap
    for idx, (action_name, ax) in enumerate(zip(actions, axes_q.flatten())):
        q_values = q_table[:, :, idx]

        sns.heatmap(
            q_values, 
            annot = True,
            fmt = ".2f",
            cmap = 'RdYlGn', 
            vmin = q_min, 
            vmax = q_max, 
            ax = ax, 
            cbar = True,
            square = True
        )
    
        ax.set_title(f'Action: {action_name}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    st.pyplot(fig_q)
    plt.close(fig_q)
    
#   LEARNED POLICY VISUALIZATION
    st.divider()
    st.header("🎯 Learned Policy")
    st.markdown("""Οι βέλτιστες ενέργειες για κάθε κατάσταση σύμφωνα με το εκπαιδευμένο Q-table:""")
    
    fig_policy, ax_policy = plt.subplots(figsize = (8, 8))
    
    #   Call the draw_base_grid function to render the grid, goal, and obstacle
    draw_base_grid(ax_policy, grid_size, st.session_state.rl_env)
    
    #   Draw arrows for best action in each state
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) != st.session_state.rl_env['goal'] and (i, j) not in st.session_state.rl_env['obstacles']:
                best_action = np.argmax(st.session_state.q_table[i, j])
                arrow_map = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
                dx, dy = arrow_map[best_action]
                ax_policy.arrow(j + 0.5 - dx, grid_size - 1 - i + 0.5 - dy, dx * 2, dy * 2, head_width = 0.15, head_length = 0.1, fc = 'blue', ec = 'blue')
    
    ax_policy.set_title('Learned Policy (Best Action per State)')
    st.pyplot(fig_policy)
    plt.close(fig_policy)
    
#   INTERACTIVE DEMO
    st.divider()
    st.header("🎮 Interactive Demo")
    st.markdown("""Δες τον agent να ακολουθεί την learned policy:""")
    
    if st.button("▶️ Run Demo Episode"):
        state = st.session_state.rl_env['start']
        path, done, steps = [state], False, 0
        
        while not done and steps < 50:
            best_action = np.argmax(st.session_state.q_table[state[0], state[1]])
            
            new_state = list(state)
            if best_action == 0:
                new_state[0] = max(0, state[0] - 1)
            elif best_action == 1:
                new_state[0] = min(grid_size - 1, state[0] + 1)
            elif best_action == 2:
                new_state[1] = max(0, state[1] - 1)
            elif best_action == 3:
                new_state[1] = min(grid_size - 1, state[1] + 1)
            
            new_state = tuple(new_state)
            path.append(new_state)
            
            if new_state == st.session_state.rl_env['goal']:
                done = True
                st.success(f"🏆 Ο agent έφτασε στον στόχο σε {len(path)-1} βήματα!")
            elif new_state in st.session_state.rl_env['obstacles']:
                done = True
                st.error("💀 Ο agent χτύπησε σε εμπόδιο!")
            
            state = new_state
            steps += 1
        
        #   Visualize path
        fig_path, ax_path = plt.subplots(figsize = (8, 8))
        
        #   Call the draw_base_grid function to render the grid, goal, and obstacle
        draw_base_grid(ax_path, grid_size, st.session_state.rl_env)
        
        #   Draw agent path
        for idx, pos in enumerate(path):
            if idx > 0:
                prev = path[idx - 1]
                ax_path.plot([prev[1] + 0.5, pos[1] + 0.5], [grid_size - 1 - prev[0] + 0.5, grid_size - 1 - pos[0] + 0.5], 'b-', linewidth = 2)
            ax_path.text(pos[1] + 0.5, grid_size - 1 - pos[0] + 0.5, str(idx), fontsize = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'circle', facecolor = 'yellow', alpha = 0.7))
        
        ax_path.set_title(f'Agent Path ({len(path)-1} steps)')
        st.pyplot(fig_path)
        plt.close(fig_path)
    
#   EXPLORATION VS EXPLOITATION
    with st.expander("📊 Exploration vs Exploitation", expanded = False):
        st.markdown("""
                    ### Η Διαλεκτική της Μάθησης
                    
                    Το Q-Learning πρέπει να ισορροπεί μεταξύ δύο στρατηγικών:
                    """)
        
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            st.info("""
                    **🔍 Exploration (Εξερεύνηση)**
                    
                    Δοκιμάζουμε τυχαίες ενέργειες για να ανακαλύψουμε νέους δρόμους.
                    
                    *Χρήσιμο στην αρχή της εκπαίδευσης*
                    """)
        
        with col_ex2:
            st.success("""
                        **💡 Exploitation (Εκμετάλλευση)**
                        
                        Χρησιμοποιούμε τη γνώση που έχουμε για να πάρουμε την καλύτερη απόφαση.
                        
                        *Χρήσιμο στο τέλος της εκπαίδευσης*
                        """)
        
        st.markdown("""
                    **Epsilon-Greedy Policy:**
                    * Με πιθανότητα ε → τυχαία ενέργεια (exploration)
                    * Με πιθανότητα 1-ε → καλύτερη ενέργεια βάσει Q-table (exploitation)
                    * Το ε μειώνεται με την πάροδο της εκπαίδευσης
                    """)
        
        #   Visualize epsilon decay
        epsilons = [epsilon_start * (epsilon_decay ** i) for i in range(200)]
        
        fig_eps, ax_eps = plt.subplots(figsize = (8, 4))
        ax_eps.plot(epsilons, 'b-', linewidth = 2)
        ax_eps.axhline(y = 0.01, color = 'r', linestyle = '--', label = 'Min ε')
        ax_eps.set_xlabel('Episode')
        ax_eps.set_ylabel('Epsilon')
        ax_eps.set_title('Epsilon Decay over Training')
        ax_eps.legend()
        ax_eps.grid(True, alpha = 0.3)
        st.pyplot(fig_eps)
        plt.close(fig_eps)
    
#   QUIZ
    render_quiz("rl")
    
#   CONCLUSION
    render_conclusion("rl")