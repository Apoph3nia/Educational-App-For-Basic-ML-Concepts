import streamlit as st, matplotlib.pyplot as plt

def apply_global_styles():
    """
    Apply global CSS styles to all Streamlit pages.
    """
    st.markdown("""
        <style>
        /* 1. Center text in success/info/warning boxes */
        .stAlert {
            text-align: center;
        }
        
        /* 2. Hide Fullscreen button (st.image) */
        button[title="View fullscreen"] {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* 3. Custom styling for timeline */
        .timeline-item {
            padding: 10px;
            border-left: 3px solid #FF4B4B;
            margin: 10px 0;
            background-color: #f0f2f6;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html = True)
    
def draw_base_grid(ax, grid_size, env):
        """Draws the base grid, goal, and obstacle on a given Axes."""
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == env['goal']:
                    color = 'lightgreen'
                elif (i, j) in env['obstacles']:
                    color = 'lightcoral'
                else:
                    color = 'white'
                
                rect = plt.Rectangle((j, grid_size - 1 - i), 1, 1, facecolor = color, edgecolor = 'black', linewidth = 2)
                ax.add_patch(rect)
        
        #   Goal and Obstacle labels
        goal = env['goal']
        ax.text(goal[1] + 0.5, grid_size - 1 - goal[0] + 0.5, 'G', fontsize = 30, ha = 'center', va = 'center', fontweight = 'bold', color = 'darkgreen')
        
        for obs in env['obstacles']:
            ax.text(obs[1] + 0.5, grid_size - 1 - obs[0] + 0.5, 'X', fontsize = 30, ha = 'center', va = 'center', fontweight = 'bold', color = 'darkred')
        
        #   Limit axes and hide ticks
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.axis('off')