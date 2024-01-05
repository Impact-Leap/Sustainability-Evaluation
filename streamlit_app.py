import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from streamlit_extras.app_logo import add_logo

add_logo("earth.png", height=300)

st.title('🌏 Earth Hack')

# Temporary metrics for sustainability evaluation
metrics = [
    "Carbon Footprint Reduction", "Water Usage Efficiency", "Recyclability", 
    "Energy Efficiency", "Resource Conservation", "Pollution Reduction", 
    "Biodiversity Preservation", "Social Impact", "Economic Viability", 
    "Innovation and Scalability"
]

# Emoji function based on score
def score_to_emoji(score):
    if score <= 4:
        return "🔴"  # Red circle for low scores
    elif 5 <= score <= 6:
        return "🟡"  # Yellow circle for medium scores
    else:
        return "🟢"  # Green circle for high scores

# Section for business idea input
st.title("💡 Business Idea Evaluation")
with st.form("business_idea_form"):
    problem = st.text_area("Problem:")
    solution = st.text_area("Solution:")
    submit_button = st.form_submit_button("Evaluate Idea")

if submit_button:
    # Simulate scores for demonstration (replace with real data later)
    scores = np.random.randint(1, 11, size=len(metrics))

    # Create DataFrame for scores and emojis
    score_df = pd.DataFrame({
        'Metric': metrics,
        'Score': scores,
        'Level': [score_to_emoji(score) for score in scores]
    })

    # Displaying scores as a styled table using markdown
    st.markdown("### Evaluation Table")
    st.markdown(
        score_df.to_html(index=False, escape=False, justify='center', classes='table'),
        unsafe_allow_html=True
    )

    # Apply custom CSS for table styling
    st.markdown("""
        <style>
            .table {width: 100%; margin-left: auto; margin-right: auto; border-collapse: collapse;}
            .table td, .table th {border: none;}
            th {text-align: center; font-size: 18px; font-weight: bold;}
            td {text-align: center;}
        </style>
        """, unsafe_allow_html=True)

    # Slider section
    st.write("### Evaluation Results:")
    for metric, score in zip(metrics, scores):
        st.slider(metric, 0, 10, score, disabled=True)

    # Radar chart
    st.write("### Radar Chart Evaluation Results:")
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    scores_list = scores.tolist()
    scores_list += scores_list[:1]  # Repeat the first score to close the radar chart

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores_list, color='green', alpha=0.25)
    ax.plot(angles, scores_list, color='green', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    st.pyplot(fig)
    
    # Seaborn barplot
    st.write("### Bar Chart Evaluation Results:")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Metric', data=score_df, palette="vlag")
    plt.xlabel('Score out of 10')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Sidebar for additional options or information
with st.sidebar:
    add_logo("earth.png", height=300)
    # For example, links or additional instructions
