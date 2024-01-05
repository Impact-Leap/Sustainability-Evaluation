import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # Displaying scores as a styled table
    st.markdown("### Evaluation Table")
    st.markdown(
        score_df.style.hide_index()
        .set_table_attributes('style="width:100%;" class="table"')
        .set_properties(subset=['Score', 'Level'], **{'text-align': 'center'})
        .set_properties(**{'font-size': '16px', 'font-weight': 'bold'})
        .render(), unsafe_allow_html=True)

    # Slider section
    st.write("### Evaluation Results:")
    for metric, score in zip(metrics, scores):
        st.slider(metric, 0, 10, score, disabled=True)

    # Bar chart section
    st.write("### Visualized Evaluation Results:")
    plt.barh(score_df['Metric'], score_df['Score'], color=score_df['Level'].replace({"🔴": "red", "🟡": "yellow", "🟢": "green"}))
    plt.xlabel('Score out of 10')
    st.pyplot()

# Sidebar for additional options or information
with st.sidebar:
    st.write("Add any sidebar content here")
    # For example, links or additional instructions
