import streamlit as st
import numpy as np
import pandas as pd

st.title('🌏 Earth Hack')

# Temporary metrics for sustainability evaluation
metrics = [
    "Carbon Footprint Reduction", "Water Usage Efficiency", "Recyclability", 
    "Energy Efficiency", "Resource Conservation", "Pollution Reduction", 
    "Biodiversity Preservation", "Social Impact", "Economic Viability", 
    "Innovation and Scalability"
]

# Section for business idea input
st.title("💡 Business Idea Evaluation")
with st.form("business_idea_form"):
    problem = st.text_area("Problem:")
    solution = st.text_area("Solution:")
    submit_button = st.form_submit_button("Evaluate Idea")

if submit_button:
    # Simulate scores for demonstration (replace with real data later)
    scores = np.random.randint(1, 11, size=len(metrics))
    df = pd.DataFrame({
        'Metric': metrics,
        'Score': scores
    })
    df = df.set_index('Metric')

    # Layout for sliders and bar chart
    col1, col2 = st.columns(2)

    with col1:
        st.write("Evaluation Results:")
        for metric, score in zip(metrics, scores):
            # Color coding (simple text representation)
            color = "red" if score <= 4 else "yellow" if 5 <= score <= 6 else "green"
            st.markdown(f"<p style='color: {color};'>{metric}: {score}</p>", unsafe_allow_html=True)

    with col2:
        # Use Streamlit's native bar_chart for visualization
        st.bar_chart(df['Score'])

# Sidebar for additional options or information
with st.sidebar:
    st.write("Add any sidebar content here")
    # For example, links or additional instructions
