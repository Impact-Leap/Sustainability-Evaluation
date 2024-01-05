import streamlit as st
import numpy as np

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

# Displaying the results as non-interactive sliders
if submit_button:
    # Simulate scores for demonstration (replace with real data later)
    scores = np.random.randint(1, 11, size=len(metrics))

    st.write("Evaluation Results:")
    for metric, score in zip(metrics, scores):
        st.slider(metric, 0, 10, score, disabled=True)

# Sidebar for additional options or information
with st.sidebar:
    st.write("Add any sidebar content here")
    # For example, links or additional instructions
