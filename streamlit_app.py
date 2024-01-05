import streamlit as st
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('🌏 Earth Hack')

st.write('Hello world!')

# Sidebar for API key input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

# Temporary metrics for sustainability evaluation
metrics = [
    "Carbon Footprint Reduction", "Water Usage Efficiency", "Recyclability", 
    "Energy Efficiency", "Resource Conservation", "Pollution Reduction", 
    "Biodiversity Preservation", "Social Impact", "Economic Viability", 
    "Innovation and Scalability"
]

# Chatbot
st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Enter your business idea (problem and solution) to evaluate its sustainability."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Here we will assume prompt is the business idea
    # For now, we'll just generate random scores for demonstration purposes
    # In the future, replace this with actual API call and logic
    scores = np.random.randint(1, 11, size=len(metrics))
    df = pd.DataFrame({
        'Metric': metrics,
        'Score': scores
    })
    df = df.set_index('Metric')

    # Plotting
    fig, ax = plt.subplots()
    df.plot(kind='barh', ax=ax, legend=False)
    ax.set_xlabel('Score out of 10')
    st.pyplot(fig)

    # Add the user's prompt to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
