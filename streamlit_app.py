import streamlit as st
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('🌏 Earth Hack')

st.write('Hello world!')

# Sidebar for API key and system prompt input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

    st.title("🔧 System Prompt")
    system_prompt = st.text_area("Enter system prompt here:")
    if st.button("Submit System Prompt"):
        if system_prompt:
            st.session_state["system_prompt_result"] = "Processed prompt: " + system_prompt
        else:
            st.session_state["system_prompt_result"] = "No prompt entered."

# Display processed system prompt result
if "system_prompt_result" in st.session_state:
    st.write(st.session_state["system_prompt_result"])

# Section for business idea input
st.title("💡 Business Idea Evaluation")
with st.form("business_idea_form"):
    problem = st.text_area("Problem:")
    solution = st.text_area("Solution:")
    submit_button = st.form_submit_button("Evaluate Idea")

# Temporary metrics for sustainability evaluation
metrics = [
    "Carbon Footprint Reduction", "Water Usage Efficiency", "Recyclability", 
    "Energy Efficiency", "Resource Conservation", "Pollution Reduction", 
    "Biodiversity Preservation", "Social Impact", "Economic Viability", 
    "Innovation and Scalability"
]

# Mock Visualization (This will be replaced with real data later)
if submit_button:
    # Mock scores (Replace with real scores from the API response in future)
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

# Chatbot
st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Enter your idea and I'll evaluate it :)"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # This part will be enabled when you have the OpenAI API key
    # openai.api_key = openai_api_key
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # st.chat_message("user").write(prompt)
    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    # msg = response.choices[0].message
    # st.session_state.messages.append(msg)
    # st.chat_message("assistant").write(msg.content)
