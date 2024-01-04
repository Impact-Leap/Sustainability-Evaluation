import streamlit as st
import openai

st.title('🌏 Earth Hack')

st.write('Hello world!')

# Sidebar for API key and system prompt input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    st.title("🔧 System Prompt")
    system_prompt = st.text_area("Enter system prompt here:")
    if st.button("Submit System Prompt"):
        # Example processing logic for system prompt
        if system_prompt:
            st.session_state["system_prompt_result"] = "Processed prompt: " + system_prompt
        else:
            st.session_state["system_prompt_result"] = "No prompt entered."

# Display processed system prompt result
if "system_prompt_result" in st.session_state:
    st.write(st.session_state["system_prompt_result"])

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

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
