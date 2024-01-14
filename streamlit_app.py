import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import requests
import json
from streamlit_extras.let_it_rain import rain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tfidf_novelty import get_tfidf_novelty
import openai
import re
import time

# Initialize session state variables
if 'api_response' not in st.session_state:
    st.session_state.api_response = None
if 'display_commercial_analysis' not in st.session_state:
    st.session_state.display_commercial_analysis = False

# testing with mock json data to save money
with open('mock_data.json', 'r') as file:
    mock_json = json.load(file)

# loading system prompt to make the code cooler
with open('system_prompt.txt', 'r') as file:
    system_prompt = file.read()

# Sidebar for a cute earth icon and for loading user's api key
with st.sidebar:
    st.image("earth.png", width=250)
    # For example, links or additional instructions
    api_key = st.text_input("Enter your API key", type="password")
    st.write("We require a GPT-4 Turbo API key, specifically the model gpt-4-1106-preview. Please note that usage may incur charges.")
    
    
def emoji():
    rain(
        emoji="🌏",
        font_size=30,
        falling_speed=10,
        animation_length="infinite",
    )

# for fun, but we commented it out
# emoji()

# st.title('🌏 Earth Hack')

# metrics list for sustainability evaluation
# [To-Do] Fix the format
metrics = [
    "1_No_Poverty", "2_Zero_Hunger", "3_Good_Health_and_Well-being",
    "4_Quality_Education", "5_Gender_Equality", "6_Clean_Water_and_Sanitation",
    "7_Affordable_and_Clean_Energy", "8_Decent_Work_and_Economic_Growth",
    "9_Industry_Innovation_and_Infrastructure", "10_Reduced_Inequality",
    "11_Sustainable_Cities_and_Communities", "12_Responsible_Consumption_and_Production",
    "13_Climate_Action", "14_Life_Below_Water", "15_Life_on_Land",
    "16_Peace_Justice_and_Strong_Institutions", "17_Partnerships_for_the_Goals"
]    

def format_metric_name(metric):
    # Split the string at the underscore and return the part after it
    return metric.split('_', 1)[1].replace('_', ' ')

formatted_metrics = [format_metric_name(metric) for metric in metrics]

# Emoji function based on score for table
def score_to_emoji(score):
    if score <= 4:
        return "🔴"  # Red circle for low scores
    elif 5 <= score <= 6:
        return "🟡"  # Yellow circle for medium scores
    else:
        return "🟢"  # Green circle for high scores


# Function to call GPT API and process the response
def evaluate_idea(problem, solution):
    # return mock_json

    # Pass the apikey to the OpenAI library
    openai.api_key = api_key

    # try:
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Problem:{problem}\n\nSolution:{solution}"}
            ],
        max_tokens= 4096,#128000,
    )
        
    response = response["choices"][0]["message"]["content"]
    # ai_response = re.sub(r'^```json\n\n|\n```$', '', response)
    
    ai_response = re.sub(r'`|json', '', response)
    
    # st.markdown("## OUTPUT 0 Response:")
    # st.write(ai_response)
                
    output = json.loads(ai_response)
        
        # st.markdown("## OUTPUT 1 Response:")
        # st.markdown(output)
    
    return output
    
    # except openai.error.InvalidRequestError as e:
    #     if e.status_code == 401:  # Unauthorized, typically due to invalid API key
    #         st.error("Invalid API key. Please check your API key and try again.")
    #     else:
    #         st.error(f"An error occurred: {e}")
    #     return None
    


# Section for displaying evaluation result


### pretty interface for user prompt

st.title("💡 Business Idea Evaluation")

# App Introduction
with st.expander("Introduction of Our App", expanded=True):
    st.write("""
        This Streamlit application is designed to evaluate business ideas based on a problem-solution approach, with a focus on assessing their sustainability. It generates scores for novelty and sustainability aspects of the idea, helping users understand the potential impact and uniqueness of their business concepts.
    """)

# Option for user to choose input method
input_method = st.radio("Choose your input method:", ('Manual Input', 'Upload CSV'))

# Placeholder text for 'Problem' and 'Solution' inputs
problem_placeholder = "More than 130 billion plastic bottles waste annually in Egypt"
solution_placeholder = "Bariq factory to recycle plastic bottles"

if input_method == 'Manual Input':
    with st.form("business_idea_form"):
        problem = st.text_area("Problem:", value="", placeholder=problem_placeholder)
        solution = st.text_area("Solution:", value="", placeholder=solution_placeholder)
        submit_button = st.form_submit_button("Evaluate Idea")

    # if sumbmitted, send the prompt to openai to rob ~0.35$ from the user
    if submit_button:
        
        if not api_key:
            st.error("Please enter an API key.")
        else:
            # get the response
            
            # progress_bar = st.progress(0)
            
            with st.spinner('Evaluating your idea, please wait...'):
    
                # for i in range(100):
                #     progress_bar.progress(i+1)
                #     time.sleep(1)
                    
                try:
                    st.session_state.api_response = evaluate_idea(problem, solution)
                    # api_response = evaluate_idea(problem, solution)
                except openai.error.InvalidRequestError as e:
                    if e.status_code == 401:  # Unauthorized, typically due to invalid API key
                        st.error("Invalid API key. Please check your API key and try again.")
                    else:
                        st.error(f"An error occurred: {e}")
                else:
                    # This 'else' block runs only if no exception was raised
                    if not st.session_state.api_response:
                        # Display warning message if API call fails to retrieve data
                        st.error("Unable to retrieve data. Please try again later.")

                           
    # if api_response:
    if st.session_state.api_response:
        api_response = st.session_state.api_response
        # Display if the idea is sustainability related with highlighting
        is_sustainable = api_response['Idea_Sustainability_Related'] == True
        
        sustainability_comment = api_response['Idea_Sustainability_Related_Comment']
        st.markdown(f"<h3 style='color:blue;'>Is the Idea Sustainability Related? {'Yes' if is_sustainable else 'No'}</h3>", unsafe_allow_html=True)
        st.write("### Sustainability Analysis:")
        st.write(sustainability_comment)

        # if the idea is not sustainable, proceed with displaying the scores.
        if is_sustainable:
            scores = [int(api_response['Evaluation']['SDG_Scores'][metric]['Score']) for metric in metrics]
            comments = [api_response['Evaluation']['SDG_Scores'][metric]['Comment'] for metric in metrics]
            total_score = int(api_response['Evaluation']['Total_Score'])  # Convert to integer
            analysis_context = api_response['Evaluation']['Summary']
            novelty_score = int(api_response['Evaluation']['Novelty_Score'])  # Convert to integer
            novelty_comment = api_response['Evaluation']['Novelty_Evaluation']['Comment']
            
            
            # Displaying novelty score and analysis with highlighting
            st.markdown(f"<h3 style='color:purple;'>Novelty Score: {novelty_score} / 100</h3>", unsafe_allow_html=True)
            st.write("### Novelty Analysis:")
            st.write(novelty_comment)

            
            # get tfidf novelty score with user-provided problem and solution
            max_similarity, tfidf_is_novelty = get_tfidf_novelty(problem, solution)
            
            # Check if the response is novelty or not
            novelty_status = "Yes" if tfidf_is_novelty else "No"
            
            # Display the novelty similarity score with decimals and novelty status
            st.markdown(f"<h3 style='color:orange;'>Novelty Similarity Score: {max_similarity:.2f} / 1</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:orange;'>Is it Novelty? {novelty_status}</h3>", unsafe_allow_html=True)

            with st.expander("Understanding the Novelty Similarity Score"):
                st.write("""
                    - The Novelty Similarity Score indicates the degree of similarity between your idea and existing ideas within the AI EarthHack Dataset.
                    - A higher score suggests that your idea is more closely related to concepts already present in the dataset.
                    - Conversely, a lower score suggests that your idea is unique or novel in the context of this dataset.
                    - Ideas with a score below a certain threshold are considered novel, highlighting their uniqueness compared to the dataset. This serves as a complementary assessment alongside the GPT-4-Turbo engine's analysis.
                """)

            # Display the summary score without decimals
            st.markdown(f"<h3 style='color:green;'>Summary Score: {total_score} / 100</h3>", unsafe_allow_html=True)
            st.write("### Summary Analysis:")
            st.write(analysis_context)

            
            # Modify DataFrame to include comments
            score_df = pd.DataFrame({
                'Metric': formatted_metrics,
                'Score': scores,
                'Comment': comments,
                'Level': [score_to_emoji(score) for score in scores]
            })

            # col1, col2 = st.columns(2)  # Creates two columns
            
            # Radar chart
            # with col1: 
            st.write("### Radar Chart Evaluation Results:")
            num_vars = len(formatted_metrics)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # Complete the loop

            # Use 'scores' directly as it is already a list
            scores_list = scores + scores[:1]  # Repeat the first score to close the radar chart

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, scores_list, color='green', alpha=0.25)
            ax.plot(angles, scores_list, color='green', linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(formatted_metrics)
        
            st.pyplot(fig)

            # with col2:
            # Seaborn barplot
            st.write("### Bar Chart Evaluation Results:")
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Score', y='Metric', data=score_df, palette="vlag")
            plt.xlabel('Score out of 10')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
                
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
            

    
            # # Slider section
            # st.write("### Evaluation Results:")
            # for metric, score in zip(formatted_metrics, scores):
            #     st.slider(metric, 0, 10, score, disabled=True)


        
## Simulate scores for demonstration (replace with real data later)
# scores = np.random.randint(1, 11, size=len(metrics))

## Calculate the summary score
# total_score = sum(scores)

    # else:
    #     # Display warning message if API call fails
    #     st.error("Unable to retrieve data. Please try again later.")
            
            # Display a section for commercial analysis
        st.markdown("---")
        st.markdown("### **Do you want further commercial analysis?💸**")
        st.markdown("*Note: This will incur additional costs with the use of the API key.*")
    
        # Button for commercial analysis
        commercial_analysis_button = st.button("Display Commercial Analysis")
    
        # Mock commercial analysis process (commented out, use for real implementation)
        if commercial_analysis_button:
            st.session_state.display_commercial_analysis = True            

    if st.session_state.display_commercial_analysis:
        with st.spinner('Processing commercial analysis, please wait...'):
            # For demonstration, using mock data
            st.write("### Commercial Analysis Response:")
            st.markdown("*This is a mock response for demonstration purposes.*")
            st.write("Imagine this text is the detailed commercial analysis provided by the AI.")

            # Uncomment and modify the following lines for actual implementation
            # with open('commercial_prompt.txt', 'r') as file:
            #     commercial_prompt = file.read()
            # response = openai.ChatCompletion.create(
            #     model="gpt-4-1106-preview",
            #     messages=[{"role": "system", "content": commercial_prompt},
            #               {"role": "user", "content": f"Problem:{problem}\n\nSolution:{solution}"}
            #     ],
            #     max_tokens=4096,
            # )
            # commercial_response = response["choices"][0]["message"]["content"]
            # st.write(commercial_response)

        # progress_bar.empty()



elif input_method == 'Upload CSV':

    with st.expander("CSV Upload Instructions", expanded=True):
        st.write("""
            Please follow these guidelines for your CSV file:
            - Your CSV file should contain only two columns: 'problem' and 'solution'.
            - If your file has headers, ensure the column titles are 'problem' and 'solution'.
            - The file should be encoded in UTF-8 or ISO-8859-1.
            - Each row in the file should represent a separate problem-solution pair.
        """)
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            # Try reading with default encoding (UTF-8)
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            # If UnicodeDecodeError, try reading with alternative encoding
            uploaded_file.seek(0)  # Reset file pointer to the beginning
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        # Convert column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Check if 'id' column exists, if not add it
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1, 1 + len(df)))

        # Check for necessary columns 'problem' and 'solution'
        if 'problem' in df.columns and 'solution' in df.columns:
            # Display the DataFrame
            st.write("Uploaded Data:")
            st.dataframe(df)
        else:
            st.error("The CSV file must contain 'problem' and 'solution' columns.")



