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


# testing with mock json data to save money
with open('mock_data.json', 'r') as file:
    mock_json = json.load(file)

# loading system prompt to make the code cooler
with open('system_prompt.txt', 'r') as file:
    system_prompt = file.read()

# Sidebar for a cute earth icon and for loading user's api key
with st.sidebar:
    st.image("earth.png", width=300)
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

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Problem:\n{problem}\n\nSolution:\n{solution}"""}
            ],
        max_tokens= 4096,#128000,
    )
    
    ai_response = response["choices"][0]["message"]["content"][8:-4]

    # st.markdown("## OUTPUT 0 Response:")
    # st.json(ai_response)
            
    output = json.loads(ai_response)
    
    # st.markdown("## OUTPUT 1 Response:")
    # st.json(output)


    return output


# Section for displaying evaluation result

st.title("💡 Business Idea Evaluation")

# pretty interface for user prompt
with st.form("business_idea_form"):
    problem = st.text_area("Problem:")
    solution = st.text_area("Solution:")
    submit_button = st.form_submit_button("Evaluate Idea")

# if sumbmitted, send the prompt to openai to rob ~0.35$ from the user
if submit_button:
    if not api_key:
        st.error("Please enter an API key.")
    else:
        # get the response
        with st.spinner('Evaluating your idea, please wait...'):
            api_response = evaluate_idea(problem, solution)
        
        if api_response:
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

                with st.expander("Instructions"):
                    st.write("""
                        Higher similarity score indicates a stronger relation with some information in the dataset(AI EarthHack Dataset.csv).
                        A similarity score lower than a certain threshold will be identified as novel, meaning that this idea is novel in the scope of the provided dataset. This novelty identifier make the dataset a complementary data source in addition to the training data for the GPT4-Turbo engine.
                    """)


                
                # Display the summary score without decimals
                st.markdown(f"<h3 style='color:green;'>Summary Score: {total_score} / 100</h3>", unsafe_allow_html=True)
                st.write("### Summary Analysis:")
                st.write(analysis_context)

                
                # Modify DataFrame to include comments
                score_df = pd.DataFrame({
                    'Metric': metrics,
                    'Score': scores,
                    'Comment': comments,
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

                # Use 'scores' directly as it is already a list
                scores_list = scores + scores[:1]  # Repeat the first score to close the radar chart

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
            
    ## Simulate scores for demonstration (replace with real data later)
    # scores = np.random.randint(1, 11, size=len(metrics))

    ## Calculate the summary score
    # total_score = sum(scores)

        else:
            # Display warning message if API call fails
            st.error("Unable to retrieve data. Please try again later.")

