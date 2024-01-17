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
from commercial import get_top_5_tfidf, get_business_status_distribution, get_percentile_by_category, generate_commercial_analysis
from parallel_summary import chat_with_openai, process_inputs_in_parallel, processed_results_to_df

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
    st.markdown("_(**Please review your API key agreement to understand privacy implications before using our evaluator.**)_", unsafe_allow_html=True)

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
    

### pretty interface for user prompt

st.title("💡 Sustainability Idea Evaluation")

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
        st.session_state.display_commercial_analysis = False
        st.session_state.api_response = False
        if not api_key:
            st.error("Please enter an API key.")
        else:
            # get the response
            
            # progress_bar = st.progress(0)
            
            with st.spinner('Evaluating your idea, please wait for aproximately 40 seconds...'):
    
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
        # st.markdown(f"<h3 style='color:blue;'>Is the Idea Sustainability Related? {'Yes' if is_sustainable else 'No'}</h3>", unsafe_allow_html=True)
        if is_sustainable:
            st.markdown("<h3 style='color: green;'>We've evaluated this, and it's exciting news: This idea is sustainability-related!</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>After careful evaluation, it appears that this idea is not sustainability-related.</h3>", unsafe_allow_html=True)

        # st.write("### Sustainability Analysis:")
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
            st.markdown(f"<h3 style='color:orange;'>Similarity Score: {max_similarity:.2f} / 1</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:orange;'>Is it Novelty? {novelty_status}</h3>", unsafe_allow_html=True)

            with st.expander("Understanding the Novelty Similarity Score"):
                st.write("""
                    - The Novelty Similarity Score indicates the degree of similarity between your idea and existing ideas within the AI EarthHack Dataset.
                    - A higher score suggests that your idea is more closely related to concepts already present in the dataset.
                    - Conversely, a lower score suggests that your idea is unique or novel in the context of this dataset.
                    - Ideas with a score below a certain threshold are considered novel, highlighting their uniqueness compared to the dataset. This serves as a complementary assessment alongside the GPT-4-Turbo engine's analysis.
                """)

            # Display the summary score without decimals
            st.markdown(f"<h3 style='color:green;'>Summary Score: {total_score} / 170</h3>", unsafe_allow_html=True)
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
            # n_colors = score_df['Score'].nunique()
            # palette = sns.color_palette("flare", n_colors=n_colors)
            palette = sns.color_palette("husl", 9)
            green_color = sns.color_palette("Greens", 6)[-2]  # A shade of green

            # sns.barplot(x='Score', y='Metric', data=score_df, color=green_color)

            sns.barplot(x='Score', y='Metric', data=score_df, palette=palette)
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


    # else:
    #     # Display warning message if API call fails
    #     st.error("Unable to retrieve data. Please try again later.")
            
        # Display a section for commercial analysis
            st.markdown("---")
            st.markdown("### **Do you want further economical analysis?💸**")
            # st.markdown("*Note: This will incur additional costs with the use of the API key.*")
        
            # Button for commercial analysis
            commercial_analysis_button = st.button("Display Economical Analysis")
        
            # Mock commercial analysis process (commented out, use for real implementation)
            if commercial_analysis_button:
                st.session_state.display_commercial_analysis = True            
        
            if st.session_state.display_commercial_analysis:
                with st.spinner('Processing economical analysis, please wait...'):
          
                    # Read the file contents
                    documents = pd.read_csv('Cleaned_ValidationSet.csv', encoding='ISO-8859-1')
                    
                    top_5_similar_docs, avg_num_competitors, avg_total_raised = get_top_5_tfidf(problem, solution)
                    df_cat = get_business_status_distribution(top_5_similar_docs)
                
                    category_summary = df_cat.groupby('Category')['Percentage'].sum()
                
                    # Find the category with the highest total percentage
                    most_likely_category = category_summary.idxmax()
                    
                    # Group by 'BusinessStatus' and sum the 'Percentage'
                    business_status_summary = df_cat.groupby('BusinessStatus')['Percentage'].sum()
                    
                    # Find the business status with the highest total percentage
                    most_likely_business_status = business_status_summary.idxmax()
                    
                    # Find the number of competitor percentile
                    NumCompetitors_percentile = round(get_percentile_by_category(documents, 'NumCompetitors',avg_num_competitors,most_likely_category),2)
                    
                    # Find the totalraised percentile
                    TotalRaised_percentile = round(get_percentile_by_category(documents, 'TotalRaised',avg_total_raised,most_likely_category),2)
                
                    output = generate_commercial_analysis(NumCompetitors_percentile, most_likely_category, most_likely_business_status, TotalRaised_percentile, avg_num_competitors, avg_total_raised)
                            
                
                    st.write('<br><br>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                
                    with col1:
                        st.metric(label="##### ⚔️Estimated Number of Competitors", value=avg_num_competitors)
                    
                    with col2:
                        st.metric(label="##### 📈Estimated Total Raised (mln)", value=f"${avg_total_raised:,}")
                
                    st.write('<br><br>', unsafe_allow_html=True)
                
                
                    ## DONUTS
                    # Combine all data into a single pie chart
                    plt.figure(figsize=(10, 6))
                    
                    # Plot each entry in the DataFrame as a separate slice in the pie chart
                    wedges, texts, autotexts = plt.pie(df_cat['Percentage'], labels=df_cat['BusinessStatus'], autopct='%1.1f%%', startangle=140)
                    
                    # Draw a circle at the center of pie to turn it into a donut chart
                    centre_circle = plt.Circle((0,0),0.70,fc='white')
                    fig = plt.gcf()
                    fig.gca().add_artist(centre_circle)
                    
                    legend_labels = [f"{category}" for category in df_cat['Category']]
                    plt.legend(wedges, legend_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    
                    plt.title('Combined Business Status Distribution')
                    st.pyplot(plt)    
                
                    # st.write(output)
                
                    output = generate_commercial_analysis(NumCompetitors_percentile, most_likely_category, most_likely_business_status, TotalRaised_percentile, avg_num_competitors, avg_total_raised)
                    st.markdown("### Economical Analysis")
                    st.write("##### *🎉Congratulations on developing your innovative idea! After a thorough comparison with our extensive industry database, we've gathered insightful findings for your venture:*")
                    # Split the output into key points
                    key_points = output.split("\n\n")
                    
                    # Display each key point as a bullet point
                
                    st.markdown("<ul>", unsafe_allow_html=True)
                    for point in key_points:
                        st.markdown(f"<li style='font-size: 18px;'>{point}</li>", unsafe_allow_html=True)
        
                    st.markdown("</ul>", unsafe_allow_html=True)
        
                    st.write("##### *Wishing you the best in your entrepreneurial journey. Your innovation has the potential to make a remarkable difference!*")
        
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

            if not api_key:
                st.error("Please enter an API key.")
            else:
                # Display the DataFrame
                # st.write("Uploaded Data:")
                # st.dataframe(df)
                
                #### 最后的最后！
                with st.spinner('Evaluating your ideas, please wait for aproximately 90 seconds...'):
                    processed_results = process_inputs_in_parallel(df, api_key)
                    final_df = processed_results_to_df(processed_results)
                    # st.dataframe(final_df)
        
                    # Find the indices of the highest total_score and novelty_score
                    highest_total_score_idx = final_df['total_score'].idxmax()
                    highest_novelty_score_idx = final_df['novelty_score'].idxmax()
                    
                    num_ideas = len(final_df)
                    summary_sentence = f"We have evaluated {num_ideas} of your ideas. Idea {highest_total_score_idx + 1} achieved the overall best score, while Idea {highest_novelty_score_idx + 1} has the highest novelty score."
                    st.markdown(f"##### {summary_sentence}")
                    
                    # Iterate through each row in final_df to display the results
                    for index, row in final_df.iterrows():
                        # Determine if the expander should be open by default
                        if index == highest_total_score_idx:
                            expanded = True
                            special_message = "Congratulations! 🎉 This idea has achieved the overall best score in our evaluation."
                        elif index == highest_novelty_score_idx:
                            expanded = True
                            special_message = "Congratulations! 🌟 This idea has the highest novelty score, showcasing its unique and innovative aspects."
                        else:
                            expanded = False
                            special_message = ""
        
                        problem_snippet = row['problem'][:50] + "..."  # Display first 50 characters of the problem
                        expander_label = f"📝 Idea {index+1}: {problem_snippet}"
        
                        with st.expander(expander_label, expanded=expanded):
                            if special_message:
                                st.markdown(f"##### **{special_message}**")
                                 
                            st.markdown(f"**Problem:** {row['problem']}")
                            st.markdown(f"**Solution:** {row['solution']}")
                    
        
        
                            # Sustainability Related
                            is_sustainable = row['is_sustainable']
                            # sustainability_status = "Yes" if is_sustainable else "No"
                            # st.markdown(f"<h3 style='color:blue;'>Is the Idea Sustainability Related? {sustainability_status}</h3>", unsafe_allow_html=True)
                            if is_sustainable:
                                st.markdown("<h3 style='color: green;'>We've evaluated this, and it's exciting news: This idea is sustainability-related!</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown("<h3 style='color: red;'>After careful evaluation, it appears that this idea is not sustainability-related.</h3>", unsafe_allow_html=True)

                            # Total Score
                            st.markdown(f"<h3 style='color:green;'>Summary Score: {row['total_score']} / 170</h3>", unsafe_allow_html=True)
        
                            # Novelty Score and Analysis
                            st.markdown(f"<h3 style='color:purple;'>Novelty Score: {row['novelty_score']} / 170</h3>", unsafe_allow_html=True)
                            st.write("### Novelty Analysis:")
                            st.write(row['novelty_comment'])
        

        else:
            st.error("The CSV file must contain 'problem' and 'solution' columns.")



        
        
        



