import concurrent.futures
import openai
import pandas as pd
import json
import re

# tool function to send given data with prompt to openai api
def chat_with_openai(input_data):

    # input data is of the form: {'id': 0, 'problem': '', 'solution': ''}

    # load system prompt from local file
    system_prompt = ""
    with open('system_prompt.txt', 'r', encoding='utf-8') as file:
        system_prompt = file.read()

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"problem: {input_data['problem']}. Solution: {input_data['solution']} "}
            ],
        max_tokens=4096,
    )
    ai_response = response["choices"][0]["message"]["content"]

    return (input_data['id'], input_data['problem'], input_data['solution'], ai_response)


# parallel function, takes a dataframe and a string of api_key as input
# return lists of (input_data['id'], input_data['problem'], input_data['solution'], ai_response)
def process_inputs_in_parallel(inputs, api_key):

    # convert the input dataframe to a dictionary
    inputs = inputs.to_dict('index')

    openai.api_key = api_key
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each input data to be processed in parallel
        futures = [executor.submit(chat_with_openai, messages) for _, messages in inputs.items()]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


# input is the returned results from process_inputs in parallel
# returns a processed dataframe with bool: is_sustainable, total_score, novelty_score.
def processed_results_to_df(processed_results):
    summary_df = pd.DataFrame(processed_results, columns=['id', 'problem', 'solution', 'ai_response'])
    summary_df['ai_response'] = summary_df['ai_response'].apply(lambda x: re.sub(r'`|json', '', x))
    # add a new column that use json to load the ai_response
    summary_df['ai_response_json'] = summary_df['ai_response'].apply(lambda x: json.loads(x))

    def safe_access_evaluation(data, key):
        # Check if 'Evaluation' and the specific key exist and have a non-empty value
        if 'Evaluation' in data and key in data['Evaluation'] and data['Evaluation'][key] != '':
            try:
                return int(data['Evaluation'][key])  # Convert to integer
            except ValueError:
                return None  # Return None if conversion fails
        else:
            return None  # Return None if keys do not exist or values are empty
    
    summary_df['is_sustainable'] = summary_df['ai_response_json'].apply(
        lambda x: x.get('Idea_Sustainability_Related', False))
    summary_df['total_score'] = summary_df['ai_response_json'].apply(
        lambda x: safe_access_evaluation(x, 'Total_Score'))
    summary_df['novelty_score'] = summary_df['ai_response_json'].apply(
        lambda x: safe_access_evaluation(x, 'Novelty_Score'))
    summary_df['novelty_comment'] = summary_df['ai_response_json'].apply(
        lambda x: x['Evaluation']['Novelty_Evaluation']['Comment'] if 'Evaluation' in x and 'Novelty_Evaluation' in x['Evaluation'] and 'Comment' in x['Evaluation']['Novelty_Evaluation'] else "")

    # sort by total score
    summary_df.sort_values(by=['total_score'], ascending=False, inplace=True)

    return summary_df

# processed_results = process_inputs_in_parallel(df)
