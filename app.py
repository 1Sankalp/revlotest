from flask import Flask, request, render_template
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import os
import wikipedia
app = Flask(__name__)

import os
os.environ.setdefault('OPENAI_API_KEY', '')

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."

def wiki_summary(file_path):
    content = read_text_file(file_path)
    return content


# Initialize OpenAI and other components
llm = OpenAI(temperature=0.5)

step_wise_prompt = PromptTemplate(
    input_variables=['title', 'local_data'],
    template='Write me all the steps I need to follow in order to {title} answering both "what to do" and "how to do" naming several tools that one needs while leveraging Wikipedia research: {local_data}'
)

memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
step_wise_chain = LLMChain(llm=llm, prompt=step_wise_prompt, verbose=True, memory=memory)
wiki = WikipediaAPIWrapper()

def format_response(response):
    # Split the response into points using line breaks
    points = response.split('\n')
    return points

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    prompt = request.form.get('prompt')

    if prompt:
        local_data = wiki_summary('final.txt')  
        max_points = 4
        response = step_wise_chain.run(title=prompt, local_data=local_data)

        response_points = response.strip().split('\n\n') 
        
        # Format the response into points
        formatted_response = response_points[:max_points]
    else:
        formatted_response = ["Please enter a prompt."]

    return render_template('result.html', response=formatted_response, user_query=prompt)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

