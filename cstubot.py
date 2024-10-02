import json
import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import openai
import numpy as numpy
from sklearn.manifold import TSNE
from ast import literal_eval
from pyngrok import ngrok
from pinecone import Pinecone, PodSpec
import matplotlib.pyplot as plt
from tenacity import retry, wait_random_exponential, stop_after_attempt
from PyPDF2 import PdfReader

os.chdir("C:/Users/chris/gptApplication")
print(os.getcwd())
import time
from datetime import datetime
import pinecone
from pinecone import Pinecone, ServerlessSpec, PodSpec
import re

# load api keys
load_dotenv()
pc = Pinecone(api_key="9619c898-d209-48f8-9d1e-33b7e6f6636d")

# constants
GPT_MODEL = "gpt-4o"
embed_model = "text-embedding-3-small"
index_name = "cstubot"
limit = 3600

indexes_info = pc.list_indexes()
index = pc.Index(index_name)
index.describe_index_stats()


def index_exists(index_name):
    # Retrieve the list of indexes and store in a variable

    # Access the 'indexes' key which contains the list of index dictionaries
    for index in indexes_info:
        # Check if the 'name' key in each index dictionary matches the index_name
        if index["name"] == index_name:
            return True
    return False


# for serverless, we will do you the following. However, as starter, we may not eligible to use it.
if not index_exists(index_name):
    print(f"Index {index_name} does not exist.")
    pc.create_index(
        name=index_name,
        dimension=1536,  # actually, it is 1536
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# connect to index
index = pc.Index(index_name)
# view index stats
index.describe_index_stats()


delimiter = "####"
chatContext = [
    {
        "role": "system",
        "content": f"""
You are an advanced and amiable virtual assistant, specifically designed to assist with queries related to 
California Science and Technology University (CSTU) website.

As the designated virtual assistant for this website, your role is to provide accurate, conscise, and helpful responses based on the data \
given. Your responses should adhere to the information contained within the specified context, marked by {delimiter}.  
If the user asks a question that could have multiple answers, ask the user to specify out of select choices given by the virtual assistant. 
For the most up to date information, direct the users to the web page with the webpath where the information was found for each 
response.

For inquiries beyond school information, direct students to the relevant web pages and don't try to answer 
anything outside of data fed. 
""",
    }
]


def retrieve(query, name_space):
    res = openai.embeddings.create(input=[query], model=embed_model)

    # retrieve from Pinecone knowledge base
    xq = res.data[0].embedding

    index = pc.Index(index_name)

    # get relevant contexts
    res = index.query(vector=xq, top_k=3, include_metadata=True, namespace=name_space)
    contexts = [x["metadata"]["text"] for x in res["matches"]]

    # print("Length of contexts: ", len(contexts))
    # print(contexts)

    # build our prompt with the retrieved contexts included
    prompt = " "

    # append contexts until hitting limit
    count = 0
    proceed = True
    while proceed and count < len(contexts):
        if len(prompt) + len(contexts[count]) >= limit:
            proceed = False
        else:
            prompt += contexts[count]

        count += 1
    # End of while loop

    prompt = delimiter + prompt + delimiter

    return prompt


def chat_complete_prompt(chat_history, temperature=0.2):
    # query against the model "gpt-3.5-turbo-1106"
    try:
        completion = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=chat_history,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except openai.OpenAIError as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def chat_complete_messages(messages, temperature=0.2):
    try:
        # query against the model "gpt-3.5-turbo-1106"
        completion = openai.chat.completions.create(
            # model="gpt-4-1106-preview",
            model="gpt-4o",
            messages=messages,
            temperature=temperature,  # this is the degree of randomness of the model's output
        )
        return completion.choices[0].message.content
    except openai.OpenAIError as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return str(e)


def format_chat_prompt(message, chat_history, knowledge_context):
    # add initial to the chat history
    prompt = chatContext.copy()
    # Include the retrieved knowledge in the prompt
    prompt.append({"role": "system", "content": knowledge_context})
    for turn in chat_history:
        user_message, bot_message = turn
        prompt.append({"role": "user", "content": turn[0]})
        prompt.append({"role": "assistant", "content": turn[1]})
    prompt.append({"role": "user", "content": message + " "})
    return prompt


def respond(message, chat_history):
    knowledge_context = retrieve(message, "cstugpt")
    formatted_prompt = format_chat_prompt(message, chat_history, knowledge_context)
    response = chat_complete_prompt(formatted_prompt)
    chat_history.append((message, response))
    return "", chat_history


with gr.Blocks() as myDemo:
    gr.Markdown("<h1>CSTU ChatBot</h1>")
    gr.Markdown(
        "<p>Hi there! I'm CSTUbot, your virtual assistant. What would you like to know about CSTU?  </p>"
    )
    chatbot = gr.Chatbot(height=240)
    msg = gr.Textbox(label="Input here to talk with the ChatBot about CSTU")
    btn = gr.Button("Send")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(
        respond, inputs=[msg, chatbot], outputs=[msg, chatbot]
    )  # Press enter to submit

myDemo.launch(share=True)
