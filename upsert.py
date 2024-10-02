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


def split_text_into_lines(input_text, max_words_per_line):
    words = input_text.split()
    lines = []
    current_line = []

    for word in words:
        if len(current_line) + len(word) + 1 <= max_words_per_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def nlp_upsert(
    filename, index_name, name_space, nlp_id, chunk_size, stride, page_begin, page_end
):
    """
    upsert a whole PDF file (with begin page and end page information) to the pinecone vector database

    Parameters:
    filename (str): The file name.
    index_name (str): The pinecone index name.
    name_space (str): The namespace we want to place for all related docuement.
    nlp_id (str): A common ID prefix to reference to document.
    chunk_size (int): The chunk size, how many lines as one chunks.
    stride (int): The overlap side, how many lines as overlap between chunks.
    page_begin (int): Which page in the PDF file to begin for upsert.
    page_end (int): Which page is the ending page for upsert.

    Returns:
    None: No return.
    """
    doc = ""

    reader = PdfReader(filename)

    for i in range(page_begin, page_end):
        doc += reader.pages[i].extract_text()
        print("page completed:", i)

    doc = split_text_into_lines(doc, 30)
    print("The total lines: ", len(doc))

    # Connect to index
    index = pc.Index(index_name)

    count = 0
    for i in range(0, len(doc), chunk_size):
        # find begining and end of the chunk
        i_begin = max(0, i - stride)
        i_end = min(len(doc), i_begin + chunk_size)

        doc_chunk = doc[i_begin:i_end]
        print("-" * 80)
        print("The ", i // chunk_size + 1, " doc chunk text:", doc_chunk)

        texts = ""
        for x in doc_chunk:
            texts += x
        print("Texts:", texts)

        # Create embeddings of the chunk texts
        try:
            res = openai.embeddings.create(input=texts, model=embed_model)
        except:
            done = False
            while not done:
                time.sleep(10)
                try:
                    res = openai.embeddings.create(input=texts, model=embed_model)
                    done = True
                except:
                    pass
        embed = res.data[0].embedding
        print("Embeds length:", len(embed))

        # Meta data preparation
        metadata = {"text": texts}

        count += 1
        print("Upserted vector count is: ", count)
        print("=" * 80)

        # upsert to pinecone and corresponding namespace

        index.upsert(
            vectors=[
                {"id": nlp_id + "_" + str(count), "metadata": metadata, "values": embed}
            ],
            namespace=name_space,
        )


def page_len(filename):
    reader = PdfReader(filename)
    page_len = len(reader.pages)
    return page_len


filename1 = r"C:\Users\chris\gptApplication\CSTUBOT\CSTU Chatbot .pdf"
filename2 = r"C:\Users\chris\gptApplication\CSTUBOT\Website Information.pdf"
print("Knowledge base file name:", filename1)


print("length of the knowledge base file:", page_len)

nlp_upsert(filename1, index_name, "cstugpt", "gpt", 5, 2, 0, page_len(filename1))

nlp_upsert(filename2, index_name, "cstugpt", "gpt", 5, 2, 0, page_len(filename2))
# signature of nlp_upsert(filename, index_name, name_space, nlp_id, chunk_size, stride, page_begin, page_end)

print(index.describe_index_stats())
