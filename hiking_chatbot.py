import os

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import torch
import streamlit as st

from transformers import AutoTokenizer
import transformers
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

st.title("Hiking Chatbot")
prompt_txt = st.text_input("Prompt")

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
# llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})

template = """
I want you to act like a hiking expert.
I will ask you questions about hiking, and I would like you to answer based on hiking expertise.
My question is "{prompt}"
"""
# prompt = PromptTemplate.from_template(template=template)

# llm_chain = LLMChain(prompt=prompt, llm=llm)

if prompt_txt:
    print(f"running - prompt: {prompt_txt}")

    sequences = pipeline(
        prompt_txt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        res = seq['generated_text']

    # res = llm_chain.run(prompt_txt)
    print(f"res: {res}")

    st.write(res)
