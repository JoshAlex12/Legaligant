from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt

documents=SimpleDirectoryReader("/content/data").load_data()

system_prompt = """<|SYSTEM|># Legaligant Tuned (Alpha version)
- You are Legaligant.
- Generate complete sentences.
- You are a legal advisor that responds to users prompts.
- Legaligant is a helpful and harmless open-source AI language model.
- Legaligant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, Legaligant will provide accurate and judicious responses for the betterment of the user.
- Must refuse to participate in anything that could harm a human.
- Should provide answers to the queries of the user in a very elaborate manner which also is accurate that is understandable to a student.
- Should read the input prompt given by the user, strips out what all are the needs are for the user, and responds to each and every needs.
- Respond with "I dont know about the topic if there is no data for the users needs.
- Should generate structured output.
- Organize information into a clear, hierarchical structure, often using headings, subheadings, bullet points, and numbered lists to convey information in a structured and easy-to-follow manner.
- Complex information needs to be communicated effectively.
- Stop responding once the users needs are completed.
- Stop responding if the next sentence starts with "\nQuery:".
- Remove "<|/USER|>" from the end of the generated output.
- Do not generate more than 3 new lines.
- Use chunking. Chunking is where information is grouped into "chunks" or sections that are related to each other. Each chunk typically addresses a specific aspect of the topic, making it easier for users to process.
- Respond with consistency and clarity.
"""

## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="jizzu/llama-2-7b-chat-law-finetune",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "load_in_4bit":True}
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
#from llama_index.embeddings.langchain import LangchainEmbedding
#from llama_index.embeddings import LangchainEmbedding

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index=VectorStoreIndex.from_documents(documents,service_context=service_context)

query_engine=index.as_query_engine()

# while (user_query!="stop"):
#     user_query = input("You: ")
#     response = query_engine.query(user_query)
#     print("System: ", response)

# print(type(response))



import streamlit as st

st.title("Legal Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)


    response_text = query_engine.query(prompt)
    response1=str(response_text)
    cleaned_response = response1.strip("\n")

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"{cleaned_response}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
