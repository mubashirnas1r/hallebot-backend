from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from flask_cors import CORS
import markdown

load_dotenv()

CHAT_MODEL = 'gpt-4o-mini'
MAX_TOKENS = 500
MAX_TEMPERATURE = 0
MAX_CHAT_HISTORY_LENGTH = 30
PROMPT = """You are a customer support and shopping assistant chatbot for our website, 'https://halle.se/'. Your name is 'HalleBot'. Your goal is to assist users with their queries, helping them discover products, provide pricing information, recommend the best options, and offer links to relevant products or pages for easy navigation.

Throughout your conversation, maintain a friendly and helpful tone while ensuring the users have a smooth shopping experience. Provide insightful information based on the details you have, including available products, current offers, and promotions. Always give users a quick, seamless way to explore or purchase the products by including relevant links. links are provided as SOURCE heading for each detail use it wisely."""


app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def list_folders(directory_path):
    try:
        folder_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
        if folder_names == []:
            return False
        return folder_names
    except FileNotFoundError:
        print(f"The directory '{directory_path}' does not exist.")
        return False

def load_chats():
    file_path = f"{os.getcwd()}/conversations.json"
    with open(file_path, 'r') as file:
        conversations = json.load(file)
    return conversations

def generate_response(previous_history=[], vector_db_collection=None, language="english"):
    print("- GPT is Writing Answer...")

    prompt = f"""{PROMPT}/n Information about our business: /n{vector_db_collection}/nAlways speak in language code: '{language}'"""

    new_messages = [
            {
                "role": "system",
                "content": prompt
            }] + previous_history
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=new_messages,
        temperature=MAX_TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content
    print("Message Generated...")
    print("Sending Response back...")
    return answer

@app.route('/v1/chatbot/<string:conversation_id>/chat', methods=['POST'])
def chat(conversation_id):

    query = request.args.get('query')
    language = request.args.get('lang','english')
    category = request.args.get('category','category')
    sub_category = request.args.get('sub_category','sub-category')

    chats = load_chats()

    print("Request Received in Chat Endpoint...")
    try:
        chat_history = chats["bot"][conversation_id]["chat"][-MAX_CHAT_HISTORY_LENGTH:]
        print("Previous chat is loaded")
    except:
        print("New chat initialized")
        chat_history = []
    print(chat_history)
    print(len(chat_history))
    chat_history += [{"role": "user", "content": query}]
    print(len(chat_history))

    embeddings = OpenAIEmbeddings()
    docs = None
    persist_directory = f'{os.getcwd()}/database'  # f'database/{page_type}'
    pdfs_indexes = list_folders(persist_directory)
    print(pdfs_indexes)
    if pdfs_indexes != False:
        print("Found Vectorized PDFs!")
        pdfs_vector_db = FAISS.load_local(f'{persist_directory}/{pdfs_indexes[0]}', embeddings, allow_dangerous_deserialization=True)
        print("Merging Vectorized PDFs..")
        for index in range(1, len(pdfs_indexes)):
            new_persist_directory = f'{persist_directory}/{pdfs_indexes[index]}'
            newVectordb = FAISS.load_local(new_persist_directory, embeddings, allow_dangerous_deserialization=True)
            pdfs_vector_db.merge_from(newVectordb)

        print("Searching for Questions in PDFs VectorDB...")
        retriever = pdfs_vector_db.as_retriever(search_type="mmr")
        docs = retriever.invoke(query, k=3)
    else:
        error_message = {"status": 403, "message": f"No Trainings Found."}
        return jsonify(error_message), 403

    print("Found Something...")
    found_pdfs = ""
    for i, doc in enumerate(docs):
        found_pdfs += str(f"{i + 1}. SOURCE: {doc.metadata['source']} \n {doc.page_content}\n____\n")

    try:
        message = generate_response(
            vector_db_collection=found_pdfs,
            previous_history=chat_history,
            language=language,
        )

        if conversation_id != "test":
            try:
                chats['bot'][conversation_id]["chat"] += [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": message}
                ]
                chats['bot'][conversation_id]["last_updated"] = str(date.today())
            except:
                chats['bot'][conversation_id] = {
                    "last_updated": str(date.today()),
                    "chat": [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": message}
                    ]
                }
            file_path = f"{os.getcwd()}/conversations.json"
            with open(file_path, 'w') as file:
                json.dump(chats, file, indent=4)

        successful_response = {'status': 200, 'message': markdown.markdown(message)[3:-4]}
        return jsonify(successful_response), 200
    except Exception as e:
        print(str(e))
        error_message = {"status": 403, "message": "Something went wrong while Chatting, please see system logs."}
        return jsonify(error_message), 403

if __name__ == '__main__':
    app.run(debug=True)
