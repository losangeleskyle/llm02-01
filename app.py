from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio
import nest_asyncio
nest_asyncio.apply()

text_loader = TextFileLoader("data/KingLear.txt")
documents = text_loader.load_documents()
len(documents)
print(documents[0][:100])

text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)
len(split_documents)
split_documents[0:1]

vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

vector_db.search_by_text("Your servant Kent. Where is your servant Caius?", k=3)

from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)

from aimakerspace.openai_utils.chatmodel import ChatOpenAI

chat_openai = ChatOpenAI()
user_prompt_template = "{content}"
user_role_prompt = UserRolePrompt(user_prompt_template)
system_prompt_template = (
    "You are an expert in {expertise}, you always answer in a kind way."
)
system_role_prompt = SystemRolePrompt(system_prompt_template)

messages = [
    user_role_prompt.create_message(
        content="What is the best way to write a loop?"
    ),
    system_role_prompt.create_message(expertise="Python"),
]

response = chat_openai.run(messages)
print(response)

RAQA_PROMPT_TEMPLATE = """
Use the provided context to answer the user's query. 

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".

Context:
{context}
"""

raqa_prompt = SystemRolePrompt(RAQA_PROMPT_TEMPLATE)

USER_PROMPT_TEMPLATE = """
User Query:
{user_query}
"""

user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    def run_pipeline(self, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)
        
        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = raqa_prompt.create_message(context=context_prompt)

        formatted_user_prompt = user_prompt.create_message(user_query=user_query)
        
        return self.llm.run([formatted_system_prompt, formatted_user_prompt])
retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai
)
retrieval_augmented_qa_pipeline.run_pipeline("Who is King Lear?")

retrieval_augmented_qa_pipeline.run_pipeline("Who is Batman?")

retrieval_augmented_qa_pipeline.run_pipeline("What happens to Cordelia?")