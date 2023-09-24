from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio
import nest_asyncio
nest_asyncio.apply()
text_loader = TextFileLoader("data/KingLear.txt")
documents = text_loader.load_documents()
len(documents)
print(documents[0][:100])