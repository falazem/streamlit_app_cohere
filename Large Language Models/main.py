import os
import cohere
import guardrails as gd
from guardrails.hub import ValidRange, ValidChoices
from pydantic import BaseModel, Field
from rich import print
from typing import List
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import evaluate
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

# Load environment variables from .env file
load_dotenv()

# Setup the Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY")) # Get your free API key: https://dashboard.cohere.com/api-keys

raw_documents = [
    {
        "title": "Crafting Effective Prompts",
        "url": "https://docs.cohere.com/docs/crafting-effective-prompts"},
    {
        "title": "Advanced Prompt Engineering Techniques",
        "url": "https://docs.cohere.com/docs/advanced-prompt-engineering-techniques"},
    {
        "title": "Prompt Truncation",
        "url": "https://docs.cohere.com/docs/prompt-truncation"},
    {
        "title": "Preambles",
        "url": "https://docs.cohere.com/docs/preambles"}
]

class Vectorstore:
    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for raw_document in self.raw_documents:
            elements = partition_html(url=raw_document["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document", embedding_types=["float"]
            ).embeddings.float
            self.docs_embs.extend(docs_embs_batch)
            
    def index(self) -> None:
        """
        Indexes the document chunks for efficient retrieval.
        """
        print("Indexing document chunks...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} document chunks.")


    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """

        # Dense retrieval
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query", embedding_types=["float"]
        ).embeddings.float
        
        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        # Reranking
        rank_fields = ["title", "text"] # We'll use the title and text fields for reranking

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]
        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )

        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results.results]

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved

vectorstore = Vectorstore(raw_documents)

print(vectorstore.retrieve("Prompting by giving examples"))

def run_chatbot(message, chat_history=[]):
    
    # Generate search queries by asking the model to create them
    query_generation_response = co.chat(
        message=f"Generate 1-3 search queries to help answer this question: {message}\nProvide only the search queries, one per line, without numbering or explanation.",
        model="command-a-03-2025",
    )
    
    # Parse the search queries from the response
    search_queries = []
    if query_generation_response.text.strip():
        # Split by newlines and clean up
        potential_queries = [q.strip() for q in query_generation_response.text.strip().split('\n') if q.strip()]
        # Remove common prefixes/numbering
        for q in potential_queries:
            # Remove numbering like "1.", "2.", etc.
            cleaned = q.lstrip('0123456789.-) ').strip('"\'')
            if cleaned and len(cleaned) > 5:  # Basic validation
                search_queries.append(cleaned)
    
    # If no queries were generated, use the original message as the query
    if not search_queries:
        search_queries = [message]

    # Retrieve the documents
    print("Retrieving information...", end="")

    # Retrieve document chunks for each query
    documents = []
    for query in search_queries:
        documents.extend(vectorstore.retrieve(query))

    # Use document chunks to respond
    response = co.chat_stream(
        message=message,
        model="command-a-03-2025",
        documents=documents,
        chat_history=chat_history,
    )
        
    # Print the chatbot response and citations
    chatbot_response = ""
    print("\nChatbot:")

    for event in response:
        if event.event_type == "text-generation":
            print(event.text, end="")
            chatbot_response += event.text
        if event.event_type == "stream-end":
            if event.response.citations:
                print("\n\nCITATIONS:")
                for citation in event.response.citations:
                    print(citation)
            if event.response.documents:
                print("\nCITED DOCUMENTS:")
                for document in event.response.documents:
                    print(document)
            # Update the chat history for the next turn
            chat_history = event.response.chat_history

    return chat_history