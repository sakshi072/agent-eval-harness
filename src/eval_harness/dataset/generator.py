"""
Generates a synthetic golden dataset from raw source documents, using
RAGAS's knowledge-graph-based TestsetGenerator - NOT hand-written
questions, and NOT built from pre-chunked fragments (RAGAS needs
coherent, reasonably complete documents to build meaningful cross-
document relationships; feeding it pre-chunked pieces would likely
produce a shallower testset).

COST DESIGN: local, free embedding model (matches rag-mcp-service's own
choice) so the embedding side of graph-building costs nothing regardless
of corpus size; a cheap LLM (RAGAS_GENERATOR_MODEL, default gpt-4o-mini)
for the genuinely necessary LLM calls.
"""
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from eval_harness.core.settings import settings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def load_source_document(source_dir:str) -> List[Document]:
    """Loads raw source files (the same ones originally ingested into
    rag-mcp-service) as LangChain Documents - NOT pre-chunked pieces."""
    docs = []
    for path in Path(source_dir).rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source":str(path)}))
    logger.info(f"Loaded {len(docs)} raw source document(s) from {source_dir}")
    return docs

def generate_golden_dataset(source_dir:str, testset_size:int=50):

    documents = load_source_document(source_dir)
    if not documents:
        raise ValueError(f"No .md documents found in {source_dir} - nothing to generate from")

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.ragas.generator_model))
    generator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.ragas.embedding_model))

    generator=TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    logger.info(
        f"Generating {testset_size} test cases from {len(documents)} documents "
        f"using {settings.ragas.generator_model} (LLM) + {settings.ragas.embedding_model} (embeddings, free/local)"
    )
    dataset = generator.generate_with_langchain_docs(documents, testset_size=testset_size)
    return dataset