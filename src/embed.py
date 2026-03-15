"""Embeddings for chunks and queries. Dev/prod router."""

import os

from langchain_core.embeddings import Embeddings


def get_embeddings() -> Embeddings:
    """
    Return embeddings instance based on ENVIRONMENT.

    - dev: OpenAIEmbeddings (standard OpenAI API)
    - prod: AzureOpenAIEmbeddings
    """
    if os.getenv("ENVIRONMENT") == "dev":
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "ENVIRONMENT=dev requires OPENAI_API_KEY in .env"
            )
        return OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small",
        )
    else:
        from langchain_openai import AzureOpenAIEmbeddings

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise ValueError(
                "ENVIRONMENT=prod requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
            )
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            azure_endpoint=endpoint,
            api_key=api_key,
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15"),
        )
