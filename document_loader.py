import os
import json
import requests
from utils import compute_sha1
from bs4 import BeautifulSoup
from typing import List
from abc import ABC, abstractmethod
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseDocumentLoader(ABC):
    def __init__(self, config_path: str = "config.yaml", config: dict = None):
        self.config = config or self._load_config(config_path)
        self.chunk_size = self.config.get("chunk", {}).get("size", 800)
        self.chunk_overlap = self.config.get("chunk", {}).get("overlap", 80)

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    @abstractmethod
    def load(self) -> List[Document]:
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        return self.assign_chunk_ids(chunks)

    @staticmethod
    def assign_chunk_ids(chunks: List[Document]) -> List[Document]:

        for doc in chunks:
            file = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", -1)
            chunk_hash = compute_sha1(f"{file}:{page}:{doc.page_content}")[:20]
            doc.metadata["file"] = file
            doc.metadata["page"] = page
            doc.metadata["chunk"] = chunk_hash
            doc.metadata["id"] = f"{file}:{page}:{chunk_hash}"

        return chunks


class PDFLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.path, file))
                docs.extend(loader.load())
        return docs


class JSONLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            if file.endswith(".json"):
                with open(os.path.join(self.path, file), "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            content = entry.get("text") or json.dumps(entry)
                            docs.append(Document(page_content=content, metadata={"source": file}))
                    elif isinstance(data, dict):
                        content = data.get("text") or json.dumps(data)
                        docs.append(Document(page_content=content, metadata={"source": file}))
        return docs


class WebPageLoader(BaseDocumentLoader):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def load(self) -> List[Document]:
        docs = []
        for file in os.listdir(self.path):
            full_path = os.path.join(self.path, file)
            if file.endswith(".txt"):  # each line is a URL
                with open(full_path, "r") as f:
                    for url in f.readlines():
                        url = url.strip()
                        try:
                            html = requests.get(url, timeout=10).text
                            text = self.extract_text(html)
                            docs.append(Document(page_content=text, metadata={"source": url}))
                        except Exception as e:
                            print(f"⚠️ Failed to load {url}: {e}")
            elif file.endswith(".html"):
                with open(full_path, "r", encoding="utf-8") as f:
                    html = f.read()
                    text = self.extract_text(html)
                    docs.append(Document(page_content=text, metadata={"source": file}))
        return docs

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)


class SmartDocumentLoader(BaseDocumentLoader):
    def __init__(self, config_path: str = "config.yaml", config: dict = None):
        super().__init__(config_path=config_path, config=config)
        self.path = self.config.get("data_path", "./data")

    def load(self) -> List[Document]:
        docs = []
        file_types = os.listdir(self.path)
        print(f"📄 Loaded {len(file_types)} documents from {self.path}")

        if any(f.endswith(".pdf") for f in file_types):
            docs.extend(PDFLoader(self.path, config=self.config).load())
        if any(f.endswith(".json") for f in file_types):
            docs.extend(JSONLoader(self.path, config=self.config).load())
        if any(f.endswith((".txt", ".html")) for f in file_types):
            docs.extend(WebPageLoader(self.path, config=self.config).load())

        return docs

