# from langchain.schema import Document
from langchain_core.documents import Document

from langflow import CustomComponent
from langflow.utils.constants import LOADERS_INFO


class SimpleDirectoryReaderComponent(CustomComponent):
    # output_types: list[str] = ["Document"]
    display_name: str = "SimpleDirectoryReader Loader"
    description: str = "Generic SimpleDirectoryReader Loader"
    beta = True

    def build_config(self):
        print("FileLoaderComponent.build_config")
        loader_options = ["Automatic"] + [loader_info["name"] for loader_info in LOADERS_INFO]

        file_types = []
        suffixes = []

        for loader_info in LOADERS_INFO:
            if "allowedTypes" in loader_info:
                file_types.extend(loader_info["allowedTypes"])
                suffixes.extend([f".{ext}" for ext in loader_info["allowedTypes"]])

        return {
            "file_path": {
                "display_name": "File Path",
                "required": True,
                "field_type": "file",
                "file_types": [
                    "json",
                    "txt",
                    "csv",
                    "jsonl",
                    "html",
                    "htm",
                    "conllu",
                    "enex",
                    "msg",
                    "pdf",
                    "srt",
                    "eml",
                    "md",
                    "pptx",
                    "docx",
                ],
                "suffixes": [
                    ".json",
                    ".txt",
                    ".csv",
                    ".jsonl",
                    ".html",
                    ".htm",
                    ".conllu",
                    ".enex",
                    ".msg",
                    ".pdf",
                    ".srt",
                    ".eml",
                    ".md",
                    ".pptx",
                    ".docx",
                ],
                # "file_types" : file_types,
                # "suffixes": suffixes,
            },
            "loader": {
                "display_name": "Loader",
                "is_list": True,
                "required": True,
                "options": loader_options,
                "value": "Automatic",
            },
            "code": {"show": False},
        }

    def build(self, file_path: str, loader: str) -> list[Document]:
        print("FileLoaderComponent.build")
        print(file_path)
        file_type = file_path.split(".")[-1]
        from llama_index import SimpleDirectoryReader
        documents = SimpleDirectoryReader(
            input_files=[file_path]
        ).load_data()

        result_documents = []

        for idx in documents:
            document = Document(
                page_content=idx.text,
                metadata={**idx.metadata, 'additional_key': 'additional_value'}
            )
            result_documents.append(document)

        return result_documents
