# from langchain.schema import Document
from langchain_core.documents import Document

from langflow import CustomComponent
from langflow.customer_base_class.document.llamaindexdocument import LLamaIndexDocument


class ConverterLCLoaderComponent(CustomComponent):
    display_name: str = "Converter Loader"
    description: str = "LangChain Loader"
    beta = True

    def build_config(self):
        print("FileLoaderComponent.build_config")

        return {
            "code": {"show": False},
            "converter_documents": {
                "display_name": "LLamaIndexDocument",
                "required": True,
                # "field_type": "LLamaIndexDocument",
            },
        }

    def build(self, converter_documents: list[LLamaIndexDocument]) -> list[Document]:
        print("FileLoaderComponent.build converter")
        print(converter_documents)

        result_documents = []
        for document in converter_documents:
            print(document)
            result_documents.append(document.to_langchain_format())
        print("============================================")
        print(result_documents)

        return result_documents
