from llama_index.schema import TextNode

"""Base schema for data structures."""
import textwrap
import uuid
from enum import Enum, auto
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Union

from llama_index.bridge.pydantic import Field
from llama_index.utils import SAMPLE_TEXT, truncate_text

if TYPE_CHECKING:
    from llama_index.bridge.langchain import Document as LCDocument

DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"
# NOTE: for pretty printing
TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70

ImageType = Union[str, BytesIO]


# class LLamaIndexDocument(Document):
#     _compat_fields = ["text", "text_as_html", "text_as_markdown", "text_as_list"]


class ObjectType(str, Enum):
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()
    DOCUMENT = auto()


class LLamaIndexDocument(TextNode):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    # TODO: A lot of backwards compatibility logic here, clean up
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID of the node.",
        alias="doc_id",
    )

    _compat_fields = {"doc_id": "id_", "extra_info": "metadata"}

    @classmethod
    def get_type(cls) -> str:
        """Get Document type."""
        return ObjectType.DOCUMENT

    @property
    def doc_id(self) -> str:
        """Get document ID."""
        return self.id_

    def __str__(self) -> str:
        source_text_truncated = truncate_text(
            self.get_content().strip(), TRUNCATE_LENGTH
        )
        source_text_wrapped = textwrap.fill(
            f"Text: {source_text_truncated}\n", width=WRAP_WIDTH
        )
        return f"Doc ID: {self.doc_id}\n{source_text_wrapped}"

    def get_doc_id(self) -> str:
        """TODO: Deprecated: Get document ID."""
        return self.id_

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._compat_fields:
            name = self._compat_fields[name]
        super().__setattr__(name, value)

    def to_langchain_format(self) -> "LCDocument":
        """Convert struct to LangChain document format."""
        from llama_index.bridge.langchain import Document as LCDocument

        metadata = self.metadata or {}
        return LCDocument(page_content=self.text, metadata=metadata)

    @classmethod
    def from_langchain_format(cls, doc: "LCDocument") -> "LLamaIndexDocument":
        """Convert struct from LangChain document format."""
        return cls(text=doc.page_content, metadata=doc.metadata)

    # @classmethod
    # def from_haystack_format(cls, doc: "HaystackDocument") -> "LLamaIndexDocument":
    #     """Convert struct from Haystack document format."""
    #     return cls(
    #         text=doc.content, metadata=doc.meta, embedding=doc.embedding, id_=doc.id
    #     )

    def to_embedchain_format(self) -> Dict[str, Any]:
        """Convert struct to EmbedChain document format."""
        return {
            "doc_id": self.id_,
            "data": {"content": self.text, "meta_data": self.metadata},
        }

    @classmethod
    def from_embedchain_format(cls, doc: Dict[str, Any]) -> "LLamaIndexDocument":
        """Convert struct from EmbedChain document format."""
        return cls(
            text=doc["data"]["content"],
            metadata=doc["data"]["meta_data"],
            id_=doc["doc_id"],
        )


    @classmethod
    def from_semantic_kernel_format(cls, doc: "MemoryRecord") -> "LLamaIndexDocument":
        """Convert struct from Semantic Kernel document format."""
        return cls(
            text=doc._text,
            metadata={"additional_metadata": doc._additional_metadata},
            embedding=doc._embedding.tolist() if doc._embedding is not None else None,
            id_=doc._id,
        )

    def to_vectorflow(self, client: Any) -> None:
        """Send a document to vectorflow, since they don't have a document object."""
        # write document to temp file
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            f.write(self.text.encode("utf-8"))
            f.flush()
            client.embed(f.name)

    @classmethod
    def example(cls) -> "LLamaIndexDocument":
        return LLamaIndexDocument(
            text=SAMPLE_TEXT,
            metadata={"filename": "README.md", "category": "codebase"},
        )

    @classmethod
    def class_name(cls) -> str:
        return "LLamaIndexDocument"
