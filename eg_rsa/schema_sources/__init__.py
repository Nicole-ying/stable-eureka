"""Schema source abstractions for EG-RSA.

A schema source answers only one question:

    Where does the initial reward schema come from?

The runner should not contain V1/V2 branching logic. It should simply call:

    schema = self.schema_source.load_or_create()
"""

from eg_rsa.schema_sources.base import SchemaSource
from eg_rsa.schema_sources.manual import ManualSchemaSource
from eg_rsa.schema_sources.llm_bootstrap import LLMBootstrapSchemaSource
from eg_rsa.schema_sources.factory import build_schema_source

__all__ = [
    "SchemaSource",
    "ManualSchemaSource",
    "LLMBootstrapSchemaSource",
    "build_schema_source",
]
