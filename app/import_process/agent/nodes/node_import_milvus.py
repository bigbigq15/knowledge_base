from typing import Dict, Any, List

from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState

CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

class NodeImportMilvus(NodeBase):
    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks_json_data,vector_dimension = self._step_1_check_input(state)
        client = self._step
        return state

    def _step_1_check_input(self,state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("核心参数chunks为空或非列表类型")


        first_chunk = chunks[0]
        vector_dimension = len(first_chunk.get("dense_vector"))
        return chunks,vector_dimension
