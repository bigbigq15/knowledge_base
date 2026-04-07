import os


from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState, create_default_state



class NodeEntry(NodeBase):
    name: str ="node_entry"

    def process(self,state: ImportGraphState)-> ImportGraphState:
        document_path =state.get("local_file_path", "")
        if not document_path:
            logger.error("核心参数local_file_path缺失")
            return state
        if document_path.endswith("pdf"):
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = document_path
            logger.info("开启pdf路径")
        elif document_path.endswith("md"):
            state["is_md_read_enabled"] = True
            state["md_path"] = document_path
            logger.info("开启md路径")
        else:
            logger.error("错误路径")
        file_name = os.path.basename(document_path)
        state["file_title"] = file_name
        logger.info(f"文件的名字问{file_name}")
        return state

if __name__ == "__main__":
    node_entry =NodeEntry()
    node_state = create_default_state(
        task_id = 1,
        local_file_path ="F:/it/py/knowledge_base/need_file/Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf",
        local_dir = "F:/it/py/knowledge_base/output"
    )
    node_state_final = node_entry(node_state)







