import os
from os.path import splitext

from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState, create_default_state


class NodeEntry(NodeBase):
    """
    节点: 入口节点 (EntryNode)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph知识库导入工作流 - 入口节点
        执行链路：__start__ → 本节点 → route_after_entry(条件路由) → ... → 流程终止
        核心职责：初始化参数校验 | 自动判断文件类型(PDF/MD) | 设置解析开关 | 提取业务标识
        :param state: 必须包含 task_id(任务ID)、local_file_path(文件路径)、local_dir(流转到第二步的时候需要)
        :return: 新增/更新 is_pdf_read_enabled/is_md_read_enabled、pdf_path/md_path、file_title
        is_pdf_read_enabled/is_md_read_enabled：如果文件的扩展名是md，is_md_read_enabled=True，如果扩展名是pdf，is_pdf_read_enabled=True
        pdf_path/md_path：如果文件的扩展名是md，将local_file_path的值赋值给md_path，如果扩展名是pdf，将local_file_path的值赋值给pdf_path
        file_title：提取文件名
        """

        # 1. 核心参数提取与非空校验
        document_path = state.get("local_file_path","")
        if not document_path:
            logger.error("核心参数local_file_path缺失")
            return state

        if document_path.endswith(".pdf"):
            state["is_pdf_read_enabled"] = True
            logger.info(f"文件类型校验通过：{document_path} → PDF格式，开启PDF解析流程")
            state["pdf_path"] = document_path
        elif document_path.endswith(".md"):
            state["is_md_read_enabled"] = True
            logger.info(f"文件类型校验通过：{document_path} → MD格式，开启MD解析流程")
            state["md_path"] = document_path
        else:
            logger.warning(f"文件类型校验失败：{document_path} → 不支持的格式，仅支持.pdf/.md")

        file_name = os.path.basename(document_path)
        name_only, extension = os.path.splitext(file_name)
        state["file_title"] = name_only
        logger.info(f"文件业务标识提取完成：file_title = {state['file_title']}")
        return state





