import os

from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph
from typing import Optional

from app.core.logger import logger
from app.import_process.agent.nodes.node_bge_embedding import NodeBgeEmbedding
from app.import_process.agent.nodes.node_document_split import NodeDocumentSplit
from app.import_process.agent.nodes.node_entry import NodeEntry
from app.import_process.agent.nodes.node_import_milvus1 import NodeImportMilvus
from app.import_process.agent.nodes.node_item_name_recognition import NodeItemNameRecognition
from app.import_process.agent.nodes.node_md_img import NodeMdImg
from app.import_process.agent.nodes.node_pdf_to_md import NodePdfToMd
from app.import_process.agent.state import ImportGraphState, create_default_state

# 初始化环境变量（类加载前执行，保证全局生效）
load_dotenv()


class KBImportWorkflow:
    """
    知识库导入工作流类
    封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """
    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(ImportGraphState)
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离）
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app: Optional[object] = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        self.node_entry = NodeEntry()
        self.node_pdf_to_md = NodePdfToMd()
        self.node_md_img = NodeMdImg()
        self.node_document_split = NodeDocumentSplit()
        self.node_item_name_recognition = NodeItemNameRecognition()
        self.node_bge_embedding = NodeBgeEmbedding()
        self.node_import_milvus = NodeImportMilvus()

    def _register_nodes(self):
        """注册所有节点到工作流（私有方法，统一管理节点注册）"""
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node("node_entry", self.node_entry)
        self.workflow.add_node("node_pdf_to_md", self.node_pdf_to_md)
        self.workflow.add_node("node_md_img", self.node_md_img)
        self.workflow.add_node("node_document_split", self.node_document_split)
        self.workflow.add_node("node_item_name_recognition", self.node_item_name_recognition)
        self.workflow.add_node("node_bge_embedding", self.node_bge_embedding)
        self.workflow.add_node("node_import_milvus", self.node_import_milvus)

    def _route_after_entry(self, state: ImportGraphState) -> str:
        """入口节点后的条件路由函数（私有方法，封装路由逻辑）"""
        if state.get("is_md_read_enabled"):
            return "node_md_img"
        elif state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        else:
            return END

    def _setup_routes(self):
        """设置工作流路由规则（私有方法，封装边的定义）"""
        # 设置入口节点
        self.workflow.set_entry_point("node_entry")
        # 注册条件路由边
        self.workflow.add_conditional_edges(
            "node_entry",
            self._route_after_entry,
            {
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",
                END: END
            }
        )
        # 注册静态顺序边
        self.workflow.add_edge("node_pdf_to_md", "node_md_img")
        self.workflow.add_edge("node_md_img", "node_document_split")
        self.workflow.add_edge("node_document_split", "node_item_name_recognition")
        self.workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
        self.workflow.add_edge("node_bge_embedding", "node_import_milvus")
        self.workflow.add_edge("node_import_milvus", END)

    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: ImportGraphState, stream: bool = False) -> ImportGraphState:
        """
        统一执行入口，支持切换invoke/stream
        :param initial_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        """"""
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

    @classmethod
    def create_and_run(cls, initial_state: ImportGraphState, stream: bool = False) -> ImportGraphState:
        """
        快捷方法：创建工作流实例并立即执行（兼容原有函数式调用习惯）
        :param initial_state: 初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        workflow = cls()
        return workflow.run(initial_state, stream)


# ===================== 用法示例 =====================

if __name__ == "__main__":

    # 定义初始状态
    initial_state = create_default_state(
        task_id="task_demo",
        local_file_path="万用表的使用.pdf"
    )

    # 用法1：标准类用法（推荐，支持多实例）
    # 创建工作流实例
    kb_import_app = KBImportWorkflow()

    # 执行工作流

    # 非流式输出
    final_state = kb_import_app.run(initial_state)
    logger.info(f"工作流执行完成！最终状态: {final_state}")

    # 流式输出
    # final_state = None
    # for chunk in kb_import_app.run(initial_state, stream=True):
    #     node_name, node_state = next(iter(chunk.items()))
    #     final_state = node_state
    #     logger.info(f"【{node_name}】流式输出，local_file_path={node_state['local_file_path']}")
    #     pass
    # logger.info(f"工作流执行完成！最终状态: {final_state}")


    # 用法2：快捷调用
    # # 非流式输出
    # final_state = KBImportWorkflow.create_and_run(initial_state, stream=True)
    # # 流式输出
    # for chunk in KBImportWorkflow.create_and_run(initial_state, stream=True):
    #     node_name, node_state = next(iter(chunk.items()))
    #     final_state = node_state
    #     logger.info(f"【{node_name}】流式输出，local_file_path={node_state['local_file_path']}")
    #     pass
    # logger.info(f"工作流执行完成！最终状态: {final_state}")