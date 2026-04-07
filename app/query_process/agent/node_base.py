from abc import ABC, abstractmethod
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task


class NodeBase(ABC):
    """
    查询流程节点基类
    所有节点类都应继承此基类，实现 process 方法
    使用示例:
        class NodeXx(BaseNode):
            name = "node_xx"
            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node_xx = NodeXx()
        workflow.add_node("node_xx", node_xx)
    """

    #节点名称，子类应覆盖
    name: str = "base_node"


    def __init__(self):
        """
        强制子类设置 name
        """
        if self.name == "base_node":
            raise ValueError(f"子类 {self.__class__.__name__} 必须覆盖 name 类属性")


    def __call__(self, state: QueryGraphState) -> QueryGraphState:
         """
         节点执行入口
         LangGraph 调用节点时会调用此方法，提供统一的日志输出、任务追踪和异常处理
         :param state: 工作流状态对象
         :return: 更新后的状态对象
         """

         # 节点启动日志，打印当前工作流状态
         logger.info(f"{'*' * 20}【{self.name}】节点启动{'*' * 20}")
         logger.debug(f"【{self.name}】节点当前工作流状态：{format_state(state)}")

         # 开始：记录节点运行状态
         # 此处为任务追踪，后面会讲
         # add_running_task(state["task_id"], self.name)

         try:

             # sleep(1)

             # 节点核心处理逻辑
             state = self.process(state)

             # 结束：记录节点运行状态
             # 此处为任务追踪，后面会讲
             # add_done_task(state["task_id"], self.name)

             # 节点完成日志，打印当前工作流状态
             logger.debug(f"【{self.name}】节点更新后工作流状态：{format_state(state)}")
             logger.info(f"{'*' * 20}【{self.name}】节点执行完成{'*' * 20}\n")

             return state

         except Exception as e:

             # 统一处理所有步骤的异常
             error_msg = f"【{self.name}】流程执行失败：{str(e)}"
             logger.exception(error_msg, e) # logger.exception 会打印完整的堆栈跟踪，等同于下面的用法
             # logger.exception(error_msg, exc_info=True)  # exc_info=True 会打印完整的堆栈跟踪

             raise  # 重新抛出异常，确保工作流引擎知道此节点失败并停止后续流程

    @abstractmethod
    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点核心处理逻辑
        子类必须实现此方法
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        pass