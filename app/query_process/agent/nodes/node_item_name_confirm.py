import json
from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import expect

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        必要参数：session_id、original_query
        更新参数：history

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 1.校验参数
        session_id, original_query = self._step_1_validate_param(state)

        # 2.获取历史记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")

        # 更新状态
        state["history"] = history

        # 3.用户初始信息保存
        message_id = save_chat_message(
            session_id=session_id,
            role="user",
            text=original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 步骤4：提取信息
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names", [])
        rewritten_query = extract_res.get("rewritten_query", original_query)
        # 更新状态
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names
        # 5. & 6. 如果有提取到商品名，进行搜索和对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:

    def _step_1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:
        session_id = state.get("session_id") or "".strip()
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query") or "".strip()
        if not original_query:
            raise ValueError("核心参数original_query缺失")
        return session_id, original_query

    def _step_4_extract_info(self, query, history) -> Dict:
        """
        利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
        若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
        :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
        :param history: 列表[字典] - 近期会话历史，每条消息含role/text等字段，格式：[{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}, ...]
        :return: 字典 - 提取结果，固定包含2个字段，格式：
                 {
                     "item_names": ["商品名1", "商品名2", ...],  # 提取的商品名列表，无则空列表
                     "rewritten_query": "改写后的完整问题"       # 包含商品名的独立问题，无则返回原始query
                 }
        """
        try:
            # 1、先获取llm客户端
            logger.info("步骤4：正在初始化 LLM 客户端...")
            client = get_llm_client(json_mode=True)
            # 2、构造历史对话文本，拼接为"角色: 内容"的格式，供LLM做上下文理解
            history_text = ""
            for msg in history:
                history_text += f"{msg['role']}: {msg['text']}\n"
            logger.info(f"步骤4： 历史上下文准备完成 (长度: {len(history_text)})")
            # 3、处理和动态拼接提示词
            prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
            logger.info(f"步骤4： 提示词加载成功")
            # 4、构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）

            messages = [
                SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
                HumanMessage(content=prompt)
            ]
            # 5、调用LLM客户端，发起请求获取提取结果
            logger.info("步骤4：正在调用 LLM...")
            response = client.invoke(messages)
            logger.info("步骤4：收到 LLM 响应：", response)
            # 6、提取响应中的文本内容
            content = response.content
            # 7、数据清洗：处理LLM可能返回的代码块格式（如```json ... ```），去除包裹符
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            # 8、数据解析：将JSON字符串转为字典
            result = json.loads(content)
            logger.info(f"步骤4： 解析 LLM 结果: {result}")
            # 9、健壮性处理
            # 确保返回结果包含item_names字段，无则设为空列表
            if "item_names" not in result:
                result["item_names"] = []
            # 确保返回结果包含rewritten_query字段，无则复用原始查询
            if "rewritten_query" not in result:
                result["rewritten_query"] = query
            # 10、返回解析后的提取结果
            return result

        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.error(f"步骤4： LLM 提取失败: {e}")
            # 异常时返回默认结果：空商品名列表+原始查询
            return {"item_names": [], "rewritten_query": query}

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """
           把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
           :param item_names: 列表[字符串] - 步骤4中 提取的商品名列表（如["苹果15", "华为P60"]）
           :return: 列表[字典] - 格式：
                [
                    {
                        "extracted_name": "提取的原始商品名",  # 如"苹果15"
                        "matches": [                          # 该商品名的TopN匹配结果，无则空列表
                            {
                                "item_name": "数据库中的商品名",  # Milvus中存储的标准化商品名
                                "score": 0.98                  # 混合搜索的相似度评分（0-1，越高越相似）
                            },
                            ...
                        ]
                    },
                    ...
                ]
        """
        logger.info(f"步骤5：开始向量化并查询条目: {item_names}")

        # 1、初始化最终返回结果列表，存储每个商品名的向量化查询结果
        results = []
        # 2、获取Milvus向量数据库的客户端连接对象
        client = get_milvus_client()
        # 3、校验Milvus客户端连接是否成功，失败则记录错误日志并返回空结果
        if not client:
            logger.error("连接 Milvus 失败")
            return results
        # 4、从环境变量中获取Milvus中存储商品名称向量的集合名（表名）
        collection_name = milvus_config.item_name_collection  # kb_item_names
        # 5、对所有商品名称批量生成BGEM3向量（稠密+稀疏），相比逐个生成提升处理效率
        # embeddings格式：{"dense": [向量1, 向量2,...], "sparse": [向量1, 向量2,...]}
        logger.info("步骤5：正在生成向量...")
        embeddings = generate_embeddings(item_names)
        logger.info(f"步骤5：已生成 {len(item_names)} 个商品名的向量。开始 Milvus 搜索...")
        # 6、遍历每个商品名称，逐个执行向量搜索（保证结果与原始商品名一一对应）
        for i in range(len(item_names)):
            try:
                logger.info(f"步骤5：正在处理商品 {i + 1}/{len(item_names)}: {item_names[i]}")
                # 从批量生成的向量结果中，取出当前商品名对应的稠密向量（高维连续值，如[0.12, 0.35,...]）
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]
                # 构造Milvus混合搜索请求对象，传入稠/稀疏向量，指定返回Top5匹配结果
                # reqs返回格式：[稠密向量搜索请求, 稀疏向量搜索请求]
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )


            except:
