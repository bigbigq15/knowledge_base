from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.state import QueryGraphState


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState):
        """
        核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
        流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
        :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                      {
                          "session_id": str,        # 会话唯一标识
                          "rewritten_query": str,   # step4改写后的完整用户问题（含商品名）
                          "item_names": list[str],  # step7已确认的标准化商品名列表
                          "is_stream": bool/None    # 是否为流式响应，可选
                      }
        :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
                 {
                     "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                     # 每个元素为一条匹配的向量数据，含业务字段
                 }
        """
        try:
            # 1、用户问题和已确认商品名
            query = state.get("rewritten_query")
            item_names =state.get("item_names")

            # 2、生成向量 (Dense + Sparse)
            logger.info("正在生成混合向量 (Embedding)...")
            embeddings = generate_embeddings([query])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]
            # 3. 获取Milvus的集合
            collection_name = milvus_config.chunks_collection
            logger.info(f"准备在集合 '{collection_name}' 中执行混合检索")
            # 4、处理 item_names 中的引号，防止注入或语法错误
            expr = None
            if item_names:
                quoted = ", ".join(f'"{v}"' for v in item_names)
                expr = f"item_name in [{quoted}]"
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")
            # 5、构造Milvus混合搜索请求对象
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=10
            )
            # 6、执行混合向量检索
            logger.info("开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client = client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8,0.2),
                norm_score=True,
                output_fields=["chunk_id", "content", "item_name"]
            )
            # 7、构造并返回结果：若检索结果非空，取res[0]，否则返回空列表
            logger.info(f"节点search_embedding处理成功 :{res}")
            return {"embedding_chunks": res[0] if res else []}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}