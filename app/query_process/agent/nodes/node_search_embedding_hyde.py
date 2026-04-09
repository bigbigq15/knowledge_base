from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState


class NodeSearchEmbeddingHyde(NodeBase):
    """
        节点功能：HyDE (Hypothetical Document Embedding)
        先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
        """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState):
        """
        HyDE (Hypothetical Document Embedding) 检索节点
        核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

        执行步骤：
        1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
        2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
        3. 混合检索 (Step 2)：
           - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
           - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
        4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

        :param state: 会话状态字典，包含 session_id, rewritten_query, item_names 等
        :return: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档) 的字典
        """

        # 1、用户问题和已确认商品名
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        try:
            # 2、生成假设性文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)
            # 3、用“重写问题 + 假设文档”检索切片
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names,
                top_k=5,
            )
            # 4、结果封装
            return {
                "hyde_embedding_chunks": res[0] if res else [],
                "hyde_doc": hyde_doc,
            }
        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            return {}

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
               阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
               HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
               从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。

               :param rewritten_query: 用户改写后的查询语句
               :return: LLM生成的假设性文档内容
               """

        logger.info("步骤1: 开始生成假设性文档")

        try:
            llm = get_llm_client()
            # 加载提示词模板，生成假设文档
            hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
            logger.debug(f"步骤1: Prompt加载成功")
            # 调用LLM生成
            response = llm.invoke(hyde_prompt)
            hyde_doc = response.content

            logger.info(f"步骤1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")
            logger.debug(f"步骤1: 文档预览: {hyde_doc}")

            return hyde_doc
        except Exception as e:
            logger.exception(f"步骤1: 生成假设文档失败: {e}")
            raise e

    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None,
            req_limit: int = 10,
            top_k: int = 5,
            ranker_weights=(0.8, 0.2),  # 调整默认权重以偏向稠密向量 (0.8, 0.2)
            norm_score: bool = True,    # 默认开启归一化
            output_fields=["chunk_id", "content", "item_name"],
    ):
        """
        阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。

        :param rewritten_query: 改写后的查询
        :param hyde_doc: Step 1 生成的假设性文档
        :param item_names: 商品名称列表，用于元数据过滤 (item_name in [...])
        :param req_limit: Milvus 搜索时的候选召回数量
        :param top_k: 最终返回的 Top K 结果数量
        :param ranker_weights: 混合检索权重 (Dense, Sparse)
        :param norm_score: 是否对分数进行归一化
        :param output_fields: 返回结果中包含的字段
        :return: 检索结果列表
        """
        try:
            # 1、拼接查询与假设文档，形成更丰富的语义上下文
            # 这里把用户问题 + 假设答案拼在一起生成向量，相当于：
            # 既保留了用户的原始意图（rewritten_query）
            # 又增强了语义丰富度（hyde_doc）
            combined_text = rewritten_query + " " + hyde_doc
            logger.info(f"步骤2: 拼接 Query + HyDE Doc, 总长度: {len(combined_text)}")

            # 2、生成向量 (Dense + Sparse)
            logger.info("步骤2: 正在生成混合向量 (Embedding)...")
            embeddings = generate_embeddings([combined_text])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]
            # 3. 获取Milvus的集合
            collection_name = milvus_config.chunks_collection
            logger.info(f"步骤2: 准备在集合 '{collection_name}' 中执行混合检索")
            # 4、处理 item_names 中的引号，防止注入或语法错误
            expr = None
            if item_names:
                quoted = ", ".join(f'"{v}"' for v in item_names)
                expr = f"item_name in [{quoted}]"
                logger.info(f"步骤2: 过滤条件: {expr}")
            else:
                logger.info("步骤2: 未指定商品名过滤，将全库检索")
            # 5、构造Milvus混合搜索请求对象
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=req_limit,
            )
            # 6、执行混合向量检索
            logger.info("步骤2: 开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=ranker_weights,
                norm_score=norm_score,
                limit=top_k,
                output_fields=list(output_fields),
            )
            return res

        except Exception as e:
            logger.error(f"步骤2: 检索过程发生异常: {e}")
            raise e