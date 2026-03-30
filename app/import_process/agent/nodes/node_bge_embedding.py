from typing import List, Dict

from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings


class NodeBgeEmbedding(NodeBase):
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        """
        LangGraph核心节点：BGE-M3文本向量化处理
        主流程（串行执行，全流程异常隔离）：
            1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用

        必要参数：task_id、chunks
        更新参数：chunks字段新增dense_vector/sparse_vector

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据校验
        texts_to_embed = self._step_1_validate_input(state)

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(texts_to_embed)

        # 步骤3：更新全局状态，将带向量的chunks回传下游
        state['chunks'] = output_data
        logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(output_data)} 条文本切片 ---")

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state

    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        向量化前置步骤1：输入数据有效性校验
        核心作用：
            1. 从全局状态提取待向量化的chunks切片列表
            2. 严格校验chunks类型和非空性，无有效数据则终止向量化
        参数：
            state: ImportGraphState - 流程全局状态对象
        返回：
            List[Dict] - 校验通过的文本切片列表
        异常：
            若chunks非列表/为空，抛出ValueError，终止当前向量化流程
        """

        # 1、从状态中提取切片数据
        texts_to_embed = state.get("chunks")

        # 2、校验：必须是非空列表，否则无法进行向量化
        if not isinstance(texts_to_embed, list) or not texts_to_embed:
            logger.error("向量化输入校验失败：chunks字段为空或非有效列表")
            raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")

        logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts_to_embed)}")
        return texts_to_embed


    def _step_2_generate_embeddings(self, texts_to_embed: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        向量化核心步骤2：批量生成稠密/稀疏双向量
        核心逻辑：
            1. 文本拼接：item_name（商品名）+ 换行 + content（切片内容）
            2. 批量调用：传入拼接后的文本，生成批量双向量
            3. 向量绑定：为每个切片复制原数据，新增dense_vector/sparse_vector字段
        参数：
            texts_to_embed: 文本切片列表
        返回：
            List[Dict[str, str]] - 带向量字段的文本切片列表
        关键配置：
            batch_size: 每批处理batch_size条，可根据服务器显存大小调整
        """

        # 初始化结果列表，存储带向量的切片数据
        output_data = []
        # 批次大小配置：平衡显存占用和处理效率，建议根据实际环境调整
        batch_size = 5

        # 按批次遍历，避免一次性处理过多数据导致显存溢出（OOM）
        total = len(texts_to_embed)
        for i in range(0, total, batch_size):
            # 截取当前批次的切片，最后一批自动适配剩余数量
            batch_texts = texts_to_embed[i:i + batch_size]
            # 计算当前批次的起止索引，用于日志展示
            start_idx, end_idx = i + 1, min(i + len(batch_texts), total)

            try:
                # 构造模型输入文本：拼接商品名+切片内容，增强核心特征
                input_texts = []
                for doc in batch_texts:
                    item_name = doc["item_name"]
                    content = doc["content"]
                    # 有商品名则拼接（换行分隔提升模型识别效率），无则直接使用内容
                    text = f"{item_name}\n{content}"
                    input_texts.append(text)

                # 调用封装函数生成批量向量，返回格式：{"dense": [稠密向量列表], "sparse": [稀疏向量列表]}
                docs_embeddings = generate_embeddings(input_texts)
                if not docs_embeddings:
                    error_msg = f"第{start_idx}-{end_idx}条切片：BGE-M3模型返回空结果，无法生成向量"
                    logger.exception(error_msg)
                    raise RuntimeError(error_msg)

                # 为当前批次每个切片绑定对应向量
                for j, doc in enumerate(batch_texts):
                    # 复制原数据避免修改state的值
                    item = doc.copy()
                    item["dense_vector"] = docs_embeddings["dense"][j]  # 绑定稠密向量
                    item["sparse_vector"] = docs_embeddings["sparse"][j]  # 绑定稀疏向量（已归一化）
                    output_data.append(item)

                logger.info(f"第{start_idx}-{end_idx}条切片：双向量生成成功")

            except Exception as e:
                # 捕获单批次所有异常，重新抛出并附加详细上下文信息
                error_msg = f"第{start_idx}-{end_idx}条切片：向量生成失败。错误原因：{str(e)}"
                logger.exception(error_msg)
                # 抛出异常，终止流程，避免产生缺失向量字段的脏数据
                raise RuntimeError(error_msg) from e #保留原始异常栈

        return output_data




