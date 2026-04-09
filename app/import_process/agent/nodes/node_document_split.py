import json
from pathlib import Path
from typing import Tuple, Any, List, Dict
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pycparser.c_ast import Continue

from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState, create_default_state

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500

class NodeDocumentSplit(NodeBase):
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→无标题兜底→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等
        :param state: 项目状态字典（ImportGraphState），必须包含md_content/task_id；可选local_dir/max_content_length/file_title
        :return: 更新后的状态字典，新增chunks键（存储最终处理后的Chunk列表，每个Chunk为含title/content/parent_title的字典）
        """
        # ===================================== 步骤1：加载并标准化输入数据 =====================================
        # 作用：从状态字典提取MD内容/文件标题/Chunk最大长度，统一换行符消除系统差异，做空值兜底
        # 输出：标准化后的md_content、文件标题、单个Chunk最大长度；无有效MD内容则直接终止节点执行.
        content, file_title, max_len = self._step_1_get_inputs(state)
        if content is None:
            logger.error(f"节点执行终止：无有效MD内容")
            return state

        # ===================================== 步骤2：按MD标题进行初次切分 =====================================
        # 作用：基于Markdown标题（#/##/###）切分文档为独立章节，自动跳过代码块内的伪标题，保证章节语义完整
        # 输出：初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # ===================================== 步骤3：无标题场景兜底处理 =====================================
        # 作用：解决MD文档无任何标题的边界情况，避免后续切分逻辑异常
        # 输出：有标题则返回步骤2的章节列表；无标题则将全文封装为单个「无标题」章节，保证数据格式统一
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # ===================================== 步骤4：Chunk精细化处理（长切短合） =====================================
        # 作用：核心切分逻辑，先将超长章节按「段落→句子」二次切分，再合并同父标题的过短章节，减少碎片化
        # 额外处理：对所有Chunk做parent_title兜底，适配Milvus向量库必填字段要求
        # 输出：长度适中、语义完整、低碎片化的最终Chunk列表（可直接用于向量入库/大模型调用）
        sections = self._step_4_refine_chunks(sections, max_len)

        # ===================================== 步骤5：输出文档切分统计信息 =====================================
        # 作用：打印核心统计数据，便于监控切分效果、调试问题（原始行数/最终Chunk数/首个Chunk预览）
        # 输出：无返回值，仅通过logger输出标准化统计日志
        self._step_5_print_stats(lines_count, sections)

        # ===================================== 步骤6：Chunk结果本地JSON备份 + 状态更新 =====================================
        # 作用1：将最终Chunk列表备份到local_dir目录的chunks.json，便于后续问题排查、数据复用
        # 作用2：将Chunk列表写入状态字典，传递给下一个节点（如向量入库、大模型摘要等）
        # 输出：状态字典新增chunks键；无local_dir则跳过备份，不影响主流程
        state["chunks"] = sections
        self._step_6_backup(state, sections)

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[Any, str, int]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题/单个Chunk最大长度（无内容则返回None,None,None）
        """
        # 从状态中提取MD原始内容
        content = state.get("md_content")
        # 空内容兜底：无MD内容则直接返回，终止后续处理
        if not content:
            logger.warning("状态字典中无有效MD内容，终止文档切分")
            return None, None, None

        # 基础标准化：统一换行符，避免Windows/Linux换行符差异导致的后续处理异常
        # 原始混合换行："# HL3070说明书\r\n## 产品概述\nHL3070是扫描枪\r\n\r\n### 操作步骤"
        # 统一后："# HL3070说明书\n## 产品概述\nHL3070是扫描枪\n\n### 操作步骤"
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        # 提取文件标题：有则用，无则默认"Unknown File"
        file_title = state.get("file_title", "Unknown File")
        # 提取最大Chunk长度：有则用状态中的配置，无则用全局默认值
        max_len = DEFAULT_MAX_CONTENT_LENGTH

        logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")
        return content, file_title, max_len

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        if not content:
            return [], 0, 0

        title_pattern = r'\s*#{1,6}\s+.+'

        lines = content.split("\n")
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        in_code_block = False  # 代码块标记：False当前没在代码块中，True当前在代码块中

        def _flush_section():
            sections.append({
                "title": current_title,
                # 每段时间使用 \n换行区分
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })
        for line in lines:
            if not line:
                continue
            stripped_line = line.strip()
            if stripped_line.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                _flush_section()
                current_title = stripped_line
                current_lines = [current_title]
                title_count += 1
                logger.info(f"识别标题：{current_title}")
            else:
                current_lines.append(line)
        _flush_section()
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(self, content: str, sections: List[Dict[str, str]], title_count: int,
                                file_title: str) -> List[Dict[str, str]]:
        """
        【步骤3】无标题兜底处理
            功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表 """

        if title_count == 0:
            logger.warning(f"未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
            sections = [{"title": "无标题", "content": content, "file_title": file_title}]

        logger.info(f"已识别到 {title_count} 个标题，文件：{file_title}")
        return sections

    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
        执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
        :param sections: 步骤3处理后的章节列表
        :return: 长度适中、低碎片化的最终Chunk列表
        """
        refined_split = []

        for sec in sections:
            # 超长chunk切分
            sub_sections = self._split_long_section(sec)

            refined_split.extend(sub_sections)

        final_sections = self._merge_short_sections(refined_split)

        return refined_split

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """

        #获取内容,判断内容的长度
        content = section.get("content","") or ""
        if len(content) <= DEFAULT_MAX_CONTENT_LENGTH:
            return [section]
        #获取章节标题
        title = section.get("title","") or ""
        #制造一个前缀,为标题\n\n 标题为空则为空字符串
        prefix = f"{title}\n\n" if title else ""
        #算出加上这个前缀的剩下的总长度
        available_len = DEFAULT_MAX_CONTENT_LENGTH - len(prefix)
        #判断标题长度是否太长了
        if available_len <= 0:
            logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
            return [section]
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title)+len(title):].lstrip()

        # 初始化LangChain递归分割器（核心工具：按优先级分隔符切分，保留语义）
        # separators：分割符优先级（从粗到细），优先按大语义单元切分，最后才硬拆
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,  # 正文部分最大长度（已扣除标题）
            chunk_overlap=0,  # 无重叠：按标题切分后语义完整，无需重叠
            # 分割符优先级：空行(段落)→换行→中文标点→英文标点→空格，最后硬拆
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],

        )
        sub_sections = []
        for idx ,chunk in enumerate(splitter.split_text(body),start = 1):
            text = chunk.strip()
            if not text:
                continue
            full_text = (prefix+text).strip()
            sub_sections.append({
                "title":f"{title}-{idx}"if title else f"chunk-{idx}",
                "content":full_text,
                "parent_title":title,
                "part":idx,
                "file_title":section.get("file_title")
            })
        logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")
        return sub_sections

    def _merge_short_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        【辅助函数】过短章节合并（减少碎片化，提升检索效果）
        核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，避免跨章节合并
        :param sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
        :return: 合并后的Chunk列表，长度适中，保留元信息
        """

        if not sections:
            logger.debug("待合并Chunk列表为空，直接返回")
            return []
        merged_sections = []
        current_chunk = None
        for sec in sections:
            if not current_chunk:
                current_chunk = sec
                continue

            # 合并条件：1.当前片段的长度不足阈值，2.和下一个片段是同一个父标题
            is_current_short = len(current_chunk["content"]) < MIN_CONTENT_LENGTH
            # current_chunk上一个片段，sec 当前片段
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                # 合并：去掉下一块片段开头重复的标题，避免内容冗余
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                     next_content = next_content[len(parent_title):].lstrip()

                current_chunk["content"] += "\n\n" + next_content

                if "part" in sec:
                    current_chunk["part"] = sec["part"]
                logger.debug(f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")
            else:
                merged_sections.append(current_chunk)
                current_chunk = sec

        if current_chunk is not None:
            merged_sections.append(current_chunk)
        logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")

        return merged_sections

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        logger.info(f"MD原始文本总行数：{lines_count}")
        logger.info(f"最终生成Chunk数量：{chunk_num}")
        if sections:
            first_title = sections[0].get("title", "无标题")
            logger.info(f"首个Chunk标题预览：{first_title}")
        logger.info("-" * 110)

    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        :param state: 项目状态字典，需包含md_dir（备份目录）
        :param sections: 最终处理后的Chunk列表
        """

        try:
            # 拼接备份文件路径：固定文件名，便于查找
            backup_path = Path(state["md_path"]).parent / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                """
                sections是Python 嵌套数据结构（List[Dict[str, str]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
                （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
                同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
                """
                json.dump(
                    sections,
                    f,
                    #开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    #开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2             # 格式化缩进，便于阅读
                )
            logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程
            logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)



if __name__ == "__main__":

    import os
    # 获取项目所在路径
    from app.utils.path_util import PROJECT_ROOT

    # 组装文件路径
    md_name= os.path.join("output/hak180产品安全手册", "hak180产品安全手册_new.md")
    # 组装文件的绝对路径
    md_path = os.path.join(PROJECT_ROOT, md_name)
    md_content = Path(md_path).read_text(encoding="utf-8")
    # 当前节点图状态初始值
    init_state = create_default_state(
        task_id="task_001",
        md_path=md_path,
        md_content=md_content,
        file_title="hak180产品安全手册"
    )
    # 执行节点的业务调用
    node_document_split = NodeDocumentSplit()
    final_state = node_document_split(init_state)

