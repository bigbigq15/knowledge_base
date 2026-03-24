import base64
import os
from collections import deque
from pathlib import Path
from typing import Tuple, List, re, Dict

from langchain_core.exceptions import LangChainException
from langchain_core.messages import HumanMessage

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.core.load_prompt import load_prompt
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
class NodeMdImg(NodeBase):
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_md_img"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        MD文件图片处理核心节点 - 五步法完成图片全流程处理
        核心流程：
        1. 获取MD内容、文件路径、图片文件夹路径
        2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
        3. 调用多模态大模型为图片生成内容摘要
        4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
        5. 备份原MD文件，保存处理后的新MD文件并更新状态
        :param state: 导入流程全局状态对象，包含task_id、md_path、md_content等核心参数
        :return: 更新后的全局状态对象（md_content、md_path为处理后新值）
        """

        # 步骤1：初始化数据，获取MD核心信息
        md_content, path_obj, images_dir = self._step_1_get_content(state)

        # 无图片文件夹，直接跳过所有图片处理逻辑
        if not images_dir.exists():
            logger.info(f"图片文件夹不存在，跳过图片处理：{images_dir.absolute()}")
            return state

        # 初始化MinIO客户端，失败则终止流程
        minio_client = get_minio_client()
        if not minio_client:
            logger.warning("MinIO客户端初始化失败，已跳过图片处理全流程")
            return state

        # 步骤2：扫描并筛选MD中引用的支持格式图片
        targets = self._step_2_scan_images(md_content, images_dir)
        if not targets:
            logger.info("未检测到MD中引用的支持格式图片，跳过后续处理")
            return state

        # 步骤3：调用多模态大模型生成图片摘要
        summaries = self._step_3_generate_summaries(path_obj.stem, targets)

        # 步骤4：上传图片至MinIO，替换MD图片路径并填充摘要
        new_md_content = self._step_4_upload_and_replace(minio_client, path_obj.stem, targets, summaries, md_content)
        state["md_content"] = new_md_content

        # 步骤5：备份并保存新MD文件，更新状态中的文件路径
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)
        state["md_path"] = new_md_file_name
        logger.info(f"MD图片处理完成，新文件已保存：{new_md_file_name}")


        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state



    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path] :
        md_path = state.get("md_path","")
        if not md_path:
            raise FileNotFoundError(f"全局状态中无有效MD文件路径：{state['md_path']}")
        path_obj = Path(md_path)

        md_content = state.get("md_content","")
        if not md_content:
            with open(path_obj,"r",encoding="utf-8") as f:
                md_content = f.read()
            logger.info(f"从文件读取MD内容完成，文件大小：{len(md_content)} 字符")
        else:
            logger.info(f"从全局状态获取MD内容完成，内容大小：{len(md_content)} 字符")
        images_dir = path_obj.parent / "images"
        return md_content, path_obj, images_dir

        # 步骤2：扫描图片文件夹，筛选MD中实际引用的支持格式图片

    def _step_2_scan_images(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
        扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
        :param md_content: MD文件完整内容
        :param images_dir: 图片文件夹路径对象
        :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
        """
        targets = []
        # 遍历图片文件夹所有文件
        for image_file in os.listdir(images_dir):
            # 过滤非支持格式的图片
            if not self._is_supported_image(image_file):
                logger.debug(f"图片格式不支持，跳过：{image_file}")
                continue

            # 组装图片完整路径
            img_path = str(images_dir / image_file)
            # 查找图片在MD中的引用上下文
            context_list = self._find_image_in_md(md_content, image_file)

            # 过滤MD中未引用的图片
            if not context_list:
                logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
                continue

            # 组装待处理图片元数据，取第一个匹配的上下文
            #image_file为图片目录路径,img_path为图片绝对路径带后缀,context_list为图片在md文档的相关上下文
            targets.append((image_file, img_path, context_list[0]))
            logger.info(f"图片加入待处理列表：{image_file}")

        logger.info(f"图片扫描完成，共筛选出待处理图片：{len(targets)} 张")
        return targets

    def _is_supported_image(self, filename: str) -> bool:
        """
        判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
        :param filename: 文件名（含后缀）
        :return: 支持返回True，否则False
        """
        return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

    def _find_image_in_md(self, md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
        """
        查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
        :param md_content: MD文件完整内容
        :param image_filename: 图片文件名（含后缀）
        :param context_len: 上下文截取长度，默认前后各100字符
        :return: 上下文列表，每个元素为(上文, 下文)元组，无匹配则返回空列表
        """

        # 匹配以下内容：
        # ![描述](http://images/图片名称.扩展名?size=100)
        # re.escape 转义图片文件名中的特殊字符，避免正则语法错误
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
        results = []

        # 迭代查找所有MD图片标签匹配项
        for m in pattern.finditer(md_content):
            start, end = m.span()
            # 截取匹配位置的上文和下文（防止索引越界）
            pre_text = md_content[max(0, start - context_len):start]
            post_text = md_content[end:min(len(md_content), end + context_len)]
            # 打印图片上下文，便于调试
            logger.debug(f"图片[{image_filename}]匹配到引用，上文：{pre_text.strip()}")
            logger.debug(f"图片[{image_filename}]匹配到引用，下文：{post_text.strip()}")
            results.append((pre_text, post_text))

        if not results:
            logger.info(f"MD内容中未找到图片[{image_filename}]的引用")
        return results

    def _step_3_generate_summaries(self, doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                                  requests_per_minute: int = 9) -> Dict[str, str]:
        summaries={}
        request_times = deque()
        for img_file, image_path, context in targets:
            apply_api_rate_limit(request_times, requests_per_minute, window_seconds=60)
            logger.info(f"开始生成图片摘要：{image_path}")
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)

        logger.info(f"图片摘要批量生成完成，共处理{len(summaries)}张图片")
        return summaries

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        """
        调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
        生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
        :param image_path: 图片本地完整路径
        :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
        :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
        :return: 图片内容摘要（异常时返回默认值"图片描述"）
        """
        base64_image = self._encode_image_to_base64(image_path)
        try:
            lvm_client = get_llm_client(model=lm_config.lv_model)
            prompt_text = load_prompt(
                name="image_summary",  # 提示词文件名（不带.prompt）
                root_folder=root_folder,  # 对应{root_folder}
                image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
            )
            messages = [
                HumanMessage(
                    content=[
                        # 文本提示词：携带上下文，限定摘要规则
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        # 多模态核心：Base64编码图片数据
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                )
            ]
            response = lvm_client.invoke(messages)
            summary = response.content.strip().replace("\n", "")
            logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")
            return summary
        except LangChainException as e:
            logger.error(f"图片摘要生成失败（LangChain框架异常）：{image_path}，错误信息：{str(e)}")
            return "图片描述"
        except Exception as e:
            logger.error(f"图片摘要生成失败（系统异常）：{image_path}，错误信息：{str(e)}")
            return "图片描述"

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        将本地图片文件编码为Base64字符串（用于多模态大模型输入）
        :param image_path: 图片本地完整路径
        :return: 图片的Base64编码字符串（UTF-8解码）
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在：{image_path}")

        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode("utf-8")
        logger.info(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
        return base64_str


