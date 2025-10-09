# main.py

# 导入所需的标准库和框架组件
import json
import re
from collections import deque, defaultdict, OrderedDict
from typing import Tuple, List, Optional
import asyncio
# MODIFIED: 移除了 os 的导入，因为路径操作将由 pathlib 处理
from dataclasses import dataclass, field

# 导入 AstrBot 框架的核心组件
from astrbot.api.event import filter as event_filter, AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
# MODIFIED: 从 astrbot.api.star 导入了 StarTools
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import AstrBotConfig, logger

@dataclass
class ReviewCycleResult:
    """封装单次审查循环结果的数据类，使函数返回更结构化。"""
    is_passed: bool
    review_comment: str
    failed_categories: List[str] = field(default_factory=list)
    is_critical_error: bool = False

@register("astrbot_plugin_hapemxg_inspect", "hapemxg", "hapemxg的审查插件", "1.1.0")
class ReviewerPlugin(Star):
    """
    这是hapemxg的LLM回复审查插件。
    """
    # --- 使用常量管理JSON和字典中的键名，以提高代码的可维护性和减少硬编码错误 ---
    PLUGIN_ID = "ReviewerPlugin"
    POLITICAL = "political"
    PORNOGRAPHIC = "pornographic"
    VERBAL_ABUSE = "verbal_abuse"
    CUSTOM_RULE = "custom_rule"
    FAIL = "FAIL"
    PASS = "PASS"
    REASON_SUFFIX = "_reason"

    # --- 配置键名常量化 ---
    class CfgKeys:
        ENABLED = "enabled"
        ENABLE_CONSOLE_LOGGING = "enable_console_logging"
        REVIEWER_PROVIDER_ID = "reviewer_provider_id"
        DEEP_REVIEW_CONTEXT_DEPTH = "deep_review_context_depth"
        HISTORY_SAVE_INTERVAL = "history_save_interval_seconds"
        RETRY_SETTINGS = "retry_settings"
        MAX_RETRIES = "max_retries"
        SEND_FAILURE_MESSAGE = "send_failure_message_on_max_retries"
        STRICT_REVIEW_MODE = "strict_review_mode"
        FAST_REVIEW_SETTINGS = "fast_review_settings"
        ENABLE_FAST_REVIEW = "enable_fast_review"
        WHITELIST_REGEX = "whitelist_regex"
        WHITELIST_PROMPTS = "whitelist_prompts"
        BLACKLIST_REGEX = "blacklist_regex"
        BLACKLIST_PROMPTS = "blacklist_prompts"
        USER_MESSAGES = "user_facing_messages"
        FINAL_FAILURE_MESSAGE = "final_failure"
        INTERNAL_ERROR_MESSAGE = "internal_error"
        RETRY_SYSTEM_PROMPT = "retry_system_prompt"
        RETRY_INSTRUCTION_PROMPT = "retry_instruction_prompt"
        RETRY_GUIDANCE = "retry_guidance"
        REVIEW_LEVELS = "review_levels"
        REVIEWER_SYSTEM_PROMPT = "reviewer_system_prompt"
        ENABLE_POLITICAL = "enable_political"
        POLITICAL_RULE_PROMPT = "political_rule_prompt"
        ENABLE_PORNOGRAPHIC = "enable_pornographic"
        PORNOGRAPHIC_RULE_PROMPT = "pornographic_rule_prompt"
        ENABLE_VERBAL_ABUSE = "enable_verbal_abuse"
        VERBAL_ABUSE_RULE_PROMPT = "verbal_abuse_rule_prompt"
        ENABLE_CUSTOM_RULE = "enable_custom_rule"
        CUSTOM_RULE_PROMPT = "custom_rule_prompt"

    def __init__(self, context: Context, config: AstrBotConfig):
        """
        插件初始化方法。
        在AstrBot加载插件时被调用，负责加载配置、初始化内部状态和启动后台任务。
        """
        super().__init__(context)
        self.config = config
        
        # 审查提供商实例，采用懒加载模式，在首次需要时再获取
        self.reviewer_provider = None

        # --- 持久化与历史记录设置 ---
        # MODIFIED: 使用 StarTools 获取规范的数据目录，不再硬编码路径
        data_root = StarTools.get_data_dir(self.__class__.PLUGIN_ID)
        data_root.mkdir(parents=True, exist_ok=True)
        # MODIFIED: 使用 pathlib 的 / 操作符进行路径拼接
        self.history_file_path = data_root / "approved_history.json"
        self.history_lock = asyncio.Lock()  # 异步锁，确保文件写入操作的原子性
        self.history_dirty = False  # 脏位，标记历史记录是否被修改，用于优化保存性能
        
        # --- 模块化加载配置 ---
        # 将不同逻辑块的配置加载过程封装到独立的函数中，使初始化逻辑更清晰
        self._load_core_settings()
        self._load_retry_settings()
        self._load_fast_review_settings()
        self._load_deep_review_settings()
        self._load_user_messages_and_prompts()
        
        # 在所有配置加载完毕后，进行一次最终的逻辑校验
        self._validate_plugin_state()
        
        # 从本地文件加载历史记录
        self.approved_history = self._load_history()

        # --- 启动后台任务 ---
        # 创建一个后台异步任务，用于周期性地将内存中的历史记录保存到磁盘
        self.save_task = asyncio.create_task(self._periodic_save_task())
        logger.info(f"[ReviewerPlugin] 后台历史保存任务已启动，保存间隔: {self.save_interval_seconds} 秒。")

    async def terminate(self):
        """
        插件终止方法。
        在AstrBot卸载或停止插件时调用，用于执行清理工作，确保资源被正确释放，数据被完整保存。
        """
        logger.info("[ReviewerPlugin] 插件正在终止，准备执行最终的历史记录保存...")
        self.save_task.cancel()  # 取消正在运行的后台保存任务
        try:
            await self.save_task
        except asyncio.CancelledError:
            logger.info("[ReviewerPlugin] 后台保存任务已成功取消。")
        
        # 执行最后一次保存操作，确保从上次保存到关闭前的所有变更都被写入磁盘
        if self.history_dirty:
            await self._save_history()
            logger.info("[ReviewerPlugin] 最终的历史记录已成功保存到磁盘。")
        else:
            logger.info("[ReviewerPlugin] 历史记录无变化，无需执行最终保存。")

    # --- 历史记录持久化辅助方法 ---
    def _load_history(self) -> defaultdict:
        """从JSON文件中加载已通过审查的对话历史。"""
        try:
            with open(self.history_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 将加载的数据转换为 `defaultdict` of `deque` 结构，与内存中的格式保持一致
                history = defaultdict(
                    lambda: deque(maxlen=self.deep_review_context_depth),
                    {key: deque(value, maxlen=self.deep_review_context_depth) for key, value in data.items()}
                )
                logger.info(f"[ReviewerPlugin] 从 '{self.history_file_path}' 成功加载 {len(history)} 条会话历史。")
                return history
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或格式错误，则创建一个新的空历史记录对象，不影响程序运行
            logger.info(f"[ReviewerPlugin] 未找到历史文件或文件格式错误，将创建新的历史记录。")
            return defaultdict(lambda: deque(maxlen=self.deep_review_context_depth))

    async def _save_history(self):
        """将内存中的对话历史异步保存到JSON文件。"""
        if not self.history_dirty: return  # 如果数据未变，则跳过保存
        async with self.history_lock:
            try:
                # 将deque转换为list以便进行JSON序列化
                history_to_save = {key: list(value) for key, value in self.approved_history.items()}
                with open(self.history_file_path, 'w', encoding='utf-8') as f:
                    json.dump(history_to_save, f, ensure_ascii=False, indent=4)
                self.history_dirty = False  # 保存成功后，重置脏位
                if self.log_to_console:
                    logger.info(f"[ReviewerPlugin] 历史记录已成功保存到 '{self.history_file_path}'。")
            except Exception as e:
                logger.error(f"[ReviewerPlugin] 保存历史记录到 '{self.history_file_path}' 失败: {e}", exc_info=True)

    async def _periodic_save_task(self):
        """作为后台任务运行的守护进程，周期性地检查并保存历史记录。"""
        while True:
            try:
                await asyncio.sleep(self.save_interval_seconds)
                if self.history_dirty:
                    if self.log_to_console:
                        logger.info("[ReviewerPlugin] 检测到历史记录变更，执行定期保存...")
                    await self._save_history()
            except asyncio.CancelledError:
                # 当 `terminate` 方法取消任务时，会进入此分支，正常退出循环
                break
            except Exception as e:
                # 捕获其他未知异常，防止后台任务意外崩溃
                logger.error(f"[ReviewerPlugin] 后台保存任务出现严重错误: {e}", exc_info=True)
                await asyncio.sleep(60) # 发生错误后等待一段时间再重试

    # --- 配置加载辅助方法 ---
    def _load_core_settings(self):
        """加载插件的核心与基础配置。"""
        self.enabled = self.config.get(self.CfgKeys.ENABLED, False)
        self.log_to_console = self.config.get(self.CfgKeys.ENABLE_CONSOLE_LOGGING, True)
        self.reviewer_provider_id = self.config.get(self.CfgKeys.REVIEWER_PROVIDER_ID, "")
        self.deep_review_context_depth = self.config.get(self.CfgKeys.DEEP_REVIEW_CONTEXT_DEPTH, 2)
        self.save_interval_seconds = self.config.get(self.CfgKeys.HISTORY_SAVE_INTERVAL, 300)

    def _load_retry_settings(self):
        """加载与审查失败后重试相关的配置。"""
        retry_settings = self.config.get(self.CfgKeys.RETRY_SETTINGS, {})
        self.max_attempts = retry_settings.get(self.CfgKeys.MAX_RETRIES, 3) + 1  # +1 是因为包含首次尝试
        self.send_failure_message = retry_settings.get(self.CfgKeys.SEND_FAILURE_MESSAGE, True)
        self.strict_mode = retry_settings.get(self.CfgKeys.STRICT_REVIEW_MODE, True)

    def _load_fast_review_settings(self):
        """加载快速审查模块（基于正则表达式）的配置。"""
        fast_review_config = self.config.get(self.CfgKeys.FAST_REVIEW_SETTINGS, {})
        self.fast_review_enabled = fast_review_config.get(self.CfgKeys.ENABLE_FAST_REVIEW, False)
        self.whitelist_rules = self._parse_key_value_config(fast_review_config.get(self.CfgKeys.WHITELIST_REGEX, ""), fast_review_config.get(self.CfgKeys.WHITELIST_PROMPTS, ""))
        self.blacklist_rules = self._parse_key_value_config(fast_review_config.get(self.CfgKeys.BLACKLIST_REGEX, ""), fast_review_config.get(self.CfgKeys.BLACKLIST_PROMPTS, ""))
        if self.fast_review_enabled and self.log_to_console:
            logger.info(f"[ReviewerPlugin] 【快速审查】模块已启用。加载了 {len(self.whitelist_rules)} 条白名单规则和 {len(self.blacklist_rules)} 条黑名单规则。")

    def _load_deep_review_settings(self):
        """加载深度审查模块（基于LLM）的配置，并构建系统提示词。"""
        if not self.reviewer_provider_id:
            logger.info("[ReviewerPlugin] 未配置 'reviewer_provider_id'，【深度审查】功能将不会激活。")
            return
        logger.info(f"[ReviewerPlugin] 【深度审查】功能已配置，将使用模型: '{self.reviewer_provider_id}' (运行时检查可用性)。")
        self._build_reviewer_system_prompt()

    def _load_user_messages_and_prompts(self):
        """加载面向用户的提示信息以及用于指导模型重试的提示模板。"""
        messages_config = self.config.get(self.CfgKeys.USER_MESSAGES, {})
        self.final_failure_message = messages_config.get(self.CfgKeys.FINAL_FAILURE_MESSAGE, "抱歉，我多次尝试后仍无法生成完全合规的回复，为安全起见，本次回答已取消。")
        self.internal_error_message = messages_config.get(self.CfgKeys.INTERNAL_ERROR_MESSAGE, "抱歉，内容审查服务暂时出现问题，本次回答已取消。")
        self.retry_system_prompt = self.config.get(self.CfgKeys.RETRY_SYSTEM_PROMPT, "")
        self.retry_instruction_prompt = self.config.get(self.CfgKeys.RETRY_INSTRUCTION_PROMPT, "")
        self.retry_guidance = self.config.get(self.CfgKeys.RETRY_GUIDANCE, {})

    def _build_reviewer_system_prompt(self):
        """根据配置文件，动态地构建用于深度审查模型的最终系统提示词。"""
        review_levels_config = self.config.get(self.CfgKeys.REVIEW_LEVELS, {})
        self.review_configs = {
            self.POLITICAL: {"enabled": review_levels_config.get(self.CfgKeys.ENABLE_POLITICAL, True), "name": "政治敏感"},
            self.PORNOGRAPHIC: {"enabled": review_levels_config.get(self.CfgKeys.ENABLE_PORNOGRAPHIC, True), "name": "色情低俗"},
            self.VERBAL_ABUSE: {"enabled": review_levels_config.get(self.CfgKeys.ENABLE_VERBAL_ABUSE, True), "name": "言语辱骂"},
            self.CUSTOM_RULE: {"enabled": review_levels_config.get(self.CfgKeys.ENABLE_CUSTOM_RULE, False), "name": "自定义规则"},
        }
        base_prompt = self.config.get(self.CfgKeys.REVIEWER_SYSTEM_PROMPT, "")
        
        # 定义占位符与对应配置内容的映射关系
        replacements = {
            "{political_rules_block}": review_levels_config.get(self.CfgKeys.POLITICAL_RULE_PROMPT, "") if self.review_configs[self.POLITICAL]["enabled"] else "",
            "{pornographic_rules_block}": review_levels_config.get(self.CfgKeys.PORNOGRAPHIC_RULE_PROMPT, "") if self.review_configs[self.PORNOGRAPHIC]["enabled"] else "",
            "{verbal_abuse_rules_block}": review_levels_config.get(self.CfgKeys.VERBAL_ABUSE_RULE_PROMPT, "") if self.review_configs[self.VERBAL_ABUSE]["enabled"] else "",
            "{custom_rules_block}": review_levels_config.get(self.CfgKeys.CUSTOM_RULE_PROMPT, "") if self.review_configs[self.CUSTOM_RULE]["enabled"] else ""
        }
        
        # 遍历映射，将模板中的占位符替换为实际的规则文本
        final_prompt = base_prompt
        for placeholder, value in replacements.items(): final_prompt = final_prompt.replace(placeholder, value)
        self.final_reviewer_system_prompt = final_prompt.strip()

    def _validate_plugin_state(self):
        """在所有配置加载后，检查是否存在逻辑冲突或无效配置，并发出警告。"""
        if self.enabled and not self.fast_review_enabled and not self.reviewer_provider_id:
            logger.warning("[ReviewerPlugin] 插件主开关已开启，但【快速审查】未启用且未配置审查模型ID，插件将不执行任何审查操作。")

    # --- 核心逻辑辅助方法 ---
    def _parse_key_value_config(self, regex_str: str, prompt_str: str) -> 'OrderedDict':
        """解析配置文件中“键: 值”格式的多行文本，并编译正则表达式。"""
        rules = OrderedDict()
        line_pattern = re.compile(r"^\s*([^:]+?)\s*(.*)$")
        def parse_to_map(text: str) -> dict:
            parsed_map = {}
            for line in text.strip().split('\n'):
                match = line_pattern.match(line)
                if match:
                    key, value = match.groups()
                    if key and value: parsed_map[key.strip()] = value.strip()
            return parsed_map
        regex_map, prompt_map = parse_to_map(regex_str), parse_to_map(prompt_str)
        for key, regex_pattern in regex_map.items():
            if key in prompt_map:
                try: rules[key] = {"regex": re.compile(regex_pattern), "prompt": prompt_map[key]}
                except re.error as e: logger.error(f"[ReviewerPlugin] 快速审查规则 '{key}' 的正则表达式编译失败: {e}")
        return rules

    def _perform_fast_review(self, text: str) -> Tuple[bool, str]:
        """执行快速审查，依次检查黑名单和白名单规则。"""
        # 检查黑名单，任何一条命中则立即失败
        for rule_name, rule_data in self.blacklist_rules.items():
            match = rule_data["regex"].search(text)
            if match:
                trigger_word = match.group(0)
                final_prompt = rule_data["prompt"].format(trigger_word=trigger_word)
                if self.log_to_console: logger.warning(f"[ReviewerPlugin] 快速审查命中黑名单规则: '{rule_name}' (触发词: '{trigger_word}')")
                return False, final_prompt
        
        # 检查白名单，必须全部满足才算通过
        failed_prompts = [rule_data["prompt"] for rule_name, rule_data in self.whitelist_rules.items() if not rule_data["regex"].search(text)]
        if failed_prompts:
            if self.log_to_console: logger.warning(f"[ReviewerPlugin] 快速审查未满足 {len(failed_prompts)} 条白名单规则。")
            return False, "\n".join(failed_prompts)
        return True, ""

    def _parse_review_result(self, text: str) -> dict:
        """解析审查模型返回的文本，提取审查结果。优先尝试解析JSON，失败则回退到正则。"""
        try:
            # 优先尝试解析标准的JSON格式，移除可能的代码块标记
            cleaned_text = re.sub(r'```json\s*|\s*```', '', text).strip()
            data = json.loads(cleaned_text)
            result = {key: value.get("result", self.PASS).upper() for key, value in data.items() if isinstance(value, dict)}
            result.update({f"{key}{self.REASON_SUFFIX}": value.get("reason", "") for key, value in data.items() if isinstance(value, dict)})
            return result
        except (json.JSONDecodeError, TypeError):
            # 如果JSON解析失败，启动备用方案：正则表达式解析
            if self.log_to_console: logger.warning("[ReviewerPlugin] 审查模型返回的不是标准JSON，尝试使用正则解析。")
            result = {}
            if not self.reviewer_provider_id: return result
            for key, config in self.review_configs.items():
                if not config["enabled"]: continue
                name = config["name"]
                # 匹配 "类别: <PASS/FAIL>"
                pattern = re.search(f"{re.escape(name)}[:：]\\s*<?(PASS|FAIL)>?", text, re.IGNORECASE)
                if pattern: result[key] = pattern.group(1).strip().upper()
                # 匹配 "类别理由: [具体理由]" (优化后的正则，对空行更具鲁棒性)
                reason_pattern = re.search(f"{re.escape(name)}理由[:：]([\\s\\S]*?)(?=\\n\\s*\\S+[:：]|$)", text)
                if reason_pattern: result[f"{key}{self.REASON_SUFFIX}"] = reason_pattern.group(1).strip()
            return result

    async def _perform_single_review_cycle(self, text_to_review: str, user_prompt: str, user_session_key: str) -> ReviewCycleResult:
        """
        执行一个完整的、单次的审查流程（包括快速和深度审查）。
        
        Args:
            text_to_review (str): 当前待审查的LLM回复文本。
            user_prompt (str): 用户本轮的输入，用于深度审查的上下文。
            user_session_key (str): 用户的唯一会话标识。

        Returns:
            ReviewCycleResult: 包含审查结果的结构化对象。
        """
        # --- 阶段一：快速审查 ---
        if self.fast_review_enabled:
            passed, reason = self._perform_fast_review(text_to_review)
            if not passed:
                return ReviewCycleResult(is_passed=False, review_comment=reason, failed_categories=["快速审查"])
        
        # 如果未配置深度审查，则到此为止，直接通过
        if not self.reviewer_provider_id:
            return ReviewCycleResult(is_passed=True, review_comment="")

        # --- 阶段二：深度审查（懒加载审查模型） ---
        if not self.reviewer_provider:
            self.reviewer_provider = self.context.get_provider_by_id(self.reviewer_provider_id)
            if not self.reviewer_provider:
                logger.warning(f"[ReviewerPlugin] 运行时无法找到ID为 '{self.reviewer_provider_id}' 的审查提供商，本次将跳过深度审查。")
                return ReviewCycleResult(is_passed=True, review_comment="")
        
        try:
            # 准备包含对话历史的、给审查模型的完整Prompt
            past_dialogue = list(self.approved_history[user_session_key])
            history_str = "".join(f"历史轮次 {i+1} - 用户: {q}\n历史轮次 {i+1} - 模型: {a}\n\n" for i, (q, a) in enumerate(past_dialogue)) if past_dialogue else "(无历史对话)\n\n"
            review_prompt = (f"...\n--- 对话历史参考 ---\n{history_str}--- 当前待审查的对话 ---\n用户的最新提问: {user_prompt}\n待审查的回复: {text_to_review}")
            if self.log_to_console: logger.info(f"[ReviewerPlugin] 【深度审查】请求:\n---\n{review_prompt}\n---")

            # 调用审查模型
            review_resp = await self.reviewer_provider.text_chat(prompt=review_prompt, session_id=f"reviewer_{user_session_key}", system_prompt=self.final_reviewer_system_prompt)
            review_text = review_resp.completion_text.strip()
            if self.log_to_console: logger.info(f"[ReviewerPlugin] 【深度审查】模型判断:\n---\n{review_text}\n---")
            
            # 解析审查结果
            parsed = self._parse_review_result(review_text)
            if not parsed:
                is_passed = not self.strict_mode
                return ReviewCycleResult(is_passed=is_passed, review_comment="审查模型返回格式无法解析", failed_categories=["审查模型解析失败"], is_critical_error=not is_passed)
            
            # 检查是否有失败的审查项
            failed_categories = [key for key, cfg in self.review_configs.items() if cfg["enabled"] and parsed.get(key) == self.FAIL]
            if failed_categories:
                # 组合重试指导意见
                guidance = "\n".join(filter(None, [self.retry_guidance.get(k, "").format(reason=parsed.get(f"{k}{self.REASON_SUFFIX}", "未提供具体理由")) for k in failed_categories]))
                failed_names = [self.review_configs[key]["name"] for key in failed_categories]
                return ReviewCycleResult(is_passed=False, review_comment=guidance, failed_categories=failed_names)
        
        except Exception as e:
            # 捕获调用审查模型过程中的任何异常
            logger.error(f"[ReviewerPlugin] 调用审查模型时发生严重错误: {e}", exc_info=True)
            is_passed = not self.strict_mode
            return ReviewCycleResult(is_passed=is_passed, review_comment=self.internal_error_message, failed_categories=["审查模型调用失败"], is_critical_error=not is_passed)
        
        # 所有审查项均通过
        return ReviewCycleResult(is_passed=True, review_comment="")

    async def _regenerate_response(self, original_request: ProviderRequest, session_id: str, failed_response: str, review_comment: str, user_prompt: str) -> str:
        """
        当审查失败时，调用主LLM以重新生成回复。
        
        Args:
            original_request (ProviderRequest): 最初触发LLM的请求对象。
            session_id (str): 当前会话ID。
            failed_response (str): 上一轮未通过审查的回复文本。
            review_comment (str): 审查后生成的指导意见。
            user_prompt (str): 用户本轮的原始输入。

        Returns:
            str: 主LLM新生成的回复文本。
        """
        main_provider = self.context.get_using_provider()
        # 构建用于重试的、包含指导意见的新Prompt
        retry_instruction = self.retry_instruction_prompt.format(review_comment=review_comment, failed_response=failed_response, user_prompt=user_prompt)
        
        # 构建重试时的上下文，将失败的回复作为一轮对话历史添加进去
        retry_contexts = original_request.contexts.copy()
        retry_contexts.append({"role": "assistant", "content": failed_response})
        
        # 确定重试时使用的系统提示词，优先使用专为重试配置的，否则回退到原始请求的
        final_retry_system_prompt = self.retry_system_prompt or original_request.system_prompt
        
        # 调用主模型进行重试
        retry_response = await main_provider.text_chat(prompt=retry_instruction, session_id=session_id, contexts=retry_contexts, system_prompt=final_retry_system_prompt)
        return retry_response.completion_text.strip()


    # --- AstrBot 事件钩子 ---
    @event_filter.on_llm_request(priority=100)
    async def store_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        事件钩子：在LLM请求发出前触发。
        此钩子的作用是“暂存”原始的LLM请求对象和用户输入文本，以便在后续的响应审查环节中使用。
        """
        if self.enabled:
            # 将原始请求对象和用户输入附加到事件对象上，方便后续钩子访问
            setattr(event, '_original_llm_request_for_reviewer', req)
            user_prompt_text = (event.message_str or "").strip()
            # 处理用户发送图片等多媒体消息的情况
            if not user_prompt_text and event.message_obj.message:
                final_user_prompt = "[用户发送了非文本消息]"
            else:
                final_user_prompt = user_prompt_text or req.prompt
            setattr(event, '_user_prompt_for_reviewer', final_user_prompt)

    @event_filter.on_llm_response(priority=10)
    async def review_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """
        事件钩子：在收到LLM响应后、发送给用户前触发。
        这是插件的核心流程控制器，负责驱动审查与重试的循环。
        """
        # --- 前置检查 ---
        if not self.enabled: return
        original_request = getattr(event, '_original_llm_request_for_reviewer', None)
        if not original_request: return
        
        # --- 初始化审查循环所需变量 ---
        user_prompt = getattr(event, '_user_prompt_for_reviewer', original_request.prompt)
        session_id = event.unified_msg_origin
        user_session_key = f"{session_id}_{event.get_sender_id()}"  # 唯一标识一个用户的会话
        current_response_text = resp.completion_text

        # --- 开始审查与重试循环 ---
        for attempt in range(1, self.max_attempts + 1):
            if self.log_to_console:
                logger.info(f"[ReviewerPlugin] (Session: {user_session_key}) 开始第 {attempt}/{self.max_attempts} 次审查...")
                logger.info(f"[ReviewerPlugin] 待审查内容:\n---\n{current_response_text}\n---")
            
            # 调用封装好的单次审查循环函数
            review_result = await self._perform_single_review_cycle(current_response_text, user_prompt, user_session_key)

            # 决策与执行
            if review_result.is_critical_error:
                # 如果发生严重错误（如审查模型调用失败且处于严格模式）
                await event.send(event.plain_result(self.internal_error_message))
                event.stop_event() # 终止事件，阻止后续所有流程
                return
            
            if review_result.is_passed:
                # 如果所有审查都通过
                if self.log_to_console: logger.info(f"[ReviewerPlugin] 第 {attempt} 次审查最终通过。")
                # 将通过的对话存入历史记录，并标记为脏数据
                self.approved_history[user_session_key].append((user_prompt, current_response_text))
                self.history_dirty = True
                # 更新原始响应对象的内容，然后正常返回，让AstrBot继续发送流程
                resp.completion_text = current_response_text
                return

            # 如果审查失败，记录日志并准备重试
            if self.log_to_console:
                logger.warning(f"[ReviewerPlugin] 第 {attempt} 次审查驳回。违反规则: {', '.join(review_result.failed_categories)}。")
                logger.warning(f"[ReviewerPlugin] 生成的指导意见:\n---\n{review_result.review_comment}\n---")
            
            # 检查是否达到最大重试次数
            if attempt >= self.max_attempts: break

            # --- 准备并执行重试 ---
            try:
                if self.log_to_console: logger.info(f"[ReviewerPlugin] 准备第 {attempt + 1} 次生成...")
                current_response_text = await self._regenerate_response(original_request, session_id, current_response_text, review_result.review_comment, user_prompt)
            except Exception as e:
                # 如果在重试过程中发生错误，则终止流程
                logger.error(f"[ReviewerPlugin] 重新生成回复时发生严重错误: {e}", exc_info=True)
                await event.send(event.plain_result(self.internal_error_message))
                event.stop_event()
                return
        
        # --- 循环结束仍未通过审查 ---
        logger.error(f"[ReviewerPlugin] 已达到最大尝试次数 ({self.max_attempts})，审查最终失败。")
        if self.send_failure_message:
            await event.send(event.plain_result(self.final_failure_message))
        event.stop_event() # 终止事件，阻止发送不合规的最终回复