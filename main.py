# main.py

# 导入所需的标准库
import asyncio
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent

# 导入 AstrBot 框架的核心组件
from astrbot.api.event import filter as event_filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register


@dataclass
class ReviewCycleResult:
    """
    审查结果数据类。
    用于封装单次审查循环（快速审查+深度审查）的返回结果，使逻辑传递更清晰。
    """

    is_passed: bool
    review_comment: str
    failed_categories: list[str] = field(default_factory=list)
    is_critical_error: bool = False


@dataclass
class ReviewerConfig:
    """
    配置管理类。
    负责插件所有配置项的加载、解析、验证以及默认值处理。
    """

    # --- 配置键名常量 ---
    class CfgKeys:
        ENABLED = "enabled"
        ENABLE_CONSOLE_LOGGING = "enable_console_logging"
        # 白名单
        WHITELISTED_USER_IDS = "whitelisted_user_ids"
        # 审查模型配置
        REVIEWER_PROVIDER_ID = "reviewer_provider_id"
        REVIEWER_IS_MULTIMODAL = "reviewer_is_multimodal"
        DEEP_REVIEW_CONTEXT_DEPTH = "deep_review_context_depth"

        # 重试与备用模型配置
        RETRY_SETTINGS = "retry_settings"
        MAX_RETRIES = "max_retries"
        SEND_FAILURE_MESSAGE = "send_failure_message_on_max_retries"
        SEND_ERROR_MESSAGE = "send_error_message_on_internal_failure"
        STRICT_REVIEW_MODE = "strict_review_mode"
        FALLBACK_PROVIDER_ID = "fallback_provider_id"
        SWITCH_PROVIDER_AFTER = "switch_provider_after_retries"
        FALLBACK_IS_MULTIMODAL = "fallback_provider_is_multimodal"

        # 快速审查配置
        FAST_REVIEW_SETTINGS = "fast_review_settings"
        ENABLE_FAST_REVIEW = "enable_fast_review"
        WHITELIST_REGEX = "whitelist_regex"
        WHITELIST_PROMPTS = "whitelist_prompts"
        BLACKLIST_REGEX = "blacklist_regex"
        BLACKLIST_PROMPTS = "blacklist_prompts"

        # 提示语配置
        USER_MESSAGES = "user_facing_messages"
        FINAL_FAILURE_MESSAGE = "final_failure"
        INTERNAL_ERROR_MESSAGE = "internal_error"
        RETRY_SYSTEM_PROMPT = "retry_system_prompt"
        RETRY_INSTRUCTION_PROMPT = "retry_instruction_prompt"
        RETRY_GUIDANCE = "retry_guidance"

        # 审查规则配置
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

    # --- 内部常量 ---
    POLITICAL = "political"
    PORNOGRAPHIC = "pornographic"
    VERBAL_ABUSE = "verbal_abuse"
    CUSTOM_RULE = "custom_rule"
    FAIL = "FAIL"
    PASS = "PASS"
    REASON_SUFFIX = "_reason"

    # --- 基础属性 ---
    raw_config: AstrBotConfig
    whitelisted_user_ids: set[str] = field(default_factory=set)
    reviewer_is_multimodal: bool = False  # [新增]

    def __post_init__(self):
        """初始化后自动加载所有配置"""
        self._load_core_settings()
        self._load_retry_settings()
        self._load_fast_review_settings()
        self._load_deep_review_settings()
        self._load_user_messages_and_prompts()
        self._validate_plugin_state()

    def _load_core_settings(self):
        """加载核心设置"""
        self.enabled = self.raw_config.get(self.CfgKeys.ENABLED, False)
        self.log_to_console = self.raw_config.get(
            self.CfgKeys.ENABLE_CONSOLE_LOGGING, True
        )
        self.reviewer_provider_id = self.raw_config.get(
            self.CfgKeys.REVIEWER_PROVIDER_ID, ""
        )
        self.reviewer_is_multimodal = self.raw_config.get(
            self.CfgKeys.REVIEWER_IS_MULTIMODAL, False
        )
        self.deep_review_context_depth = self.raw_config.get(
            self.CfgKeys.DEEP_REVIEW_CONTEXT_DEPTH, 2
        )

        # 解析白名单用户ID
        whitelisted_ids_list = self.raw_config.get(
            self.CfgKeys.WHITELISTED_USER_IDS, []
        )
        self.whitelisted_user_ids = {
            str(uid).strip() for uid in whitelisted_ids_list if str(uid).strip()
        }
        if self.whitelisted_user_ids and self.log_to_console:
            logger.info(
                f"[ReviewerPlugin] 已加载 {len(self.whitelisted_user_ids)} 个白名单用户。"
            )

    def _load_retry_settings(self):
        """加载重试、备用模型及错误处理配置"""
        retry_settings = self.raw_config.get(self.CfgKeys.RETRY_SETTINGS, {})
        self.max_attempts = retry_settings.get(self.CfgKeys.MAX_RETRIES, 3) + 1
        self.send_failure_message = retry_settings.get(
            self.CfgKeys.SEND_FAILURE_MESSAGE, True
        )
        self.send_error_message = retry_settings.get(
            self.CfgKeys.SEND_ERROR_MESSAGE, True
        )
        self.strict_mode = retry_settings.get(self.CfgKeys.STRICT_REVIEW_MODE, True)

        # 备用模型配置
        self.fallback_provider_id = retry_settings.get(
            self.CfgKeys.FALLBACK_PROVIDER_ID, ""
        )
        self.switch_provider_after_retries = retry_settings.get(
            self.CfgKeys.SWITCH_PROVIDER_AFTER, 0
        )
        self.fallback_provider_is_multimodal = retry_settings.get(
            self.CfgKeys.FALLBACK_IS_MULTIMODAL, False
        )

    def _load_fast_review_settings(self):
        """加载基于正则的快速审查配置"""
        fast_review_config = self.raw_config.get(self.CfgKeys.FAST_REVIEW_SETTINGS, {})
        self.fast_review_enabled = fast_review_config.get(
            self.CfgKeys.ENABLE_FAST_REVIEW, False
        )
        self.whitelist_rules = self._parse_key_value_config(
            fast_review_config.get(self.CfgKeys.WHITELIST_REGEX, ""),
            fast_review_config.get(self.CfgKeys.WHITELIST_PROMPTS, ""),
        )
        self.blacklist_rules = self._parse_key_value_config(
            fast_review_config.get(self.CfgKeys.BLACKLIST_REGEX, ""),
            fast_review_config.get(self.CfgKeys.BLACKLIST_PROMPTS, ""),
        )
        if self.fast_review_enabled and self.log_to_console:
            logger.info(
                f"[ReviewerPlugin] 快速审查已启用 (白名单规则: {len(self.whitelist_rules)}, 黑名单规则: {len(self.blacklist_rules)})。"
            )

    def _load_deep_review_settings(self):
        """加载基于LLM的深度审查配置"""
        if not self.reviewer_provider_id:
            logger.info("[ReviewerPlugin] 未配置审查模型ID，深度审查功能将不启用。")
            self.final_reviewer_system_prompt = ""
            self.review_configs = {}
            return

        logger.info(
            f"[ReviewerPlugin] 深度审查功能已配置 (Model: {self.reviewer_provider_id}, Multimodal: {self.reviewer_is_multimodal})。"
        )
        self._build_reviewer_system_prompt()

    def _load_user_messages_and_prompts(self):
        """加载提示语模板"""
        messages_config = self.raw_config.get(self.CfgKeys.USER_MESSAGES, {})
        self.final_failure_message = messages_config.get(
            self.CfgKeys.FINAL_FAILURE_MESSAGE, "抱歉，无法生成合规内容。"
        )
        self.internal_error_message = messages_config.get(
            self.CfgKeys.INTERNAL_ERROR_MESSAGE, "服务暂时出现问题。"
        )
        self.retry_system_prompt = self.raw_config.get(
            self.CfgKeys.RETRY_SYSTEM_PROMPT, ""
        )
        self.retry_instruction_prompt = self.raw_config.get(
            self.CfgKeys.RETRY_INSTRUCTION_PROMPT, ""
        )
        self.retry_guidance = self.raw_config.get(self.CfgKeys.RETRY_GUIDANCE, {})

    def _build_reviewer_system_prompt(self):
        """根据启用的规则动态构建审查员的System Prompt"""
        review_levels_config = self.raw_config.get(self.CfgKeys.REVIEW_LEVELS, {})
        self.review_configs = {
            self.POLITICAL: {
                "enabled": review_levels_config.get(
                    self.CfgKeys.ENABLE_POLITICAL, True
                ),
                "name": "政治敏感",
            },
            self.PORNOGRAPHIC: {
                "enabled": review_levels_config.get(
                    self.CfgKeys.ENABLE_PORNOGRAPHIC, True
                ),
                "name": "色情低俗",
            },
            self.VERBAL_ABUSE: {
                "enabled": review_levels_config.get(
                    self.CfgKeys.ENABLE_VERBAL_ABUSE, True
                ),
                "name": "言语辱骂",
            },
            self.CUSTOM_RULE: {
                "enabled": review_levels_config.get(
                    self.CfgKeys.ENABLE_CUSTOM_RULE, False
                ),
                "name": "自定义规则",
            },
        }
        base_prompt = self.raw_config.get(self.CfgKeys.REVIEWER_SYSTEM_PROMPT, "")
        replacements = {
            "{political_rules_block}": review_levels_config.get(
                self.CfgKeys.POLITICAL_RULE_PROMPT, ""
            )
            if self.review_configs[self.POLITICAL]["enabled"]
            else "",
            "{pornographic_rules_block}": review_levels_config.get(
                self.CfgKeys.PORNOGRAPHIC_RULE_PROMPT, ""
            )
            if self.review_configs[self.PORNOGRAPHIC]["enabled"]
            else "",
            "{verbal_abuse_rules_block}": review_levels_config.get(
                self.CfgKeys.VERBAL_ABUSE_RULE_PROMPT, ""
            )
            if self.review_configs[self.VERBAL_ABUSE]["enabled"]
            else "",
            "{custom_rules_block}": review_levels_config.get(
                self.CfgKeys.CUSTOM_RULE_PROMPT, ""
            )
            if self.review_configs[self.CUSTOM_RULE]["enabled"]
            else "",
        }
        final_prompt = base_prompt
        for placeholder, value in replacements.items():
            final_prompt = final_prompt.replace(placeholder, value)
        self.final_reviewer_system_prompt = final_prompt.strip()

    def _validate_plugin_state(self):
        if (
            self.enabled
            and not self.fast_review_enabled
            and not self.reviewer_provider_id
        ):
            logger.warning(
                "[ReviewerPlugin] 插件已开启，但所有审查模块均未配置，插件将处于空闲状态。"
            )

    def _parse_key_value_config(self, regex_str: str, prompt_str: str) -> "OrderedDict":
        rules = OrderedDict()
        line_pattern = re.compile(r"^\s*([^:]+?)\s*:\s*(.*)$")
        regex_map, prompt_map = {}, {}

        for line in regex_str.strip().split("\n"):
            match = line_pattern.match(line)
            if match:
                regex_map[match.group(1).strip()] = match.group(2).strip()

        for line in prompt_str.strip().split("\n"):
            match = line_pattern.match(line)
            if match:
                prompt_map[match.group(1).strip()] = match.group(2).strip()

        for key, regex_pattern in regex_map.items():
            if key in prompt_map:
                try:
                    rules[key] = {
                        "regex": re.compile(regex_pattern),
                        "prompt": prompt_map[key],
                    }
                except re.error as e:
                    logger.error(f"[ReviewerPlugin] 规则 '{key}' 正则编译失败: {e}")
        return rules


# 注册插件，更新版本号
@register("astrbot_plugin_hapemxg_inspect", "hapemxg", "hapemxg的审查插件", "1.2.1")
class ReviewerPlugin(Star):
    """
    ReviewerPlugin - LLM内容合规性审查插件
    提供基于正则的快速审查和基于LLM的深度审查，支持自动重试、备用模型切换及多模态内容处理。
    """

    PLUGIN_ID = "ReviewerPlugin"
    NON_TEXT_MESSAGE_PLACEHOLDER = "[用户发送了非文本消息]"

    def __init__(self, context: Context, config: AstrBotConfig):
        """初始化插件"""
        super().__init__(context)
        self.plugin_config = ReviewerConfig(raw_config=config)

        # --- 懒加载占位符 ---
        self._reviewer_provider = None
        self._fallback_provider = None
        self._reviewer_provider_checked = False  # 标记是否已尝试加载
        self._fallback_provider_checked = False
        self.provider_init_lock = asyncio.Lock()

        if self.plugin_config.log_to_console:
            logger.info("[ReviewerPlugin] 插件初始化完成 (Provider 采用懒加载模式)。")

    # --- 懒加载获取方法 (支持重试) ---
    async def _get_reviewer_provider(self):
        """获取深度审查用的模型实例 (带锁的懒加载，支持重试)"""
        if not self.plugin_config.reviewer_provider_id:
            return None
        # 已加载成功，直接返回
        if self._reviewer_provider is not None:
            return self._reviewer_provider
        # 未加载或之前加载失败，尝试加载
        async with self.provider_init_lock:
            if self._reviewer_provider is None:
                provider = self.context.get_provider_by_id(
                    self.plugin_config.reviewer_provider_id
                )
                if provider:
                    self._reviewer_provider = provider
                    if self.plugin_config.log_to_console:
                        logger.info(
                            f"[ReviewerPlugin] 成功加载审查模型: {self.plugin_config.reviewer_provider_id}"
                        )
                else:
                    # 不缓存失败状态，下次调用会重试
                    logger.warning(
                        f"[ReviewerPlugin] 无法获取审查模型: {self.plugin_config.reviewer_provider_id}，请检查配置。"
                    )
        return self._reviewer_provider

    async def _get_fallback_provider(self):
        """获取备用模型实例 (带锁的懒加载，支持重试)"""
        if not self.plugin_config.fallback_provider_id:
            return None
        if self._fallback_provider is not None:
            return self._fallback_provider
        async with self.provider_init_lock:
            if self._fallback_provider is None:
                provider = self.context.get_provider_by_id(
                    self.plugin_config.fallback_provider_id
                )
                if provider:
                    self._fallback_provider = provider
                    if self.plugin_config.log_to_console:
                        logger.info(
                            f"[ReviewerPlugin] 成功加载备用模型: {self.plugin_config.fallback_provider_id}"
                        )
                else:
                    logger.warning(
                        f"[ReviewerPlugin] 无法获取备用模型: {self.plugin_config.fallback_provider_id}"
                    )
        return self._fallback_provider

    async def terminate(self):
        """插件停止时的清理工作"""
        logger.info("[ReviewerPlugin] 插件已停止。")

    def _perform_fast_review(self, text: str) -> tuple[bool, str]:
        """执行正则快速审查"""
        # 1. 检查黑名单
        for rule_name, rule_data in self.plugin_config.blacklist_rules.items():
            match = rule_data["regex"].search(text)
            if match:
                trigger_word = match.group(0)
                if self.plugin_config.log_to_console:
                    logger.warning(
                        f"[ReviewerPlugin] 命中黑名单: '{rule_name}' (触发词: {trigger_word})"
                    )
                return False, rule_data["prompt"].format(trigger_word=trigger_word)

        # 2. 检查白名单
        failed_prompts = [
            rule_data["prompt"]
            for rule_name, rule_data in self.plugin_config.whitelist_rules.items()
            if not rule_data["regex"].search(text)
        ]
        if failed_prompts:
            if self.plugin_config.log_to_console:
                logger.warning(
                    f"[ReviewerPlugin] 未满足 {len(failed_prompts)} 条白名单规则。"
                )
            return False, "\n".join(failed_prompts)

        return True, ""

    def _parse_review_result(self, text: str) -> dict:
        """
        解析审查模型的返回结果。
        策略：优先尝试标准JSON解析，若失败则使用增强的正则提取（Regex Fallback）。
        """
        try:
            # 尝试清理Markdown标记并解析JSON
            cleaned_text = re.sub(r"```json\s*|\s*```", "", text).strip()
            data = json.loads(cleaned_text)
            result = {
                key: value.get("result", self.plugin_config.PASS).upper()
                for key, value in data.items()
                if isinstance(value, dict)
            }
            result.update(
                {
                    f"{key}{self.plugin_config.REASON_SUFFIX}": value.get("reason", "")
                    for key, value in data.items()
                    if isinstance(value, dict)
                }
            )
            return result
        except (json.JSONDecodeError, TypeError):
            # JSON解析失败，启用正则回退机制
            if self.plugin_config.log_to_console:
                logger.warning("[ReviewerPlugin] JSON解析失败，启用正则提取模式。")
            result = {}
            if not self.plugin_config.reviewer_provider_id:
                return result

            # 确定启用的审查类别及其在文本中的位置
            enabled_categories = []
            for key, cfg in self.plugin_config.review_configs.items():
                if cfg["enabled"]:
                    match = re.search(re.escape(cfg["name"]), text)
                    if match:
                        enabled_categories.append((key, cfg["name"], match.start()))

            enabled_categories.sort(key=lambda x: x[2])

            # 逐块解析
            for i, (key, name, start_index) in enumerate(enabled_categories):
                end_index = (
                    enabled_categories[i + 1][2]
                    if i + 1 < len(enabled_categories)
                    else len(text)
                )
                block = text[start_index:end_index]

                pass_fail_pattern = re.search(
                    "[:：]\\s*<?(PASS|FAIL)>?", block, re.IGNORECASE
                )
                if pass_fail_pattern:
                    result[key] = pass_fail_pattern.group(1).strip().upper()

                reason_pattern = re.search("理由[:：](.*)", block, re.DOTALL)
                if reason_pattern:
                    result[f"{key}{self.plugin_config.REASON_SUFFIX}"] = (
                        reason_pattern.group(1).strip()
                    )

            return result

    async def _perform_single_review_cycle(
        self,
        text_to_review: str,
        user_prompt: str,
        contexts: list[dict],
        user_session_key: str,
        image_urls: list[str] | None = None,
    ) -> ReviewCycleResult:
        """执行单次完整的审查流程 (快速审查 + 深度审查)"""
        # 1. 快速审查
        if self.plugin_config.fast_review_enabled:
            passed, reason = self._perform_fast_review(text_to_review)
            if not passed:
                return ReviewCycleResult(
                    is_passed=False,
                    review_comment=reason,
                    failed_categories=["快速审查"],
                )

        if not self.plugin_config.reviewer_provider_id:
            return ReviewCycleResult(is_passed=True, review_comment="")

        # 2. 获取审查模型 (懒加载)
        reviewer_provider = await self._get_reviewer_provider()
        if not reviewer_provider:
            logger.warning(
                f"[ReviewerPlugin] 审查模型 '{self.plugin_config.reviewer_provider_id}' 不可用，跳过深度审查。"
            )
            return ReviewCycleResult(
                is_passed=not self.plugin_config.strict_mode,
                review_comment="审查服务不可用",
                failed_categories=["Internal Error"],
                is_critical_error=self.plugin_config.strict_mode,
            )

        try:
            # 3. 从框架contexts构造审查上下文 (根据配置截取最近N轮)
            history_str = ""
            max_rounds = self.plugin_config.deep_review_context_depth
            if contexts and max_rounds > 0:
                # 将contexts转换为轮次列表 (每轮包含user+assistant)
                rounds = []
                current_round = {}
                for ctx in contexts:
                    role = ctx.get("role", "")
                    content = ctx.get("content", "")
                    # 处理多模态content (list格式)
                    if isinstance(content, list):
                        text_parts = [
                            item.get("text", "")
                            for item in content
                            if item.get("type") == "text"
                        ]
                        content = " ".join(text_parts) if text_parts else "[多模态内容]"
                    if role == "user":
                        current_round["user"] = content
                    elif role == "assistant":
                        current_round["assistant"] = content
                        if current_round.get("user"):
                            rounds.append(current_round)
                        current_round = {}

                # 只取最近的N轮
                recent_rounds = (
                    rounds[-max_rounds:] if len(rounds) > max_rounds else rounds
                )
                for i, r in enumerate(recent_rounds, 1):
                    history_str += f"历史轮次 {i} - 用户: {r.get('user', '')}\n"
                    history_str += f"历史轮次 {i} - 模型: {r.get('assistant', '')}\n\n"

            if not history_str:
                history_str = "(无历史对话)\n\n"

            # 处理审查模型的多模态支持
            final_image_urls = None
            image_hint_prompt = ""

            if image_urls:
                if self.plugin_config.reviewer_is_multimodal:
                    final_image_urls = image_urls
                else:
                    final_image_urls = None
                    image_hint_prompt = f"\n[系统提示：用户在本次对话中发送了 {len(image_urls)} 张图片，但由于技术限制你无法查看。请仅基于文本内容进行审查。]"

            review_prompt = f"--- 对话历史参考 ---\n{history_str}--- 当前待审查的对话 ---\n用户的最新提问: {user_prompt}{image_hint_prompt}\n待审查的回复: {text_to_review}"

            if self.plugin_config.log_to_console:
                log_msg = (
                    f"\n{'=' * 20} [Reviewer Request (审查请求)] {'=' * 20}\n"
                    f"▶ System Prompt (审查员设定):\n{self.plugin_config.final_reviewer_system_prompt}\n\n"
                    f"▶ User Prompt (待审内容):\n{review_prompt}\n"
                )
                if final_image_urls:
                    log_msg += f"\n▶ Image URLs:\n{final_image_urls}\n"
                log_msg += f"{'=' * 65}"
                logger.info(log_msg)

            # 4. 调用审查模型
            review_resp = await reviewer_provider.text_chat(
                prompt=review_prompt,
                session_id=f"reviewer_{user_session_key}",
                system_prompt=self.plugin_config.final_reviewer_system_prompt,
                image_urls=final_image_urls,
            )
            review_text = review_resp.completion_text.strip()

            if self.plugin_config.log_to_console:
                logger.info(
                    f"\n{'=' * 20} [Reviewer Response (审查结果)] {'=' * 20}\n"
                    f"{review_text}\n"
                    f"{'=' * 66}"
                )

            # 5. 解析结果
            parsed = self._parse_review_result(review_text)
            if not parsed:
                is_passed = not self.plugin_config.strict_mode
                return ReviewCycleResult(
                    is_passed=is_passed,
                    review_comment="审查结果格式错误",
                    failed_categories=["Format Error"],
                    is_critical_error=not is_passed,
                )

            failed_categories = [
                key
                for key, cfg in self.plugin_config.review_configs.items()
                if cfg["enabled"] and parsed.get(key) == self.plugin_config.FAIL
            ]
            if failed_categories:
                guidance = "\n".join(
                    filter(
                        None,
                        [
                            self.plugin_config.retry_guidance.get(k, "").format(
                                reason=parsed.get(
                                    f"{k}{self.plugin_config.REASON_SUFFIX}", "无理由"
                                )
                            )
                            for k in failed_categories
                        ],
                    )
                )
                failed_names = [
                    self.plugin_config.review_configs[key]["name"]
                    for key in failed_categories
                ]
                return ReviewCycleResult(
                    is_passed=False,
                    review_comment=guidance,
                    failed_categories=failed_names,
                )

        except Exception as e:
            logger.error(f"[ReviewerPlugin] 深度审查发生异常: {e}", exc_info=True)
            is_passed = not self.plugin_config.strict_mode
            return ReviewCycleResult(
                is_passed=is_passed,
                review_comment="Review Internal Error",
                failed_categories=["Internal Error"],
                is_critical_error=not is_passed,
            )

        return ReviewCycleResult(is_passed=True, review_comment="")

    async def _regenerate_response(
        self,
        provider_to_use,
        original_request: ProviderRequest,
        session_id: str,
        failed_response: str,
        review_comment: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
        use_multimodal: bool = True,
    ) -> str:
        """
        修正回复逻辑 (Reflexion模式)。
        将"错误回复"、"批评意见"和"原始问题"打包发回给LLM进行重写。
        """
        # 多模态兼容处理
        final_image_urls_for_llm = image_urls if use_multimodal else None
        multimodal_context_hint = ""
        if image_urls and not use_multimodal:
            multimodal_context_hint = f"\n\n[System Note: User sent {len(image_urls)} images, but you cannot see them now. Reply based on text only.]"

        # 智能构建包含上下文的 User Prompt
        original_full_prompt = original_request.prompt
        clean_user_prompt = user_prompt.strip()

        if clean_user_prompt == self.NON_TEXT_MESSAGE_PLACEHOLDER:
            user_prompt_with_context = original_full_prompt
        else:
            prefix_to_preserve = ""
            last_occurrence_index = original_full_prompt.rfind(clean_user_prompt)
            if (
                last_occurrence_index != -1
                and original_full_prompt[last_occurrence_index:].strip()
                == clean_user_prompt
            ):
                prefix_to_preserve = original_full_prompt[:last_occurrence_index]
            elif original_full_prompt.strip().endswith(clean_user_prompt):
                prefix_to_preserve = original_full_prompt.strip()[
                    : -len(clean_user_prompt)
                ]
            user_prompt_with_context = prefix_to_preserve + clean_user_prompt

        # 组装重试指令
        retry_instruction_template = self.plugin_config.retry_instruction_prompt.format(
            review_comment=review_comment,
            failed_response=failed_response,
            user_prompt=user_prompt_with_context,
        )
        final_prompt_for_llm = retry_instruction_template + multimodal_context_hint

        retry_contexts = original_request.contexts.copy()
        retry_contexts.append({"role": "assistant", "content": failed_response})
        final_retry_system_prompt = (
            self.plugin_config.retry_system_prompt or original_request.system_prompt
        )

        # 调用模型生成修正后的回复
        retry_response = await provider_to_use.text_chat(
            prompt=final_prompt_for_llm,
            session_id=session_id,
            contexts=retry_contexts,
            system_prompt=final_retry_system_prompt,
            image_urls=final_image_urls_for_llm,
        )
        return retry_response.completion_text.strip()

    # --- AstrBot 事件钩子 ---

    @event_filter.on_llm_request(priority=100)
    async def store_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """钩子：在LLM请求发出前截获并存储上下文。"""
        if self.plugin_config.enabled:
            setattr(event, "_original_llm_request_for_reviewer", req)
            setattr(event, "_image_urls_for_reviewer", req.image_urls or [])
            user_prompt_to_store = (
                getattr(event, "_translated_option_text", None)
                or (event.message_str or "").strip()
            )
            if not user_prompt_to_store and event.message_obj.message:
                user_prompt_to_store = self.NON_TEXT_MESSAGE_PLACEHOLDER
            setattr(event, "_user_prompt_for_reviewer", user_prompt_to_store)

    @event_filter.on_llm_response(priority=10)
    async def review_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """钩子：在LLM响应后、发送给用户前进行拦截审查。"""
        if not self.plugin_config.enabled:
            return
        original_request = getattr(event, "_original_llm_request_for_reviewer", None)
        if not original_request:
            return

        sender_id = str(event.get_sender_id())
        if sender_id in self.plugin_config.whitelisted_user_ids:
            if self.plugin_config.log_to_console:
                logger.info(f"[ReviewerPlugin] 用户 {sender_id} 在白名单中，跳过审查。")
            return

        user_prompt = getattr(
            event, "_user_prompt_for_reviewer", original_request.prompt
        )
        image_urls = getattr(event, "_image_urls_for_reviewer", [])
        session_id = event.unified_msg_origin
        user_session_key = f"{session_id}_{event.get_sender_id()}"
        current_response_text = resp.completion_text
        max_attempts = self.plugin_config.max_attempts

        # 获取当前主模型 (用于重试生成)
        try:
            main_provider = self.context.get_using_provider()
        except ValueError as e:
            logger.warning(f"[ReviewerPlugin] 无法获取主模型，跳过审查: {e}")
            return

        for attempt in range(1, max_attempts + 1):
            if self.plugin_config.log_to_console:
                logger.info(
                    f"[ReviewerPlugin] (Session: {user_session_key}) 执行第 {attempt}/{max_attempts} 轮审查..."
                )

            review_result = await self._perform_single_review_cycle(
                current_response_text,
                user_prompt,
                original_request.contexts,
                user_session_key,
                image_urls=image_urls,
            )

            if review_result.is_critical_error:
                if self.plugin_config.send_error_message:
                    await event.send(
                        event.plain_result(self.plugin_config.internal_error_message)
                    )
                event.stop_event()
                return

            if review_result.is_passed:
                if self.plugin_config.log_to_console:
                    logger.info(f"[ReviewerPlugin] 审查通过 (轮次: {attempt})。")
                resp.completion_text = current_response_text
                return

            if self.plugin_config.log_to_console:
                logger.warning(
                    f"[ReviewerPlugin] 审查驳回: {', '.join(review_result.failed_categories)}。"
                )

            if attempt >= max_attempts:
                break

            try:
                provider_for_this_attempt = main_provider
                use_multimodal_for_this_attempt = True

                if (
                    self.plugin_config.switch_provider_after_retries > 0
                    and attempt >= self.plugin_config.switch_provider_after_retries
                ):
                    # 【关键改动】：懒加载备用模型
                    fallback_provider = await self._get_fallback_provider()
                    if fallback_provider:
                        provider_for_this_attempt = fallback_provider
                        use_multimodal_for_this_attempt = (
                            self.plugin_config.fallback_provider_is_multimodal
                        )
                        if self.plugin_config.log_to_console:
                            logger.info(
                                f"[ReviewerPlugin] 切换至备用模型 '{self.plugin_config.fallback_provider_id}' 进行重试。"
                            )

                if not provider_for_this_attempt:
                    raise RuntimeError("No available LLM provider for regeneration.")

                current_response_text = await self._regenerate_response(
                    provider_for_this_attempt,
                    original_request,
                    session_id,
                    current_response_text,
                    review_result.review_comment,
                    user_prompt,
                    image_urls=image_urls,
                    use_multimodal=use_multimodal_for_this_attempt,
                )
            except Exception as e:
                logger.error(f"[ReviewerPlugin] 重试生成失败: {e}", exc_info=True)
                if self.plugin_config.send_error_message:
                    await event.send(
                        event.plain_result(self.plugin_config.internal_error_message)
                    )
                event.stop_event()
                return

        logger.error(f"[ReviewerPlugin] 达到最大重试次数 ({max_attempts})，拦截消息。")
        if self.plugin_config.send_failure_message:
            await event.send(
                event.plain_result(self.plugin_config.final_failure_message)
            )
        event.stop_event()
