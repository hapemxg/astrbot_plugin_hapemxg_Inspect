# main.py

# 导入所需的标准库和框架组件
import json
import re
from collections import deque, defaultdict, OrderedDict

# 导入 AstrBot 框架的核心组件
from astrbot.api.event import filter as event_filter, AstrMessageEvent
import astrbot.api.message_components as Comp
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

@register("ReviewerPlugin", "hapemxg", "hapemxg的审查插件", "1.0.0")
class ReviewerPlugin(Star):
    """
    这是hapemxg的LLM回复审查插件。

    核心功能:
    1.  **多层审查机制**: 在LLM生成回复后、发送给用户前，执行快速（基于正则）和深度（基于模型）两层审查。
    2.  **闭环重试与修正**: 当审查失败时，能自动生成指导意见，引导主模型重新生成回复，形成“审查-反馈-修正”的闭环。
    3.  **上下文感知**: 在进行深度审查时，会参考最近的对话历史，以做出更精准的判断。
    4.  **高度可配置**: 从审查开关到重试逻辑，再到每个审查维度的具体标准和提示语，几乎所有行为都可通过配置文件进行调整。
    5.  **健壮的错误处理**: 对模型调用失败、返回格式错误等异常情况进行了妥善处理，确保插件的稳定运行。
    """

    # --- 使用常量管理JSON和字典中的键名，以提高代码的可维护性和减少硬编码错误 ---
    POLITICAL = "political"
    PORNOGRAPHIC = "pornographic"
    VERBAL_ABUSE = "verbal_abuse"
    CUSTOM_RULE = "custom_rule"
    FAIL = "FAIL"
    PASS = "PASS"
    REASON_SUFFIX = "_reason"

    def __init__(self, context: Context, config: AstrBotConfig):
        """
        插件初始化方法。
        在AstrBot加载插件时被调用，负责加载配置、初始化内部状态和检查依赖。
        """
        super().__init__(context)
        self.config = config
        
        # --- 基础配置加载 ---
        self.enabled = self.config.get("enabled", False)
        self.log_to_console = self.config.get("enable_console_logging", True)
        self.reviewer_provider_id = self.config.get("reviewer_provider_id", "")
        self.deep_review_context_depth = self.config.get("deep_review_context_depth", 2)

        # --- 重试逻辑配置 ---
        retry_settings = self.config.get("retry_settings", {})
        self.max_attempts = retry_settings.get("max_retries", 3) + 1  # +1 是因为包含首次尝试
        self.send_failure_message = retry_settings.get("send_failure_message_on_max_retries", True)
        self.strict_mode = retry_settings.get("strict_review_mode", True)

        # --- 快速审查模块配置 ---
        fast_review_config = self.config.get("fast_review_settings", {})
        self.fast_review_enabled = fast_review_config.get("enable_fast_review", False)
        # 解析文本配置，编译正则表达式，为快速审查做准备
        self.whitelist_rules = self._parse_key_value_config(fast_review_config.get("whitelist_regex", ""), fast_review_config.get("whitelist_prompts", ""))
        self.blacklist_rules = self._parse_key_value_config(fast_review_config.get("blacklist_regex", ""), fast_review_config.get("blacklist_prompts", ""))
        if self.fast_review_enabled and self.log_to_console:
            logger.info(f"[ReviewerPlugin] 【快速审查】模块已启用。加载了 {len(self.whitelist_rules)} 条白名单规则和 {len(self.blacklist_rules)} 条黑名单规则。")

        # --- 深度审查模块配置与校验 ---
        self.deep_review_enabled = True
        if not self.reviewer_provider_id:
            self.deep_review_enabled = False
            logger.info("[ReviewerPlugin] 未配置 'reviewer_provider_id'，【深度审查】模块将不会运行。")
        else:
            # 校验配置的审查模型ID是否存在且有效
            reviewer_provider = self.context.get_provider_by_id(self.reviewer_provider_id)
            if not reviewer_provider:
                self.deep_review_enabled = False
                logger.error(f"[ReviewerPlugin] 配置的审查模型ID '{self.reviewer_provider_id}' 无效或未找到，【深度审查】模块将禁用。")
            else:
                self.reviewer_provider = reviewer_provider
                logger.info(f"[ReviewerPlugin] 【深度审查】模块已启用，使用模型: '{self.reviewer_provider_id}'。")

        # --- 插件总体状态警告 ---
        if self.enabled and not self.fast_review_enabled and not self.deep_review_enabled:
            logger.warning("[ReviewerPlugin] 插件主开关已开启，但【快速审查】和【深度审查】均未有效配置，插件将不执行任何操作。")

        # --- 用户提示信息与重试指令模板加载 ---
        messages_config = self.config.get("user_facing_messages", {})
        self.final_failure_message = messages_config.get("final_failure", "抱歉，我多次尝试后仍无法生成完全合规的回复，为安全起见，本次回答已取消。")
        self.internal_error_message = messages_config.get("internal_error", "抱歉，内容审查服务暂时出现问题，本次回答已取消。")
        self.retry_system_prompt = self.config.get("retry_system_prompt", "")
        self.retry_instruction_prompt = self.config.get("retry_instruction_prompt", "")
        self.retry_guidance = self.config.get("retry_guidance", {})

        # --- 动态构建深度审查的系统提示词 ---
        if self.deep_review_enabled:
            review_levels_config = self.config.get("review_levels", {})
            self.review_configs = {
                self.POLITICAL: {"enabled": review_levels_config.get("enable_political", True), "name": "政治敏感"},
                self.PORNOGRAPHIC: {"enabled": review_levels_config.get("enable_pornographic", True), "name": "色情低俗"},
                self.VERBAL_ABUSE: {"enabled": review_levels_config.get("enable_verbal_abuse", True), "name": "言语辱骂"},
                self.CUSTOM_RULE: {"enabled": review_levels_config.get("enable_custom_rule", False), "name": "自定义规则"},
            }
            # 根据用户开启的审查项，动态地将对应的规则描述填充到主模板中
            base_system_prompt = self.config.get("reviewer_system_prompt", "")
            political_rule_prompt = review_levels_config.get("political_rule_prompt", "")
            pornographic_rule_prompt = review_levels_config.get("pornographic_rule_prompt", "")
            verbal_abuse_rule_prompt = review_levels_config.get("verbal_abuse_rule_prompt", "")
            custom_rule_prompt = review_levels_config.get("custom_rule_prompt", "")
            final_prompt = base_system_prompt.replace("{political_rules_block}", political_rule_prompt if self.review_configs[self.POLITICAL]["enabled"] else "") \
                                            .replace("{pornographic_rules_block}", pornographic_rule_prompt if self.review_configs[self.PORNOGRAPHIC]["enabled"] else "") \
                                            .replace("{verbal_abuse_rules_block}", verbal_abuse_rule_prompt if self.review_configs[self.VERBAL_ABUSE]["enabled"] else "") \
                                            .replace("{custom_rules_block}", custom_rule_prompt if self.review_configs[self.CUSTOM_RULE]["enabled"] else "")
            self.final_reviewer_system_prompt = final_prompt.strip()

        # --- 初始化对话历史记录器 ---
        # 用于存储每个用户通过审查的对话，以便在后续审查中作为上下文参考
        self.approved_history = defaultdict(lambda: deque(maxlen=self.deep_review_context_depth))

    def _parse_key_value_config(self, regex_str: str, prompt_str: str) -> 'OrderedDict':
        """
        解析配置文件中“键: 值”格式的多行文本，并编译正则表达式。
        """
        rules = OrderedDict()
        regex_map = {}
        prompt_map = {}
        
        # 定义一个正则表达式来匹配 "key: value" 格式
        # ^\s*        - 匹配行首的任意空格
        # ([^:]+?)    - 第1个捕获组 (key): 匹配一个或多个不为冒号的字符，非贪婪
        # \s*:\s*     - 匹配key和value之间的冒号，以及两边的任意空格
        # (.*)        - 第2个捕获组 (value): 匹配行内余下的所有字符
        # $           - 匹配行尾
        line_pattern = re.compile(r"^\s*([^:]+?)\s*:\s*(.*)$")

        # --- 解析正则表达式 ---
        for line in regex_str.strip().split('\n'):
            match = line_pattern.match(line)
            if match:
                # 如果匹配成功，group(1)是key，group(2)是value
                key, value = match.groups()
                if key and value: # 确保key和value都不是空字符串
                    regex_map[key] = value

        # --- 解析提示词 ---
        for line in prompt_str.strip().split('\n'):
            match = line_pattern.match(line)
            if match:
                key, value = match.groups()
                if key and value:
                    prompt_map[key] = value
        
        # --- 合并并编译 ---
        for key, regex_pattern in regex_map.items():
            if key in prompt_map:
                try:
                    rules[key] = { "regex": re.compile(regex_pattern), "prompt": prompt_map[key] }
                except re.error as e:
                    logger.error(f"[ReviewerPlugin] 快速审查规则 '{key}' 的正则表达式编译失败: {e}")
        return rules

    def _perform_fast_review(self, text: str) -> (bool, str):
        """
        执行快速审查。
        依次检查黑名单和白名单规则。

        Args:
            text (str): 待审查的LLM回复文本。

        Returns:
            tuple[bool, str]: 一个元组，第一个元素表示是否通过，第二个元素是未通过时的指导意见。
        """
        # 检查黑名单，任何一条命中则立即失败
        for rule_name, rule_data in self.blacklist_rules.items():
            match = rule_data["regex"].search(text)
            if match:
                trigger_word = match.group(0)
                final_prompt = rule_data["prompt"].format(trigger_word=trigger_word)
                if self.log_to_console:
                    logger.warning(f"[ReviewerPlugin] 快速审查命中黑名单规则: '{rule_name}' (触发词: '{trigger_word}')")
                return False, final_prompt
        
        # 检查白名单，必须全部满足才算通过
        failed_whitelist_prompts = []
        for rule_name, rule_data in self.whitelist_rules.items():
            if not rule_data["regex"].search(text):
                if self.log_to_console: 
                    logger.warning(f"[ReviewerPlugin] 快速审查未满足白名单规则: '{rule_name}'")
                failed_whitelist_prompts.append(rule_data["prompt"])
        
        if failed_whitelist_prompts:
            return False, "\n".join(failed_whitelist_prompts)
        
        if self.log_to_console:
            logger.info("[ReviewerPlugin] 快速审查通过。")
        return True, ""

    def _parse_review_result(self, text: str) -> dict:
        """
        解析审查模型返回的文本，提取审查结果。
        优先尝试解析JSON，如果失败则回退到使用正则表达式进行宽松解析。

        Args:
            text (str): 审查模型返回的原始文本。

        Returns:
            dict: 包含各审查项结果和理由的字典。
        """
        try:
            # 优先尝试解析标准的JSON格式
            cleaned_text = re.sub(r'```json\s*|\s*```', '', text).strip()
            data = json.loads(cleaned_text)
            result = {key: value.get("result", self.PASS).upper() for key, value in data.items() if isinstance(value, dict)}
            result.update({f"{key}{self.REASON_SUFFIX}": value.get("reason", "") for key, value in data.items() if isinstance(value, dict)})
            return result
        except (json.JSONDecodeError, TypeError):
            # 如果JSON解析失败，启动备用方案：正则表达式解析
            if self.log_to_console:
                logger.warning("[ReviewerPlugin] 审查模型返回的不是标准JSON，尝试使用正则解析。")
            result = {}
            if self.deep_review_enabled:
                for key in self.review_configs.keys():
                    name = self.review_configs[key]["name"]
                    # 匹配 "类别: <PASS/FAIL>"
                    pattern = re.search(f"{name}[:：]\\s*<?(PASS|FAIL)>?", text, re.IGNORECASE)
                    if pattern:
                        result[key] = pattern.group(1).strip().upper()
                    # 匹配 "类别理由: [具体理由]"
                    reason_pattern = re.search(f"{name}理由[:：]([\\s\\S]*?)(?=\\n[^\n]+[:：]|$)", text)
                    if reason_pattern:
                        result[f"{key}{self.REASON_SUFFIX}"] = reason_pattern.group(1).strip()
            return result

    @event_filter.on_llm_request(priority=100)
    async def store_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        事件钩子：在LLM请求发出前触发。
        此钩子的作用是“暂存”原始的LLM请求对象和用户输入文本，以便在后续的响应审查环节中使用。
        """
        if self.enabled:
            # 将原始请求对象附加到事件对象上，方便后续钩子访问
            setattr(event, '_original_llm_request_for_reviewer', req)
            # 使用 event.message_str 获取消息中的纯文本内容，如果为空则回退到使用请求中的prompt
            final_user_prompt = event.message_str or req.prompt
            setattr(event, '_user_prompt_for_reviewer', final_user_prompt)

    @event_filter.on_llm_response(priority=10)
    async def review_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """
        事件钩子：在收到LLM响应后、发送给用户前触发。
        这是插件的核心逻辑，负责执行完整的审查与重试流程。
        """
        # --- 前置检查：插件是否启用，以及前一个钩子是否成功暂存了数据 ---
        if not self.enabled or (not self.fast_review_enabled and not self.deep_review_enabled):
            return
        original_request = getattr(event, '_original_llm_request_for_reviewer', None)
        if not original_request:
            return
        
        # --- 初始化审查循环所需变量 ---
        user_prompt_for_review = getattr(event, '_user_prompt_for_reviewer', original_request.prompt)
        session_id = event.unified_msg_origin
        attempt = 0
        current_response_text = resp.completion_text
        user_session_key = f"{session_id}_{event.get_sender_id()}" # 唯一标识一个用户的会话

        # --- 开始审查与重试循环 ---
        while attempt < self.max_attempts:
            attempt += 1
            if self.log_to_console:
                logger.info(f"[ReviewerPlugin] (User Session: {user_session_key}) 开始第 {attempt}/{self.max_attempts} 次审查...")
                logger.info(f"[ReviewerPlugin] 待审查内容:\n---\n{current_response_text}\n---")
            
            is_passed = True
            final_review_comment = ""
            failed_categories_names = []

            # --- 阶段一：快速审查 ---
            if self.fast_review_enabled:
                passed_fast_review, reason = self._perform_fast_review(current_response_text)
                if not passed_fast_review:
                    is_passed = False
                    final_review_comment = reason
                    failed_categories_names.append("快速审查")

            # --- 阶段二：深度审查 (仅在快速审查通过后执行) ---
            if is_passed and self.deep_review_enabled:
                is_passed = False # 先假设深度审查失败，只有明确通过才置为True
                try:
                    # 准备包含对话历史的、给审查模型的完整Prompt
                    past_dialogue_str = "--- 对话历史参考 ---\n" + ("".join(f"历史轮次 {i+1} - 用户提问: {q}\n历史轮次 {i+1} - 模型回答: {a}\n\n" for i, (q, a) in enumerate(list(self.approved_history[user_session_key]))) if self.approved_history[user_session_key] else "(无历史对话)\n\n")
                    current_turn_str = f"--- 当前待审查的对话 ---\n用户的最新提问: {user_prompt_for_review}\n待审查的回复: {current_response_text}"
                    review_prompt = f"请结合以下对话历史，判断“当前待审查的对话”中的回复是否合规。\n\n{past_dialogue_str}{current_turn_str}"
                    
                    # 打印将要发送给审查模型的完整请求
                    if self.log_to_console:
                        logger.info(f"[ReviewerPlugin] 第 {attempt} 次【深度审查】发送给审查模型的请求:\n---\n{review_prompt}\n---")

                    # 调用审查模型
                    review_result_resp = await self.reviewer_provider.text_chat(prompt=review_prompt, session_id=f"reviewer_{user_session_key}", system_prompt=self.final_reviewer_system_prompt)
                    review_result_text = review_result_resp.completion_text.strip()
                    if self.log_to_console:
                        logger.info(f"[ReviewerPlugin] 第 {attempt} 次【深度审查】模型判断:\n---\n{review_result_text}\n---")
                    
                    # 解析审查结果
                    parsed_result = self._parse_review_result(review_result_text)
                    if not parsed_result:
                        # 如果解析失败，根据严格模式决定是否拦截
                        if self.strict_mode:
                            failed_categories_names.append("审查模型解析失败")
                    else:
                        # 根据解析出的失败项，从配置中组合重试指导意见
                        guidance_blocks = [self.retry_guidance.get(key, "").format(reason=parsed_result.get(f"{key}{self.REASON_SUFFIX}", "未提供具体理由")) for key, config in self.review_configs.items() if config["enabled"] and parsed_result.get(key) == self.FAIL]
                        failed_categories_names.extend([config["name"] for key, config in self.review_configs.items() if config["enabled"] and parsed_result.get(key) == self.FAIL])
                        if guidance_blocks:
                            final_review_comment = "\n".join(filter(None, guidance_blocks))
                        else:
                            # 如果没有任何失败项，则判定为通过
                            is_passed = True
                except Exception as e:
                    logger.error(f"[ReviewerPlugin] 调用审查模型时发生严重错误: {e}")
                    if self.strict_mode:
                        failed_categories_names.append("审查模型调用失败")
                        await event.send(event.plain_result(self.internal_error_message))
                        event.stop_event() # 发生严重错误，终止后续所有流程
                        return
            
            # --- 阶段三：决策与执行 ---
            if is_passed:
                # 如果所有审查都通过
                if self.log_to_console:
                    logger.info(f"[ReviewerPlugin] 第 {attempt} 次审查最终通过。")
                # 将通过的对话存入历史记录
                self.approved_history[user_session_key].append((user_prompt_for_review, current_response_text))
                # 更新原始响应对象的内容，然后正常返回，让AstrBot继续发送流程
                resp.completion_text = current_response_text
                return

            # 如果审查失败，记录日志并准备重试
            if self.log_to_console:
                logger.warning(f"[ReviewerPlugin] 第 {attempt} 次审查驳回。违反规则: {', '.join(failed_categories_names)}。")
                logger.warning(f"[ReviewerPlugin] 生成的指导意见:\n---\n{final_review_comment}\n---")
            
            # 检查是否达到最大重试次数
            if attempt >= self.max_attempts:
                logger.error(f"[ReviewerPlugin] 已达到最大尝试次数 ({self.max_attempts})。")
                if self.send_failure_message:
                    await event.send(event.plain_result(self.final_failure_message))
                event.stop_event() # 终止事件，阻止发送不合规的最终回复
                return

            # --- 阶段四：准备并执行重试 ---
            try:
                main_provider = self.context.get_using_provider()
                # 构建用于重试的、包含指导意见的新Prompt
                retry_instruction = self.retry_instruction_prompt.format(review_comment=final_review_comment, failed_response=current_response_text, user_prompt=user_prompt_for_review)
                
                # 构建重试时的上下文，将失败的回复作为一轮对话历史添加进去
                retry_contexts = original_request.contexts.copy()
                retry_contexts.append({"role": "assistant", "content": current_response_text})
                
                # 确定重试时使用的系统提示词
                final_retry_system_prompt = self.retry_system_prompt or original_request.system_prompt
                
                if self.log_to_console:
                    logger.info(f"[ReviewerPlugin] 准备第 {attempt + 1} 次生成...")
                
                # 调用主模型进行重试
                retry_response = await main_provider.text_chat(prompt=retry_instruction, session_id=session_id, contexts=retry_contexts, system_prompt=final_retry_system_prompt)
                current_response_text = retry_response.completion_text.strip()
            except Exception as e:
                # 如果在重试过程中发生错误，则终止流程
                logger.error(f"[ReviewerPlugin] 重新生成回复时发生错误: {e}")
                await event.send(event.plain_result(self.internal_error_message))
                event.stop_event()
                return