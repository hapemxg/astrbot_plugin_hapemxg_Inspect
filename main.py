import json
import re
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

@register("ReviewerPlugin", "YourName", "LLM回复分级审查插件", "1.0.0") 
class ReviewerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        self.enabled = self.config.get("enabled", False)
        self.reviewer_provider_id = self.config.get("reviewer_provider_id", "")
        
        retry_settings = self.config.get("retry_settings", {})
        self.max_attempts = retry_settings.get("max_retries", 3) + 1
        # 新增：加载重试专用的System Prompt配置
        self.retry_system_prompt = retry_settings.get("retry_system_prompt", "")

        review_levels_config = self.config.get("review_levels", {})
        self.enable_political = review_levels_config.get("enable_political", True)
        self.enable_pornographic = review_levels_config.get("enable_pornographic", True)
        self.enable_verbal_abuse = review_levels_config.get("enable_verbal_abuse", True)
        self.enable_custom_rule = review_levels_config.get("enable_custom_rule", False)
        custom_rule_prompt = review_levels_config.get("custom_rule_prompt", "无")

        base_system_prompt = self.config.get("reviewer_system_prompt", "")
        self.final_reviewer_system_prompt = base_system_prompt.replace("{custom_rule_definition}", custom_rule_prompt if self.enable_custom_rule else "无自定义规则，此项默认通过。")
        
        self.retry_instruction_prompt = self.config.get("retry_instruction_prompt", "")

    def _parse_review_result(self, text: str) -> dict:
        result = {}
        patterns = {
            "political": r"政治敏感[:：]\s*<?(PASS|FAIL)>?", "pornographic": r"色情低俗[:：]\s*<?(PASS|FAIL)>?",
            "verbal_abuse": r"言语辱骂[:：]\s*<?(PASS|FAIL)>?", "custom_rule": r"自定义规则[:：]\s*<?(PASS|FAIL)>?",
            "reason": r"理由[:：]([\s\S]+)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match: result[key] = match.group(1).strip()
            else: result[key] = None
        if result.get("reason"): result["reason"] = result["reason"].strip()
        return result

    @filter.on_llm_request(priority=100)
    async def store_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if self.enabled:
            setattr(event, '_original_llm_request_for_reviewer', req)

    @filter.on_llm_response(priority=10)
    async def review_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        if not self.enabled or not self.reviewer_provider_id:
            return

        original_request = getattr(event, '_original_llm_request_for_reviewer', None)
        if not original_request:
            return

        reviewer_provider = self.context.get_provider_by_id(self.reviewer_provider_id)
        if not reviewer_provider:
            logger.warning(f"[ReviewerPlugin] 未找到审查模型，跳过审查。")
            return

        attempt = 0
        current_response_text = resp.completion_text

        while attempt < self.max_attempts:
            attempt += 1
            logger.info(f"[ReviewerPlugin] 开始第 {attempt}/{self.max_attempts} 次审查...")
            logger.info(f"[ReviewerPlugin] 待审查内容:\n---\n{current_response_text}\n---")

            history_for_review = [f"{msg.get('role', 'unknown')}: {msg.get('content')}" for msg in original_request.contexts if msg and msg.get('content')]
            history_str = "\n".join(history_for_review)
            review_prompt = f"--- 对话历史 ---\n{history_str}\n\n--- 用户的最新提问 ---\n{original_request.prompt}\n\n--- 待审查的回复 ---\n{current_response_text}"
            
            try:
                review_result_resp = await reviewer_provider.text_chat(prompt=review_prompt, session_id=f"reviewer_{event.unified_msg_origin}", system_prompt=self.final_reviewer_system_prompt)
                review_result_text = review_result_resp.completion_text.strip()
                logger.info(f"[ReviewerPlugin] 第 {attempt} 次审查模型判断:\n---\n{review_result_text}\n---")
            except Exception as e:
                logger.error(f"[ReviewerPlugin] 调用审查模型失败: {e}")
                resp.completion_text = current_response_text
                return

            parsed_result = self._parse_review_result(review_result_text)
            failed_categories = []
            if self.enable_political and parsed_result.get("political") == "FAIL": failed_categories.append("政治敏感")
            if self.enable_pornographic and parsed_result.get("pornographic") == "FAIL": failed_categories.append("色情低俗")
            if self.enable_verbal_abuse and parsed_result.get("verbal_abuse") == "FAIL": failed_categories.append("言语辱骂")
            if self.enable_custom_rule and parsed_result.get("custom_rule") == "FAIL": failed_categories.append("自定义规则")

            if not failed_categories:
                logger.info(f"[ReviewerPlugin] 第 {attempt} 次审查通过。最终回复已确定。")
                resp.completion_text = current_response_text
                return

            review_reason = parsed_result.get("reason", "无详细理由。")
            logger.warning(f"[ReviewerPlugin] 第 {attempt} 次审查驳回。违反规则: {', '.join(failed_categories)}。理由: {review_reason}")

            if attempt >= self.max_attempts:
                logger.error(f"[ReviewerPlugin] 已达到最大尝试次数 ({self.max_attempts})，仍未生成合规内容。终止事件传播。")
                resp.completion_text = "抱歉，我多次尝试后仍无法生成完全合规的回复，为安全起见，本次回答已取消。"
                event.stop_event()
                return

            try:
                main_provider = self.context.get_using_provider()
                retry_instruction_block = self.retry_instruction_prompt.format(fail_categories=', '.join(failed_categories), review_comment=review_reason, failed_response=current_response_text)
                new_user_prompt = f"{retry_instruction_block}\n\n{original_request.prompt}"

                # --- [核心修改点：应用可配置的System Prompt] ---
                # 检查用户是否配置了专用的重试System Prompt
                if self.retry_system_prompt:
                    final_retry_system_prompt = self.retry_system_prompt
                    logger.debug("[ReviewerPlugin] 使用了配置中指定的专用重试System Prompt。")
                else:
                    # 如果未配置，则使用当前对话的原始System Prompt
                    final_retry_system_prompt = original_request.system_prompt
                    logger.debug("[ReviewerPlugin] 使用当前对话的原始System Prompt进行重试。")
                # --- [修改结束] ---
                
                logger.info(f"[ReviewerPlugin] 准备第 {attempt + 1} 次生成...")
                retry_response = await main_provider.text_chat(
                    prompt=new_user_prompt, 
                    session_id=event.unified_msg_origin,
                    contexts=original_request.contexts,
                    system_prompt=final_retry_system_prompt # 使用我们最终决定的System Prompt
                )
                current_response_text = retry_response.completion_text.strip()

            except Exception as e:
                logger.error(f"[ReviewerPlugin] 重新生成回复时发生错误: {e}")
                resp.completion_text = "抱歉，我在尝试修正我的回复时遇到了一个内部错误。"
                return