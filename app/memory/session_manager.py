import json
from typing import Dict, List

import redis

from app.core.config import settings


class SessionManager:
    """
    基于 Redis 的多轮对话记忆管理器
    策略：滑动窗口（最近 N 轮）+ 超出时生成摘要
    """

    def __init__(self):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        self.window_size = settings.memory_window_size
        self.ttl = settings.session_ttl

    def _key(self, session_id: str) -> str:
        return f"zhiyuan:session:{session_id}"

    def add_message(self, session_id: str, role: str, content: str):
        """添加一条消息到会话历史"""
        key = self._key(session_id)
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        max_messages = self.window_size * 2
        if len(history) > max_messages:
            history = history[-max_messages:]
        self.redis.setex(key, self.ttl, json.dumps(history, ensure_ascii=False))

    def get_history(self, session_id: str) -> List[Dict]:
        """获取会话历史"""
        key = self._key(session_id)
        data = self.redis.get(key)
        return json.loads(data) if data else []

    def format_history_for_prompt(self, session_id: str) -> str:
        """将历史格式化为 Prompt ��使用的文本"""
        history = self.get_history(session_id)
        lines = []
        for msg in history:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    def clear(self, session_id: str):
        """清除会话"""
        self.redis.delete(self._key(session_id))