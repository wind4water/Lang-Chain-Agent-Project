"""
飞书客户端封装
基于 lark-oapi SDK，提供卡片消息发送、更新、回复等功能
参考 feishu-claude-code 的实现方式
"""
import json
import os
import tempfile
import time
from typing import Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
)

from app.integrations.feishu_config import FeishuConfig


class FeishuClient:
    """飞书客户端 - 封装 lark-oapi SDK"""

    def __init__(self, config: FeishuConfig):
        self.config = config
        self._client = lark.Client.builder() \
            .app_id(config.app_id) \
            .app_secret(config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()

    # ------------------------------------------------------------------
    # 卡片 JSON 构造
    # ------------------------------------------------------------------
    def _card_json(self, content: str, loading: bool = False) -> str:
        """构造飞书卡片消息 JSON"""
        if loading:
            display_text = content if content else "🤔 思考中..."
        else:
            display_text = content

        # 截断超长内容
        if len(display_text) > self.config.max_card_content:
            display_text = display_text[:self.config.max_card_content - 3] + "..."

        card = {
            "config": {"wide_screen_mode": True},
            "elements": [
                {
                    "tag": "markdown",
                    "content": display_text,
                }
            ],
        }
        return json.dumps(card, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 重试机制
    # ------------------------------------------------------------------
    async def _retry_with_backoff(self, coro_func, max_retries: int = 3, initial_delay: float = 0.5):
        """指数退避重试"""
        delay = initial_delay
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await coro_func()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"[feishu retry] 第 {attempt + 1} 次失败，{delay:.1f}s 后重试: {e}", flush=True)
                    import asyncio
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    print(f"[feishu retry] 已达最大重试次数 {max_retries + 1}，放弃", flush=True)

        raise last_error

    # ------------------------------------------------------------------
    # 发送消息
    # ------------------------------------------------------------------
    async def send_card_to_user(self, open_id: str, content: str = "", loading: bool = True) -> str:
        """向用户发送卡片消息，返回 message_id"""
        async def _send():
            req = (
                CreateMessageRequest.builder()
                .receive_id_type("open_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(open_id)
                    .msg_type("interactive")
                    .content(self._card_json(content, loading=loading))
                    .build()
                )
                .build()
            )
            resp = await self._client.im.v1.message.acreate(req)
            if not resp.success():
                raise RuntimeError(f"发送卡片消息失败: {resp.code} {resp.msg}")
            return resp.data.message_id

        return await self._retry_with_backoff(_send, max_retries=3)

    async def reply_card(self, message_id: str, content: str = "", loading: bool = True) -> str:
        """回复用户消息（卡片形式），返回回复消息的 message_id"""
        async def _reply():
            req = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("interactive")
                    .content(self._card_json(content, loading=loading))
                    .build()
                )
                .build()
            )
            resp = await self._client.im.v1.message.areply(req)
            if not resp.success():
                raise RuntimeError(f"回复卡片消息失败: {resp.code} {resp.msg}")
            return resp.data.message_id

        return await self._retry_with_backoff(_reply, max_retries=3)

    async def update_card(self, message_id: str, content: str):
        """用 patch 更新已发送的卡片内容（流式核心）"""
        async def _update():
            req = (
                PatchMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    PatchMessageRequestBody.builder()
                    .content(self._card_json(content, loading=False))
                    .build()
                )
                .build()
            )
            resp = await self._client.im.v1.message.apatch(req)
            if not resp.success():
                raise RuntimeError(f"patch 卡片失败: {resp.code} {resp.msg}")

        try:
            await self._retry_with_backoff(_update, max_retries=3)
        except Exception as e:
            # 更新失败仅打印警告，不中断主流程
            print(f"[feishu warn] 更新卡片最终失败: {e}", flush=True)

    async def send_text_to_user(self, open_id: str, text: str) -> str:
        """发送纯文本消息"""
        async def _send():
            req = (
                CreateMessageRequest.builder()
                .receive_id_type("open_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(open_id)
                    .msg_type("text")
                    .content(json.dumps({"text": text}, ensure_ascii=False))
                    .build()
                )
                .build()
            )
            resp = await self._client.im.v1.message.acreate(req)
            if not resp.success():
                raise RuntimeError(f"发送文本消息失败: {resp.code} {resp.msg}")
            return resp.data.message_id

        return await self._retry_with_backoff(_send, max_retries=3)

    # ------------------------------------------------------------------
    # 下载图片
    # ------------------------------------------------------------------
    async def download_image(self, message_id: str, image_key: str) -> str:
        """下载飞书图片到临时文件，返回本地路径"""
        import asyncio
        import ssl
        import urllib.request

        ctx = ssl.create_default_context()

        # 获取 tenant_access_token
        token_body = json.dumps({
            "app_id": self.config.app_id,
            "app_secret": self.config.app_secret
        }).encode()
        token_req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=token_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _get_token():
            with urllib.request.urlopen(token_req, context=ctx, timeout=10) as r:
                return json.loads(r.read())["tenant_access_token"]

        token = await asyncio.get_event_loop().run_in_executor(None, _get_token)

        # 下载图片
        url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{image_key}?type=image"
        img_req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})

        def _download():
            with urllib.request.urlopen(img_req, context=ctx, timeout=15) as r:
                ct = r.headers.get("Content-Type", "")
                ext = ".jpg"
                if "png" in ct:
                    ext = ".png"
                elif "gif" in ct:
                    ext = ".gif"
                elif "webp" in ct:
                    ext = ".webp"

                tmp_path = os.path.join(tempfile.gettempdir(), f"feishu-img-{int(time.time())}{ext}")
                with open(tmp_path, "wb") as f:
                    f.write(r.read())
                return tmp_path

        return await asyncio.get_event_loop().run_in_executor(None, _download)
