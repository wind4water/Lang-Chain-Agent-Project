"""
飞书事件处理器
WebSocket 长连接模式，接收飞书消息并调用 Agent 回复
参考 feishu-claude-code 的实现方式
"""
import asyncio
import os
import ssl
import time
from typing import Optional, Callable, Awaitable

import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

from app.integrations.feishu_config import FeishuConfig
from app.integrations.feishu_client import FeishuClient


# per-chat 消息队列锁，保证同一群组的消息串行处理，允许不同群组并发
_chat_locks: dict[str, asyncio.Lock] = {}
_MAX_CHAT_LOCKS = 200

# 记录最后事件时间（用于看门狗）
_last_event_time: float = 0.0


def _get_chat_lock(chat_id: str) -> asyncio.Lock:
    """获取指定 chat 的锁"""
    if chat_id not in _chat_locks:
        if len(_chat_locks) >= _MAX_CHAT_LOCKS:
            _chat_locks.clear()
        _chat_locks[chat_id] = asyncio.Lock()
    return _chat_locks[chat_id]


def extract_chat_info(event: P2ImMessageReceiveV1) -> tuple[str, str, bool]:
    """从事件中提取用户信息

    Returns:
        (user_id, chat_id, is_group)
    """
    sender = event.event.sender
    user_id = sender.sender_id.open_id

    message = event.event.message
    chat_type = message.chat_type
    chat_id_raw = message.chat_id

    is_group = (chat_type == "group")
    chat_id = chat_id_raw if is_group else user_id

    return user_id, chat_id, is_group


def extract_message_text(event: P2ImMessageReceiveV1) -> tuple[str, Optional[str]]:
    """提取消息文本内容和图片 key

    Returns:
        (text, image_key) - image_key 为 None 表示没有图片
    """
    message = event.event.message
    msg_type = message.message_type
    content = message.content

    image_key = None

    if msg_type == "text":
        import json
        try:
            data = json.loads(content)
            text = data.get("text", "")
        except json.JSONDecodeError:
            text = content
    elif msg_type == "image":
        import json
        try:
            data = json.loads(content)
            image_key = data.get("image_key", "")
        except json.JSONDecodeError:
            image_key = content
        text = "[图片]"
    else:
        text = f"[不支持的消息类型: {msg_type}]"

    return text, image_key


class FeishuHandler:
    """飞书消息处理器"""

    def __init__(
        self,
        config: FeishuConfig,
        client: FeishuClient,
        get_agent: Callable[[], Optional[object]],
    ):
        self.config = config
        self.client = client
        self.get_agent = get_agent
        self._ws_client: Optional[lark.ws.Client] = None
        self._running = False

    # ------------------------------------------------------------------
    # 事件回调
    # ------------------------------------------------------------------
    def on_message_receive(self, data: P2ImMessageReceiveV1) -> None:
        """飞书 SDK 同步回调 - 调度到异步任务"""
        global _last_event_time
        _last_event_time = time.time()
        asyncio.ensure_future(self._handle_message_async(data))

    # ------------------------------------------------------------------
    # 异步消息处理
    # ------------------------------------------------------------------
    async def _handle_message_async(self, data: P2ImMessageReceiveV1) -> None:
        """异步处理飞书消息"""
        user_id, chat_id, is_group = extract_chat_info(data)
        text, image_key = extract_message_text(data)

        # 群聊只响应 @机器人的消息
        if is_group:
            mentions = getattr(data.event.message, 'mentions', None) or []
            if not mentions:
                return  # 没有 @mention，忽略

            # 去掉 @mention 占位符
            for mention in mentions:
                key = getattr(mention, 'key', '')
                if key:
                    text = text.replace(key, '').strip()

        if not text and not image_key:
            return

        # 获取 per-chat 锁
        lock = _get_chat_lock(chat_id)
        async with lock:
            await self._process_message(user_id, chat_id, is_group, text, image_key, data)

    async def _process_message(
        self,
        user_id: str,
        chat_id: str,
        is_group: bool,
        text: str,
        image_key: Optional[str],
        event: P2ImMessageReceiveV1,
    ) -> None:
        """实际处理消息逻辑"""
        agent = self.get_agent()
        if agent is None:
            print("[feishu] Agent 未初始化，无法处理消息", flush=True)
            return

        # 构建 session_id（飞书用户/群组唯一标识）
        session_id = f"feishu:{chat_id}"

        # 处理图片
        if image_key and image_key != "[图片]":
            try:
                tmp_path = await self.client.download_image(
                    event.event.message.message_id,
                    image_key
                )
                text = f"[用户发送了图片，本地路径: {tmp_path}]\n{text}"
            except Exception as e:
                print(f"[feishu] 图片下载失败: {e}", flush=True)
                text = f"[图片下载失败]\n{text}"

        # 发送"思考中"占位卡片
        try:
            if is_group:
                card_msg_id = await self.client.reply_card(
                    event.event.message.message_id,
                    content="🤔 思考中...",
                    loading=True,
                )
            else:
                card_msg_id = await self.client.send_card_to_user(
                    user_id,
                    content="🤔 思考中...",
                    loading=True,
                )
        except Exception as e:
            print(f"[feishu] 发送占位卡片失败: {e}", flush=True)
            return

        # 流式调用 Agent
        accumulated = ""
        chars_since_push = 0
        last_push_time = time.time()

        try:
            async for chunk in agent.chat_stream(text, session_id):
                accumulated += chunk
                chars_since_push += len(chunk)

                # 按字符数或时间间隔推送
                now = time.time()
                interval_sec = self.config.stream_interval_ms / 1000.0

                if (chars_since_push >= self.config.stream_chunk_size or
                        now - last_push_time >= interval_sec):
                    await self.client.update_card(card_msg_id, accumulated)
                    chars_since_push = 0
                    last_push_time = now

            # 最终更新
            if accumulated:
                await self.client.update_card(card_msg_id, accumulated)
            else:
                await self.client.update_card(card_msg_id, "（无回复内容）")

        except Exception as e:
            print(f"[feishu] 流式处理失败: {e}", flush=True)
            try:
                await self.client.update_card(card_msg_id, f"❌ 处理出错: {e}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动 WebSocket 长连接（在后台线程中运行）"""
        if not self.config.is_configured():
            print("[feishu] 飞书未启用或配置不完整，跳过启动")
            return

        if self._running:
            print("[feishu] 已经在运行中")
            return

        self._running = True

        # 如果不验证 SSL，设置环境变量和创建自定义 SSL 上下文
        if not self.config.ssl_verify:
            print("[feishu] SSL 验证已禁用（FEISHU_SSL_VERIFY=false）")
            # 设置环境变量影响底层 websocket 库
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            # 为当前线程创建不验证的 SSL 上下文
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # 构建事件处理器
        handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self.on_message_receive) \
            .build()

        # 创建 WebSocket 客户端
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )

        # 在后台线程启动（不阻塞主线程）
        import threading
        def _run():
            # 在此线程中设置 SSL 上下文（如果不验证）
            if not self.config.ssl_verify:
                # 创建不验证的默认 SSL 上下文
                _original_create_default_context = ssl.create_default_context
                def _create_unverified_context(*args, **kwargs):
                    context = _original_create_default_context(*args, **kwargs)
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    return context
                ssl._create_default_https_context = _create_unverified_context
                ssl.create_default_context = _create_unverified_context

            try:
                print(f"[feishu] WebSocket 长连接已启动 (app_id={self.config.app_id[:8]}...)")
                self._ws_client.start()
            except Exception as e:
                print(f"[feishu] WebSocket 连接异常: {e}", flush=True)
            finally:
                self._running = False

        self._ws_thread = threading.Thread(target=_run, name="feishu-ws", daemon=True)
        self._ws_thread.start()

    def stop(self) -> None:
        """停止 WebSocket 长连接"""
        if not self._running:
            return

        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
                print("[feishu] WebSocket 长连接已停止")
            except Exception as e:
                print(f"[feishu] 停止连接时出错: {e}", flush=True)

        # 等待线程结束
        if hasattr(self, '_ws_thread') and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
