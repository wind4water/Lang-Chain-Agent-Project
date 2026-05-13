# 确保工作目录是项目根目录（支持 python app/main.py 和 uvicorn 两种启动方式）
import os
import sys

# 获取项目根目录（main.py 所在目录的上级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# 必须先加载环境变量，再导入其他模块
from dotenv import load_dotenv
load_dotenv()

# 配置日志输出到控制台
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from app.agents.sqlite_with_tools import SqliteAgentWithTools
from app.rag import rag_system  # RAG 系统
from app.token_usage import request_token_scope, get_request_token_usage
import uvicorn
import os
import json

# 全局Agent实例
agent = None


def _get_agent():
    """供外部集成模块获取当前 Agent 实例。"""
    return agent


def _with_request_tokens(payload: dict) -> dict:
    """为接口返回追加本次请求 token 汇总字段。"""
    result = dict(payload)
    result["request_total_tokens"] = get_request_token_usage()["total_tokens"]
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global agent
    # 启动时初始化Agent（支持工具调用的SQLite版本）
    # 可以通过环境变量控制是否启用工具
    enable_tools = os.getenv("ENABLE_TOOLS", "true").lower() == "true"
    agent = SqliteAgentWithTools(enable_tools=enable_tools)
    print("✅ LangChain Agent initialized with Tool Calling + SQLite checkpoint support")

    # 初始化 RAG 系统（如果启用）
    try:
        await rag_system.initialize()
    except Exception as e:
        print(f"⚠️ RAG 系统初始化失败: {e}")
        print("   对话功能不受影响，RAG 功能不可用")

    yield
    # 关闭时清理（如果需要）
    print("🔄 Shutting down...")


# 创建FastAPI应用
app = FastAPI(
    title="LangChain对话Agent",
    description="基于LangChain和LangGraph的对话Agent，支持checkpoint会话管理",
    version="1.0.0",
    lifespan=lifespan
)



# 请求模型
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    session_id: str = Field(default="default", description="会话ID，用于区分不同用户会话")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI回复")
    session_id: str = Field(..., description="会话ID")
    request_total_tokens: int = Field(default=0, description="本次请求累计 token 数")


class HistoryResponse(BaseModel):
    session_id: str
    history: list
    request_total_tokens: int = Field(default=0, description="本次请求累计 token 数")


@app.get("/")
async def root():
    """根路径"""
    async with request_token_scope():
        return _with_request_tokens({
            "message": "LangChain对话Agent API (with Tool Calling + RAG)",
            "storage": "SQLite Persistent Storage",
            "features": [
                "Tool Calling - 自动调用工具完成任务",
                "Context Compression - 上下文压缩",
                "Persistent Storage - SQLite 持久化",
                "Multi-Session - 多会话管理",
                "Token Statistics - Token使用统计和成本追踪",
                "RAG - 检索增强生成（知识库问答）",
            ],
            "endpoints": {
                "POST /chat": "发送消息进行对话",
                "POST /chat/stream": "流式对话（Server-Sent Events）",
                "GET /history/{session_id}": "获取会话历史",
                "DELETE /history/{session_id}": "清除会话历史",
                "GET /sessions": "查看所有会话列表",
                "GET /database/stats": "查看数据库统计信息",
                "GET /database/sessions/{session_id}": "查看指定会话的详细信息",
                "GET /tools": "查看所有可用工具",
                "GET /stats/tokens/{session_id}": "查看指定会话的Token统计",
                "GET /stats/tokens/daily": "查看每日Token统计",
                "GET /stats/tokens/monthly": "查看每月Token统计",
                "GET /health": "健康检查",
                "POST /rag/query": "RAG 知识库问答",
                "POST /rag/rebuild": "重建知识库（全量）",
                "POST /rag/sync": "智能同步知识库（增量更新）⭐推荐",
                "GET /rag/stats": "RAG 系统统计信息",
            }
        })


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与Agent对话

    - **message**: 用户消息
    - **session_id**: 会话ID（可选，默认为"default"）
    """
    async with request_token_scope():
        try:
            response = await agent.chat(request.message, request.session_id)
            return ChatResponse(
                response=response,
                session_id=request.session_id,
                request_total_tokens=get_request_token_usage()["total_tokens"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理消息时出错: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    与Agent流式对话（Server-Sent Events）

    - **message**: 用户消息
    - **session_id**: 会话ID（可选，默认为"default"）

    返回流式响应，逐字输出AI回复
    """
    async def generate():
        async with request_token_scope():
            try:
                async for chunk in agent.chat_stream(request.message, request.session_id):
                    # SSE格式：data: {json}\n\n
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

                # 发送本次请求 token 汇总
                usage = get_request_token_usage()
                yield f"data: {json.dumps({'request_total_tokens': usage['total_tokens']}, ensure_ascii=False)}\n\n"

                # 发送完成标记
                yield "data: [DONE]\n\n"
            except Exception as e:
                # 发送错误信息
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用nginx缓冲
        }
    )



@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """
    获取指定会话的历史记录

    - **session_id**: 会话ID
    """
    async with request_token_scope():
        try:
            history = await agent.get_history(session_id)
            return HistoryResponse(
                session_id=session_id,
                history=history,
                request_total_tokens=get_request_token_usage()["total_tokens"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取历史记录时出错: {str(e)}")


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """
    清除指定会话的历史记录

    - **session_id**: 会话ID
    """
    async with request_token_scope():
        try:
            await agent.clear_history(session_id)
            return _with_request_tokens({"message": f"会话 {session_id} 的历史记录已清除"})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"清除历史记录时出错: {str(e)}")


@app.get("/health")
async def health():
    """健康检查"""
    async with request_token_scope():
        return _with_request_tokens({
        "status": "healthy",
        "agent_initialized": agent is not None,
        "storage_type": "SQLite",
        "tools_enabled": agent.enable_tools if agent else False,
        "tools_count": len(agent.tools) if agent else 0,
        "langfuse_enabled": agent.langfuse_enabled if agent else False,
        "langfuse_sample_rate": getattr(agent, "langfuse_sample_rate", 0.0) if agent else 0.0,
    })


@app.get("/tools")
async def list_tools():
    """
    列出所有可用的工具

    返回系统中注册的所有工具及其描述
    """
    async with request_token_scope():
        try:
            if agent is None:
                raise HTTPException(status_code=500, detail="Agent 未初始化")

            if not agent.enable_tools:
                return _with_request_tokens({
                    "enabled": False,
                    "message": "工具系统未启用",
                    "tools": []
                })

            tools = agent.list_available_tools()
            return _with_request_tokens({
                "enabled": True,
                "total": len(tools),
                "tools": tools
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取工具列表时出错: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """
    列出所有会话ID

    返回数据库中所有的session_id列表
    """
    async with request_token_scope():
        try:
            sessions = await agent.list_all_sessions()
            return _with_request_tokens({
                "total": len(sessions),
                "sessions": sessions
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"查询会话列表时出错: {str(e)}")


@app.get("/database/stats")
async def get_database_stats():
    """
    获取数据库统计信息

    返回：
    - 总会话数
    - 总checkpoint数
    - 数据库文件大小
    - 每个会话的详细信息
    """
    async with request_token_scope():
        try:
            stats = await agent.get_database_stats()
            return _with_request_tokens(stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取数据库统计信息时出错: {str(e)}")


@app.get("/database/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """
    获取指定会话的详细信息

    - **session_id**: 会话ID

    返回该会话的所有checkpoint详细信息
    """
    async with request_token_scope():
        try:
            detail = await agent.get_session_detail(session_id)
            return _with_request_tokens(detail)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取会话详细信息时出错: {str(e)}")


@app.get("/stats/tokens/{session_id}")
async def get_token_stats(session_id: str):
    """
    获取指定会话的Token使用统计

    - **session_id**: 会话ID

    返回：
    - 请求次数
    - 总Token数（提示词+完成）
    - 总成本（美元）
    - 按模型分组的详细信息
    """
    async with request_token_scope():
        try:
            stats = await agent.get_token_stats(session_id)
            return _with_request_tokens(stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取Token统计时出错: {str(e)}")


@app.get("/stats/tokens/daily")
async def get_daily_token_stats(date: str = None):
    """
    获取每日Token使用汇总

    - **date**: 日期 (YYYY-MM-DD)，默认为今天

    返回：
    - 唯一会话数
    - 总请求数
    - 总Token数
    - 总成本
    """
    async with request_token_scope():
        try:
            stats = await agent.get_daily_token_stats(date)
            return _with_request_tokens(stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取每日统计时出错: {str(e)}")


@app.get("/stats/tokens/monthly")
async def get_monthly_token_stats(year_month: str = None):
    """
    获取每月Token使用汇总

    - **year_month**: 年月 (YYYY-MM)，默认为当月

    返回：
    - 唯一会话数
    - 总请求数
    - 总Token数
    - 总成本
    """
    async with request_token_scope():
        try:
            stats = await agent.get_monthly_token_stats(year_month)
            return _with_request_tokens(stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取每月统计时出错: {str(e)}")


# ============================================================================
# RAG 相关接口
# ============================================================================

class RAGQueryRequest(BaseModel):
    """RAG 查询请求"""
    question: str = Field(..., description="用户问题")
    with_sources: bool = Field(default=True, description="是否返回来源信息")
    metadata_filter: dict | None = Field(
        default=None,
        description="可选的元数据过滤条件（Chroma filter），例如 {\"filename\": \"README.md\"}"
    )


class RAGQueryResponse(BaseModel):
    """RAG 查询响应"""
    answer: str = Field(..., description="答案")
    sources: list = Field(default=[], description="来源列表")
    source_count: int = Field(default=0, description="来源数量")


@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    """
    RAG 知识库问答

    - **question**: 用户问题
    - **with_sources**: 是否返回来源信息（默认 true）
    - **metadata_filter**: 可选元数据过滤条件（例如 `{\"extension\": \".md\"}`）

    返回基于知识库的答案及来源
    """
    async with request_token_scope():
        try:
            if not rag_system._initialized:
                raise HTTPException(
                    status_code=503,
                    detail="RAG 系统未初始化。请检查 RAG_ENABLED 配置或查看启动日志"
                )

            result = await rag_system.query(
                question=request.question,
                with_sources=request.with_sources,
                metadata_filter=request.metadata_filter
            )

            return _with_request_tokens(result)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG 查询失败: {str(e)}")


@app.post("/rag/rebuild")
async def rag_rebuild():
    """
    重建知识库（全量重建）

    清空并重新索引所有文档
    """
    async with request_token_scope():
        try:
            if not rag_system._initialized:
                raise HTTPException(
                    status_code=503,
                    detail="RAG 系统未初始化。请检查 RAG_ENABLED 配置"
                )

            result = await rag_system.rebuild_knowledge_base()
            return _with_request_tokens({
                "message": "知识库全量重建完成",
                **result
            })

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重建知识库失败: {str(e)}")


@app.post("/rag/sync")
async def rag_sync():
    """
    智能同步知识库（增量更新）⭐

    自动检测文档变更，只更新变化的部分：
    - 新增文档：索引新文档
    - 修改文档：删除旧版本，索引新版本
    - 删除文档：从索引中移除

    推荐：日常使用此接口，性能更好
    """
    async with request_token_scope():
        try:
            if not rag_system._initialized:
                raise HTTPException(
                    status_code=503,
                    detail="RAG 系统未初始化。请检查 RAG_ENABLED 配置"
                )

            result = await rag_system.sync_knowledge_base()

            if result["has_changes"]:
                return _with_request_tokens({
                    "message": "知识库智能同步完成",
                    **result
                })
            else:
                return _with_request_tokens({
                    "message": "知识库无变更，跳过同步",
                    **result
                })

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"同步知识库失败: {str(e)}")


@app.get("/rag/stats")
async def rag_stats():
    """
    获取 RAG 系统统计信息

    返回：
    - 系统状态
    - 文档数量
    - 配置信息
    """
    async with request_token_scope():
        try:
            stats = rag_system.get_stats()
            return _with_request_tokens(stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取 RAG 统计信息失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
