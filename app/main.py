from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from app.agents.sqlite_with_tools import SqliteAgentWithTools
from dotenv import load_dotenv
import uvicorn
import os
import json


# 加载环境变量
load_dotenv()

# 全局Agent实例
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global agent
    # 启动时初始化Agent（支持工具调用的SQLite版本）
    # 可以通过环境变量控制是否启用工具
    enable_tools = os.getenv("ENABLE_TOOLS", "true").lower() == "true"
    agent = SqliteAgentWithTools(enable_tools=enable_tools)
    print("✅ LangChain Agent initialized with Tool Calling + SQLite checkpoint support")
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


class HistoryResponse(BaseModel):
    session_id: str
    history: list


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LangChain对话Agent API (with Tool Calling)",
        "storage": "SQLite Persistent Storage",
        "features": [
            "Tool Calling - 自动调用工具完成任务",
            "Context Compression - 上下文压缩",
            "Persistent Storage - SQLite 持久化",
            "Multi-Session - 多会话管理",
            "Token Statistics - Token使用统计和成本追踪"
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
            "GET /health": "健康检查"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与Agent对话

    - **message**: 用户消息
    - **session_id**: 会话ID（可选，默认为"default"）
    """
    try:
        response = await agent.chat(request.message, request.session_id)
        return ChatResponse(
            response=response,
            session_id=request.session_id
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
        try:
            async for chunk in agent.chat_stream(request.message, request.session_id):
                # SSE格式：data: {json}\n\n
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

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
    try:
        history = await agent.get_history(session_id)
        return HistoryResponse(
            session_id=session_id,
            history=history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录时出错: {str(e)}")


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """
    清除指定会话的历史记录

    - **session_id**: 会话ID
    """
    try:
        await agent.clear_history(session_id)
        return {"message": f"会话 {session_id} 的历史记录已清除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除历史记录时出错: {str(e)}")


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "storage_type": "SQLite",
        "tools_enabled": agent.enable_tools if agent else False,
        "tools_count": len(agent.tools) if agent else 0
    }


@app.get("/tools")
async def list_tools():
    """
    列出所有可用的工具

    返回系统中注册的所有工具及其描述
    """
    try:
        if agent is None:
            raise HTTPException(status_code=500, detail="Agent 未初始化")

        if not agent.enable_tools:
            return {
                "enabled": False,
                "message": "工具系统未启用",
                "tools": []
            }

        tools = agent.list_available_tools()
        return {
            "enabled": True,
            "total": len(tools),
            "tools": tools
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工具列表时出错: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """
    列出所有会话ID

    返回数据库中所有的session_id列表
    """
    try:
        sessions = await agent.list_all_sessions()
        return {
            "total": len(sessions),
            "sessions": sessions
        }
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
    try:
        stats = await agent.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据库统计信息时出错: {str(e)}")


@app.get("/database/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """
    获取指定会话的详细信息

    - **session_id**: 会话ID

    返回该会话的所有checkpoint详细信息
    """
    try:
        detail = await agent.get_session_detail(session_id)
        return detail
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
    try:
        stats = await agent.get_token_stats(session_id)
        return stats
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
    try:
        stats = await agent.get_daily_token_stats(date)
        return stats
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
    try:
        stats = await agent.get_monthly_token_stats(year_month)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取每月统计时出错: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
