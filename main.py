"""
向后兼容层 - main.py
实际实现已移至 app/main.py

使用方式：
    python main.py  # 仍然可以像以前一样启动
"""
from app.main import app

# 保持兼容
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # 指向这个兼容层
        host="0.0.0.0",
        port=8000,
        reload=True
    )
