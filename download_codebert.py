#!/usr/bin/env python3
"""
下载 CodeBERT 模型脚本
"""
import os
import urllib.request
import ssl

# 禁用 SSL 验证（如果证书有问题）
ssl._create_default_https_context = ssl._create_unverified_context

# 模型文件列表
files = [
    "config.json",
    "pytorch_model.bin",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
]

base_url = "https://hf-mirror.com/microsoft/codebert-base/resolve/main"

# 下载目录
download_dir = "./models/codebert-base"
os.makedirs(download_dir, exist_ok=True)

print("开始下载 CodeBERT 模型...")
print(f"保存到: {os.path.abspath(download_dir)}")
print()

for filename in files:
    url = f"{base_url}/{filename}"
    filepath = os.path.join(download_dir, filename)
    
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {filename} 已存在 ({size} bytes)")
        continue
    
    print(f"⬇️  下载 {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"✅ {filename} 下载完成 ({size} bytes)")
    except Exception as e:
        print(f"❌ {filename} 下载失败: {e}")

print()
print("下载完成！")
print(f"文件列表:")
for f in os.listdir(download_dir):
    size = os.path.getsize(os.path.join(download_dir, f))
    print(f"  {f}: {size} bytes")
