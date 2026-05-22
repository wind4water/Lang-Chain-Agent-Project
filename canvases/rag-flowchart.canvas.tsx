import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Spacer,
  Stack,
  Text,
  useHostTheme,
  useCanvasState,
  computeDAGLayout
} from 'cursor/canvas';

// 节点详细信息
const NODE_DETAILS: Record<string, {
  title: string;
  phase: string;
  description: string;
  files: string[];
  key_logic: string;
  why_exists: string;
  logs_to_watch: string;
}> = {
  request_in: {
    title: "1. API 请求接收",
    phase: "请求接入层",
    description: "FastAPI 接收到 POST /rag/query 请求，包含 question（问题），with_sources（是否包含来源）和可选的 metadata_filter。",
    files: ["app/main.py"],
    key_logic: "async def rag_query(request: RAGQueryRequest):\n    async with request_token_scope():\n        result = await rag_system.query(\n            question=request.question,\n            with_sources=request.with_sources,\n            metadata_filter=request.metadata_filter\n        )",
    why_exists: "作为系统唯一的 RAG 问答入口，在此处会启动请求级别的 token 追踪作用域（request_token_scope），统一记录此请求的总 token 消耗与 LLM 计费成本。",
    logs_to_watch: "控制台打印 `POST /rag/query` 访问，以及请求处理时间。"
  },
  check_init: {
    title: "2. 初始化校验",
    phase: "健康保护层",
    description: "校验 RAG 系统是否已经成功完成了嵌入模型、向量库（Chroma）和全文检索（ES）等底层模块的加载。",
    files: ["app/rag/system.py"],
    key_logic: "if not self._initialized:\n    raise RuntimeError(\"RAG 系统未初始化\")",
    why_exists: "避免底层由于缺乏模型缓存、网络断开等原因导致服务崩溃，保证请求安全快速失败。如果未启用，对话 Agent 在调用 RAG 工具时也会降级处理。",
    logs_to_watch: "若未初始化，会返回 `503 Service Unavailable`。"
  },
  route_chain: {
    title: "3. 检索链路路由",
    phase: "查询隔离层",
    description: "检查本次请求是否包含元数据过滤器。若包含，会基于过滤器现场动态编译一个临时的 Retriever 与 RAG 链；否则复用系统启动时的常驻全局 Chain。",
    files: ["app/rag/system.py"],
    key_logic: "query_chain = self.rag_chain\nif metadata_filter:\n    retriever = self._build_retriever(metadata_filter=metadata_filter)\n    query_chain = RAGChain(\n        llm=self.llm,\n        retriever=retriever\n    )",
    why_exists: "因为 Chroma 和 Elasticsearch 在多租户、多项目隔离检索时，需要通过 metadata 进行过滤。动态构建 chain 能保证查询彼此物理隔离、不发生数据越界偏色。",
    logs_to_watch: "控制台输出：`[RAG Filter] 检测到活跃的元数据过滤条件: {'filename': '...'}`。"
  },
  retrieval: {
    title: "4. 混合检索调度",
    phase: "双路召回总控",
    description: "启动检索器。系统根据 `RAG_RETRIEVAL_MODE` 变量（当前为 `hybrid`）选择启动单路向量，或者多路混合双重检索并进行排序融合。",
    files: ["app/rag/core/chain.py", "app/rag/core/hybrid_retriever.py"],
    key_logic: "docs = await self._retrieve_with_fallback_async(question)\n# 最终分发到 HybridRetriever._hybrid_search(query)",
    why_exists: "通过中央调度，规范首轮检索与后置重排序逻辑。保证单次召回具有最高的可扩展性，易于植入其他异构数据库（如 Redis、Milvus 等）。",
    logs_to_watch: "日志显示 `RAG 异步查询（含来源）`，并开始执行检索。"
  },
  vector_search: {
    title: "4.1 向量检索 (Chroma)",
    phase: "语义召回双轨之一",
    description: "调用向量数据库 Chroma 执行密集向量检索。它会利用本地缓存的 BGE/智谱等嵌入模型将文本转为特征向量，进行余弦相似度匹配，设定硬阈值 score_threshold=0.5。",
    files: ["app/rag/core/vectorstore.py"],
    key_logic: "retriever = self.vectorstore_manager.get_retriever(\n    search_type=self.search_type,\n    k=self.k,\n    score_threshold=self.score_threshold,\n    metadata_filter=self.metadata_filter,\n)\ndocs = retriever.invoke(query)",
    why_exists: "专门用于捕捉自然语言的泛化语义。即使拼写不完全一致、使用了同义词（例如“口头禅”与“常说的话”），依然能通过向量距离计算被正确召回。",
    logs_to_watch: "控制台输出：`Hybrid 向量检索开始 / 完成: hits=N`。若 Chroma 匿名遥测报错可在此观察。"
  },
  keyword_search: {
    title: "4.2 关键词检索 (ES)",
    phase: "语义召回双轨之二",
    description: "调用 Elasticsearch 里的 `rag_keyword_chunks` 索引执行 BM25 文本匹配检索。如果 ES 不可用，则会自动无缝降级到本地轻量级词频重叠召回（Token-Overlap）。",
    files: ["app/rag/core/es_keyword_retriever.py"],
    key_logic: "if self.es_keyword_retriever and self.es_keyword_retriever.available:\n    docs = self.es_keyword_retriever.search(query=query, k=self.keyword_k)\nelse:\n    docs = self._keyword_search_local(query)",
    why_exists: "用于解决专有名词、API 接口名、拼音缩写、特定英文变量等向量匹配易失效的场景。比如精准检索 `ThirdPayTipWordOutput` 时表现极佳。",
    logs_to_watch: "控制台输出：`ES 关键词召回开始` 接着 `POST http://127.0.0.1:9200/... [status:200]` 打印命中结果数和最高分。"
  },
  hybrid_fusion: {
    title: "4.3 加权排序融合 (RRF)",
    phase: "双路数据融归",
    description: "计算各文档在两个检索队列（向量和文本）中的表现，使用加权名次评分（向量权重=2.0，文本权重=1.0）进行线性综合重排，去重后截取前 Top_K（默认=4）个高分片段。",
    files: ["app/rag/core/hybrid_retriever.py"],
    key_logic: "for i, doc in enumerate(vector_docs):\n    score_map[key] = score_map.get(key, 0.0) + (self.VECTOR_WEIGHT * (len(vector_docs) - i))\nfor i, doc in enumerate(keyword_docs):\n    score_map[key] = score_map.get(key, 0.0) + (self.KEYWORD_WEIGHT * (len(keyword_docs) - i))",
    why_exists: "由于不同数据库打分标准不同（Chroma相似度 vs ES的BM25无界分数），无法直接相加。采用交叉加权名次融合（Reciprocal Rank Fusion 变体）能最大程度平衡两种检索器的准确度。",
    logs_to_watch: "控制台输出：`Hybrid 融合检索完成: merged_hits=4, top_docs=['file1.java', 'file2.md'], top_scores=[8.0, 7.0]`。"
  },
  structured_check: {
    title: "5. 结构化提问检测",
    phase: "意图增强层",
    description: "使用预设正则表达式或特定字段模式（如下划线、大写字母、包含 dot、以及 hint 关键词如 enum、字段等）检测该问题是否偏向技术结构、数据库 schema 查询。",
    files: ["app/rag/core/chain.py"],
    key_logic: "def _is_structured_question(cls, question: str) -> bool:\n    q = (question or \"\").lower()\n    if \".\" in q or \"_\" in q: return True\n    return any(hint in q for hint in cls.STRUCTURED_QUERY_HINTS)",
    why_exists: "技术类、数据库设计类或代码级别的问答极难单次召回，一旦错过依赖关系（如接口与它的实现类分别在两个文件），就会导致拒答。检测到此类意图会触发深度检索。",
    logs_to_watch: "控制台输出补充检索（同步/异步）标识。"
  },
  fallback_expansion: {
    title: "6. 多路查询扩展 (Expansion)",
    phase: "深度重检层",
    description: "如果确认是结构化提问，且第一轮召回得到的文档片段不足 12 个（MAX_CONTEXT_DOCS），系统会自动生成 3 个英文和中文变体 Query，执行并发重检索并合并去重。",
    files: ["app/rag/core/chain.py"],
    key_logic: "for query in self._build_expanded_queries(question):\n    extra = await self._retrieve_once_async(query)\n    expanded_docs = self._merge_unique_docs(expanded_docs, extra, 12)",
    why_exists: "在代码问答场景下，“查询漂移”和“上下文缺失”非常严重。通过向向量和 ES 并发查询“{q} 定义/枚举/取值/values/list”等多个方向，能一网打尽所有零散的定义片段。",
    logs_to_watch: "控制台输出：`RAG 补充检索（异步）: base_docs=4, merged_docs=10`。"
  },
  hint_inject: {
    title: "7. 运行时提示注入",
    phase: "知识图谱微校准",
    description: "检测问题是否针对 attachAttributes 类枚举字段。如果是，会利用正则从已召回代码中提取对应的 Java 枚举常量名，拼装为一个临时的虚拟文本文档注入到上下文头部。",
    files: ["app/rag/core/chain.py"],
    key_logic: "if not self._is_attach_attributes_question(question): return docs\ncandidates = self._collect_attach_attribute_candidates(docs)\nhint_doc = Document(page_content=..., metadata={'filename': 'RAGDerivedAttachAttributes.txt'})\nreturn self._merge_unique_docs([hint_doc], docs, 12)",
    why_exists: "由于大模型在处理晦涩的代码命名和多层级依赖时容易犯晕，通过硬编码的提取规则，相当于临时制作了一张“答案卡片”递给模型，能瞬间纠正模型的判断，避免幻觉拒答。",
    logs_to_watch: "若满足条件，来源列表里会出现虚拟生成的 `RAGDerivedAttachAttributes.txt` 来源。"
  },
  prompt_compile: {
    title: "8. 组装 Prompt",
    phase: "语义映射层",
    description: "将搜集去重后的所有文档片段进行格式化。每个片段都会被统一渲染为：`【文档 X】（来源: xxx.md）\\n [文本内容]`，并附加到精心设计的 System Prompt 中。",
    files: ["app/rag/core/chain.py"],
    key_logic: "context = self._format_docs(docs)\nmessages = prompt.format_messages(context=context, question=question)",
    why_exists: "统一、标准的上下文容器格式能够让模型以极高精度抓取分块细节。通过在 prompt 中强行限制大模型“仅能基于上下文回答，不要胡乱发挥”，保证企业级问答的严肃性与高可信度。",
    logs_to_watch: "日志输出提示 RAG Prompt 的 context 变量拼装大小。"
  },
  llm_generate: {
    title: "9. LLM 答案生成",
    phase: "大脑生成层",
    description: "调用 ChatOpenAI 初始化的大语言模型（目前线上使用 `glm-4.5-air`），以 temperature=0.7 发射并等待模型生成回答文本。",
    files: ["app/rag/core/chain.py", "app/rag/system.py"],
    key_logic: "response = await self.llm.ainvoke(messages)\nanswer = StrOutputParser().invoke(response)",
    why_exists: "借助基座模型强大的中文语言组织、推理和阅读理解能力，把冰冷杂乱的代码和文档碎片组织成人类易读、高可读性的流畅回答。",
    logs_to_watch: "此时由于我们改写了 callbacks（禁用 callback trace 避免分裂，采用手动单根 trace 模式），它会开始调用 LLM 的 API。"
  },
  refusal_check: {
    title: "10. 拒答行为检测",
    phase: "质量守卫层",
    description: "检查模型返回的最终答案是否包含预置的敏感语，例如“根据现有资料无法回答该问题”、“没有相关信息”等拒答关键词。",
    files: ["app/rag/core/chain.py"],
    key_logic: "def _is_refusal_answer(cls, answer: str) -> bool:\n    normalized = re.sub(r\"\\s+\", \"\", (answer or \"\").lower())\n    return any(marker in normalized for marker in cls.REFUSAL_MARKERS)",
    why_exists: "这是 RAG 非常优雅的自我救赎机制。有时第一轮精确检索由于 Top_K 太小没有检索到核心事实，与其直接把“无法回答”丢给用户，不如立刻捕获并做强力补救检索。",
    logs_to_watch: "如果命中，会触发下一个 `boost_retry` 步骤。"
  },
  boost_retry: {
    title: "11. 强力重试检索 (Fallback Boost)",
    phase: "终极兜底检索层",
    description: "一旦检测到“拒答”，在满足拥有基础召回的前提下，系统会强行抛开意图检测分类器，全量开启 `force_boost=True`。将之前跳过的“多路查询扩展”和“属性枚举提取”等流程重新暴利扫荡一遍，获取更多背景并交由大模型二次生成。",
    files: ["app/rag/core/chain.py"],
    key_logic: "if self._is_refusal_answer(answer) and docs:\n    retry_docs = await self._retrieve_with_fallback_async(question, force_boost=True)\n    retry_answer = await self._ainvoke_from_docs(question, retry_docs)",
    why_exists: "最大限度降低系统的“假阴性率”。用高频、并发的扫射检索，拯救因第一轮高精度剪枝过滤被冤枉刷掉的有用知识文档。",
    logs_to_watch: "控制台输出：`RAG 异步拒答兜底重试成功: docs=12` 或失败记录。"
  },
  source_dedup: {
    title: "12. 来源规范去重",
    phase: "参考标准格式化",
    description: "整理在整个链条中发挥最终作用的文本块，按照底层绝对路径进行合并去重，每个源文件提取首个最高频片段，并保留前 100 字作为 Rel_Source 预览。",
    files: ["app/rag/core/chain.py"],
    key_logic: "sources = []\nseen_files = set()\nfor doc in docs:\n    if source not in seen_files:\n        seen_files.add(source)\n        sources.append({\"filename\": filename, \"source\": source, \"content_preview\": ...})",
    why_exists: "保证用户可以一目了然看清所有的参考文件。如果某个大类引用了多个代码块，只显示不重合的干净文件目录，使界面更加美观规整。",
    logs_to_watch: "控制台输出：`RAG 回答生成成功，使用了 N 个来源`。"
  },
  done: {
    title: "13. 格式化 JSON 响应",
    phase: "响应收尾层",
    description: "最终统计调用中产生的 LLM 输入与输出总 Token 数并汇总上报到 Token Tracker（对于 `/chat` 也会被挂载到单根手动 trace），格式化返回给 API 客户端。",
    files: ["app/main.py"],
    key_logic: "return {\n    \"answer\": result[\"answer\"],\n    \"sources\": result[\"sources\"],\n    \"source_count\": result[\"source_count\"],\n    \"request_total_tokens\": get_request_token_usage()[\"total_tokens\"]\n}",
    why_exists: "向前端或调用 Agent 回执状态，包含最详尽的追溯物证。同时提供流式 (SSE) 及普通响应两种完美的格式分发。",
    logs_to_watch: "控制台最终打印绿色勾号：`[Tool End] 工具执行完毕: knowledge_base_search`。"
  }
};

const nodesInput = Object.keys(NODE_DETAILS).map(id => ({ id }));
const edgesInput = [
  { from: "request_in", to: "check_init" },
  { from: "check_init", to: "route_chain" },
  { from: "route_chain", to: "retrieval" },
  { from: "retrieval", to: "vector_search" },
  { from: "retrieval", to: "keyword_search" },
  { from: "vector_search", to: "hybrid_fusion" },
  { from: "keyword_search", to: "hybrid_fusion" },
  { from: "hybrid_fusion", to: "structured_check" },
  { from: "structured_check", to: "fallback_expansion" },
  { from: "structured_check", to: "hint_inject" },
  { from: "fallback_expansion", to: "hint_inject" },
  { from: "hint_inject", to: "prompt_compile" },
  { from: "prompt_compile", to: "llm_generate" },
  { from: "llm_generate", to: "refusal_check" },
  { from: "refusal_check", to: "boost_retry" },
  { from: "refusal_check", to: "source_dedup" },
  { from: "boost_retry", to: "source_dedup" },
  { from: "source_dedup", to: "done" }
];

export default function RAGQueryFlowchart() {
  const theme = useHostTheme();
  const [selectedNodeId, setSelectedNodeId] = useCanvasState("selectedNodeId", "request_in");

  // 使用 DAG Layout 计算几何位置
  const layout = computeDAGLayout({
    nodes: nodesInput,
    edges: edgesInput,
    direction: "vertical",
    nodeWidth: 190,
    nodeHeight: 55,
    rankGap: 45,
    nodeGap: 30,
    padding: 24
  });

  const selectedNode = NODE_DETAILS[selectedNodeId] || NODE_DETAILS.request_in;

  return (
    <div style={{ height: "100%", background: theme.bg.editor, color: theme.text.primary, display: "flex", flexDirection: "column" }}>
      {/* 头部标题区 */}
      <div style={{ padding: "16px 24px", borderBottom: `1px solid ${theme.stroke.secondary}`, flexShrink: 0 }}>
        <H1 style={{ margin: 0, fontSize: "20px" }}>RAG 系统全链路问答解析图</H1>
        <Text tone="secondary" size="small" style={{ marginTop: "4px" }}>
          基于系统中 `/rag/query` 的实际调用链路与底层框架代码生成。点击左侧流程图中的节点，可在右侧深度探究源码逻辑与设计细节。
        </Text>
      </div>

      {/* 主体分栏区 */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* 左分栏：交互 DAG 渲染 */}
        <div style={{ width: "55%", overflow: "auto", position: "relative", borderRight: `1px solid ${theme.stroke.secondary}`, padding: "16px" }}>
          <div style={{ position: "relative", width: layout.width, height: layout.height, margin: "0 auto" }}>
            {/* SVG 线条绘制 */}
            <svg width={layout.width} height={layout.height} style={{ position: "absolute", top: 0, left: 0, overflow: "visible", pointerEvents: "none" }}>
              <defs>
                <marker
                  id="arrow"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="5"
                  markerHeight="5"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill={theme.text.quaternary} />
                </marker>
                <marker
                  id="arrow-active"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="5"
                  markerHeight="5"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill={theme.accent.primary} />
                </marker>
              </defs>
              {layout.edges.map((edge, i) => {
                const isActive = selectedNodeId === edge.from || selectedNodeId === edge.to;
                return (
                  <path
                    key={i}
                    d={`M ${edge.sourceX} ${edge.sourceY} L ${edge.targetX} ${edge.targetY}`}
                    stroke={isActive ? theme.accent.primary : theme.stroke.primary}
                    strokeWidth={isActive ? 2 : 1}
                    fill="none"
                    markerEnd={`url(#${isActive ? "arrow-active" : "arrow"})`}
                  />
                );
              })}
            </svg>

            {/* position:absolute 容器卡片 */}
            {layout.nodes.map((node) => {
              const isSelected = selectedNodeId === node.id;
              const nodeInfo = NODE_DETAILS[node.id];
              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  style={{
                    position: "absolute",
                    left: node.x,
                    top: node.y,
                    width: 190,
                    height: 55,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    borderRadius: "6px",
                    border: `1px solid ${isSelected ? theme.accent.primary : theme.stroke.primary}`,
                    background: isSelected ? theme.fill.quaternary : theme.bg.elevated,
                    cursor: "pointer",
                    padding: "4px 8px",
                    textAlign: "center",
                    userSelect: "none",
                    boxSizing: "border-box",
                    transition: "all 0.15s ease-in-out"
                  }}
                >
                  <Text
                    weight={isSelected ? "semibold" : "normal"}
                    size="small"
                    style={{
                      color: isSelected ? theme.accent.primary : theme.text.primary,
                      fontSize: "12px",
                      lineHeight: "1.2"
                    }}
                  >
                    {nodeInfo.title}
                  </Text>
                  <Text tone="secondary" style={{ fontSize: "9px", marginTop: "2px" }}>
                    {nodeInfo.phase}
                  </Text>
                </div>
              );
            })}
          </div>
        </div>

        {/* 右分栏：节点详情与源码精读 */}
        <div style={{ width: "45%", overflow: "auto", padding: "24px", background: theme.bg.chrome }}>
          <Stack gap={20}>
            {/* 步骤卡片 */}
            <Card>
              <CardHeader trailing={<Pill tone="info" active>{selectedNode.phase}</Pill>}>
                详细解析
              </CardHeader>
              <CardBody>
                <Stack gap={12}>
                  <H2 style={{ margin: 0, fontSize: "16px" }}>{selectedNode.title}</H2>
                  <Text style={{ lineHeight: "1.5" }}>{selectedNode.description}</Text>
                  
                  <Divider />
                  
                  {/* 代码文件 */}
                  <div>
                    <Text tone="secondary" size="small" weight="semibold">涉及文件:</Text>
                    <Row gap={6} style={{ marginTop: "4px", flexWrap: "wrap" }}>
                      {selectedNode.files.map((file, i) => (
                        <Pill key={i} size="sm" tone="neutral">{file}</Pill>
                      ))}
                    </Row>
                  </div>
                </Stack>
              </CardBody>
            </Card>

            {/* 设计意图 */}
            <Card>
              <CardHeader>
                设计意图
              </CardHeader>
              <CardBody>
                <Text style={{ lineHeight: "1.5" }}>{selectedNode.why_exists}</Text>
              </CardBody>
            </Card>

            {/* 关键源码精读 */}
            <Card>
              <CardHeader>
                关键代码实现片段
              </CardHeader>
              <CardBody style={{ padding: 0 }}>
                <pre style={{
                  margin: 0,
                  padding: "12px",
                  fontSize: "11px",
                  fontFamily: "monospace",
                  lineHeight: "1.4",
                  overflowX: "auto",
                  background: theme.bg.editor,
                  color: theme.text.primary,
                }}>
                  <code>{selectedNode.key_logic}</code>
                </pre>
              </CardBody>
            </Card>

            {/* 调试日志 */}
            <Card>
              <CardHeader>
                运行日志指引
              </CardHeader>
              <CardBody>
                <Text tone="secondary" style={{ lineHeight: "1.4" }}>
                  {selectedNode.logs_to_watch}
                </Text>
              </CardBody>
            </Card>
          </Stack>
        </div>
      </div>
    </div>
  );
}
