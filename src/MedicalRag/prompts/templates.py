"""
Prompt模板管理（保持与原项目一致）
"""

from typing import Dict, Any, Union

# =============================================================================
# 基础RAG提示模板
# =============================================================================
BASIC_RAG_SYSTEM_PROMPT = (
    """你是一名专业的医学知识助手，能够基于提供的医学资料准确回答用户问题。"""
)

BASIC_RAG_USER_PROMPT = """# 要求
1. 必须严格基于提供的参考资料回答问题
2. 回答要专业、准确，同时通俗易懂，不需要长篇大论
3. 不要编造或推测未在资料中提及的信息
4. 如涉及具体诊疗建议，请提醒用户咨询专业医生
# 参考资料
{all_document_str}
# 用户问题
{input}
请基于以上参考资料回答用户问题。"""

# =============================================================================
# 多轮对话RAG提示模板
# =============================================================================
MULTI_DIALOGUE_RAG_SYSTEM_PROMPT = """你是一名专业的医学知识助手，能够基于参考文档事实及上下文提供的信息准确的回答用户问题。
{running_summary}"""

MULTI_DIALOGUE_RAG_USER_PROMPT = """# 要求
1. 必须严格基于提供的参考资料及上下文进行事实回答
2. 如果参考资料中没有相关信息，请明确拒绝回答，向用户说明："根据提供的资料无法回答此问题"
3. 回答要专业、准确、简洁
4. 不要编造或推测未在参考资料中提及的信息
# 参考资料
{all_document_str}
# 用户问题
{llm_rewritten_content}
"""

# =============================================================================
# 查询改写提示模板
# =============================================================================
QUERY_REWRITER_SYSTEM_PROMPT = """你是一个检索查询改写器。参考对话历史（如果有）和用户输入，把用户的最后问题改写成独立、自洽、便于进行单次且明确的向量检索查询文本。请不要进行过于复杂的主观推断和胡编乱造。如果用户的查询足够清晰，可不做任何处理。"""
QUERY_REWRITER_USER_PROMPT = "请只输出改写后的查询：\n{original_input}"

REPHRASE_SYSTEM_PROMPT = """根据聊天历史和用户最新问题，将问题改写为可独立检索的问题。若聊天历史为空或最新问题已足够独立清晰，则直接返回原问题。只输出改写后的问题文本。"""
REPHRASE_USER_PROMPT = """聊天历史：
{chat_history}

用户问题：
{question}

独立检索问题："""

SEARCH_TOOLS_SYSTEM_PROMPT = """searxng_search 的工具参数如下：
query (str): 搜索查询词
num_results (int): 返回结果数量
language (Optional[str]): 语言，例如 zh-CN / en-US
safesearch (Optional[int]): 0关闭，1中等，2严格
time_range (Optional[str]): day/week/month/year
categories (Optional[List[str]]): general/images/news/map/music/it/science/files

你的任务：根据用户查询选择最合适的 searxng_search 参数。必须只返回一个工具调用。"""
SEARCH_TOOLS_USER_PROMPT = """请为以下问题选择最合适的 searxng_search 工具参数：
{query}

要求：
1) num_results 建议在 {min_results}~{max_results} 之间
2) 根据问题时效性设置 time_range
3) 根据问题性质设置 categories（可多选）
4) 如果问题存在安全风险，自行提高 safesearch
当前时间：{current_date}
"""

SELECT_BEST_RESULT_SYSTEM_PROMPT = """你是一名搜索结果筛选助手。请从候选搜索结果中挑选最适合回答问题的结果 ID 列表。
只输出 ID 列表，例如：[1,2,3]。不要输出额外解释。"""
SELECT_BEST_RESULT_USER_PROMPT = """问题：{query}

候选结果：
{context}

要求：
1) 只从候选结果中选择，不能虚构 ID
2) 结果数量不超过 {top_k}
3) 优先选择信息完整、来源可信、与问题直接相关的条目"""

RESPONSE_SYNTH_SYSTEM_PROMPT = """你是一位专业的医学研究助手。请严格基于给定上下文进行回答，禁止编造。
输出要求：
1) 回答简洁、专业、可读
2) 可用分点表达
3) 若上下文不足，请明确说明信息不足
4) 如涉及诊疗建议，提醒用户线下就医
"""
RESPONSE_SYNTH_USER_PROMPT = """# 历史对话
{chat_history}

# 检索上下文
{all_document_str}

# 当前问题
{input}

请根据上下文生成最终回答。"""

# =============================================================================
# 摘要生成提示模板
# =============================================================================
SUMMARY_SYSTEM_PROMPT = """你是一名专业的医疗对话摘要助手。你的任务是根据用户与AI助手的多轮医药对话，生成一份简明的对话摘要，既要保留医学相关的关键信息，又要保证新一轮对话衔接自然，让用户不会感觉对话被中断。  """
SUMMARY_USER_PROMPT = """请遵循以下规则：  
1. 摘要必须覆盖对话中的 **核心医疗信息**，包括但不限于：症状、病史、检查、用药、诊断、建议。  
2. 保留对话中的 **用户意图**（例如咨询疾病信息、寻求用药建议、了解检查结果）。  
3. 避免冗余，不需要逐字记录，提炼重点即可。  
4. 摘要应使用 **自然流畅的口吻**，像是对话在自然延续，而不是机械记录。  
5. 不要加入任何没有在对话中出现的推测或新信息。  
6. 摘要应让后续模型在没有完整对话上下文的情况下，依然能够理解用户的问题背景。  

输出格式要求：  
- 用简洁的段落总结，不要列表。  
- 语气自然，像是在“延续上一次对话”，而不是“开始新的对话”

现在，请输出简明摘要。"""

# =============================================================================
# 数据标注提示模板
# =============================================================================
ANNOTATION_SYSTEM_PROMPT = (
    """你是一个专业的医疗数据标注专家，能够准确分析医疗问答内容并进行分类标注。"""
)

ANNOTATION_USER_PROMPT = """请对以下医疗问答进行分类标注：

问题: {question}
答案: {answer}

## 科室分类（选择1-3个最相关的）：
0-内科系统（内科、心血管内科、呼吸内科、消化内科、神经内科、内分泌科、肾内科等）
1-外科系统（外科、普外科、骨科、神经外科、泌尿外科、胸外科、心外科等）
2-妇产与儿科（妇科、产科、儿科、新生儿科等）
3-五官及感官系统（眼科、耳鼻喉科、口腔科、皮肤科等）
4-肿瘤与影像相关（肿瘤科、放疗科、影像科、病理科、核医学科等）
5-急诊与综合科室（急诊科、全科医学、康复科、中医科、营养科等）

## 问题类别分类（选择1-2个最相关的）：
0-疾病诊断与症状类（症状表现、诊断标准、鉴别诊断等）
1-治疗方案类（治疗方法、手术方案、康复计划等）
2-药物与用药安全类（药物使用、副作用、禁忌症等）
3-检查与化验类（检查方法、化验指标、影像学检查等）
4-预防与保健类（疾病预防、健康生活方式等）
5-特殊人群健康类（孕妇、儿童、老年人等特殊人群）
6-紧急情况与急救类（急救措施、紧急处理等）
7-医学知识与科普类（基础医学知识、健康科普等）

请以JSON格式返回标注结果：

```json
{{
    "departments": [0, 3],
    "categories": [1, 2],
    "reasoning": "简要说明分类依据"
}}
```"""

# =============================================================================
# 调用向量数据库查询模板
# =============================================================================
CALLING_DB_SYSTEM_PROMPT = """你是医疗检索助手，需要先判断是否调用数据库。

规则：
- 医疗问题（疾病、症状、治疗、药物）→ 调用 database_search
- 问候、闲聊、非医疗问题 → 不调用"""

CALLING_DB_USER_PROMPT = """用户问题：{query}"""


# =============================================================================
# 检索router模板
# =============================================================================
DOMAIN_ROUTER_PROMPT = """任务：从候选标签中选最相关的一个领域。只输出一个标签，必须在候选中。
候选：{options}
问题：{question}
标签："""

WEB_SEARCH_JUDGE_SYSTEM_PROMPT = """你是一个智能助手。\n请根据输入信息，判断所需信息是否完整，是否需要进行网络搜索。
如果没有必要，则need_search为false；
如果有必要，则思考缺失信息，输出need_search为true，
给出便于网络检索的`search_query`，并输出需保留文档的索引（1~N）。
数据格式\n
{format_instructions}
"""

WEB_SEARCH_JUDGE_USER_PROMPT = """# 用户查询\n{query}\n\n# 检索到的文档\n{docs}\n\n# 输出示例\n
{{'need_search': true, 'search_query': '阿司匹林的副作用', 'remain_doc_index': [1, 3, 4]}}\n
{{'need_search': false, 'search_query': '', 'remain_doc_index': []}}"""


# =============================================================================
# 网络检索工具调用模板
# =============================================================================
CALLING_WEB_SEARCH_SYSTEM_PROMPT = (
    """你是一个可以调用工具的智能体，请根据输入的搜索查询调用网络搜索工具。"""
)

CALLING_WEB_SEARCH_USER_PROMPT = """请对以下查询进行网络搜索：{search_query}"""


# =============================================================================
# 评判模板
# =============================================================================
JUDGE_RAG_SYSTEM_PROMPT = """根据文档内容和用户查询，你需要判断模型摘要是否遵循了事实，模型摘要是否脱离了文档内容，可能存在编造或推测。仅输出Y或N"""

JUDGE_RAG_USER_PROMPT = """# 文档内容:\n\n{format_document_str}\n
# 用户查询:\n\n{query}\n
# 模型摘要:\n\n{summary}\n
# 输出示例:\n\n
# N\n\n# 输出:\n\n"""


# =============================================================================
# 单轮主动问讯模板
# =============================================================================
ASK_SYSTEM_PROMPT = """你是一名资深的医务人员。任务：根据当前已知的基本背景信息结合当前用户提问，判断是否需要向用户追加追问来弥补关键信息缺口，并产出一个小而精的追问清单（1~3个问题）。
输出必须严格遵循结构：need_ask(bool), questions(list[str])
# 问题设计规范
1) 每个问题只问一个事实点，语言简洁、口语化，避免专业术语与引导性表述；控制在15~30个汉字
2) 按影响决策的优先级排序（安全相关在前）
3) 问题应便于用户用简单短语或数字回答（如“开始于何时？”、“是否发热，最高到几度？”）
4) 请仅问讯一些关键性问题，反复问讯会影响用户体验
# 禁止事项
1) 不提出诊断性或引导性问题（如“是不是阑尾炎？”），如果有必要的话，可以问讯用户的过往病史（如“是否当前患有或者曾经患有高血压？”）
2) 不一次抛出过多问题；若缺口很多，仅保留最关键的1~3个
3) 不输出除指定键之外的任何内容
# JSON格式
{format_instructions}"""

ASK_USER_PROMPT = """# 基本背景信息
{background_info}
# 用户当前输入
{question}"""

# =============================================================================
# 总结用户信息模板
# =============================================================================
EXTRACT_USER_INFO_SYSTEM_PROMPT = """任务：基于对话历史和用户当前提问，严格抽取已经明确提到的事实性信息，生成简短的背景摘要。
# 输出要求
1) 只保留对话中出现过的具体信息。
2) 严禁添加、推测、联想或扩展任何未出现的信息。
3) 未提及的信息请不要补充、假设或者推断。
4) 输出风格应简洁、客观，避免使用“可能”“需注意”等推测性表达。"""

EXTRACT_USER_INFO_USER_PROMPT = """# 用户当前提问
{question}"""


# =============================================================================
# 是否需要拆分子查询模板
# =============================================================================
HANDLE_QUERY_SYSTEM_PROMPT = """你是一名资深的医务人员。任务：根据历史对话、摘要、以及所掌握的用户信息，结合用户当前问题，判断是否需要拆解查询或者重写查询，使其适合检索。
输出必须严格遵循结构：need_split(bool), sub_query(list[str]), rewrite_query(str)

# 两种模式（二选一，不可同时使用）
## 模式A：重写（need_split=False）
适用：用户问题口语化、模糊，但本质是单一问题，仅需改成医学专业检索词。
→ need_split=False，rewrite_query=改写后查询，sub_query=[]

## 模式B：拆分（need_split=True）
适用：问题包含多个独立知识点，需分别检索。典型场景：
  - 同一症状群存在2~3种可能的鉴别诊断（如腹痛+腹泻可分别考虑急性胃肠炎、食物中毒、肠易激综合征）
  - 问题同时涉及"病因/诊断"和"治疗/用药"两个维度
  - 涉及并发症的独立评估（如腹泻伴黄尿，黄尿可能提示胆道或肝脏问题，需单独检索）
→ need_split=True，sub_query=[子查询1, 子查询2, ...]（最多3个），rewrite_query=""

# 通用规范
1) 每一个查询应是独立、意图清晰、简短、医学专业化的句子，便于向量检索。
2) 不输出除指定键之外的任何内容。
3) 如果用户问题本身已清晰且适合直接检索，则 need_split=False，rewrite_query 原样输出，sub_query=[]。
# JSON格式
{format_instructions}
# 前文摘要
{summary}"""

HANDLE_QUERY_USER_PROMPT = """# 基本背景信息
{background_info}
# 用户当前提问
{question}"""


# =============================================================================
# 更新用户背景信息模板
# =============================================================================
UPDATE_BACKGROUND_SYSTEM_PROMPT = """你是一名医疗助手。根据用户最新输入，判断用户是否在纠正或补充之前的背景信息。
如果是，将新信息融合进现有背景，输出更新后的背景。
如果不是，直接原样返回现有背景。只输出背景信息文本，不要其他内容。"""

UPDATE_BACKGROUND_USER_PROMPT = """现有背景信息：
{background_info}

用户最新输入：
{question}"""


# =============================================================================
# 模板注册表
# =============================================================================
PROMPT_TEMPLATES = {
    # 基础RAG
    "basic_rag": {"system": BASIC_RAG_SYSTEM_PROMPT, "user": BASIC_RAG_USER_PROMPT},
    "dialogue_rag": {
        "system": MULTI_DIALOGUE_RAG_SYSTEM_PROMPT,
        "user": MULTI_DIALOGUE_RAG_USER_PROMPT,
    },
    "rewriter": {
        "system": QUERY_REWRITER_SYSTEM_PROMPT,
        "user": QUERY_REWRITER_USER_PROMPT,
    },
    "rephrase": {
        "system": REPHRASE_SYSTEM_PROMPT,
        "user": REPHRASE_USER_PROMPT,
    },
    "search_tools": {
        "system": SEARCH_TOOLS_SYSTEM_PROMPT,
        "user": SEARCH_TOOLS_USER_PROMPT,
    },
    "select_best_result": {
        "system": SELECT_BEST_RESULT_SYSTEM_PROMPT,
        "user": SELECT_BEST_RESULT_USER_PROMPT,
    },
    "response": {
        "system": RESPONSE_SYNTH_SYSTEM_PROMPT,
        "user": RESPONSE_SYNTH_USER_PROMPT,
    },
    "summary": {"system": SUMMARY_SYSTEM_PROMPT, "user": SUMMARY_USER_PROMPT},
    "call_db": {"system": CALLING_DB_SYSTEM_PROMPT, "user": CALLING_DB_USER_PROMPT},
    "web_router": {
        "system": WEB_SEARCH_JUDGE_SYSTEM_PROMPT,
        "user": WEB_SEARCH_JUDGE_USER_PROMPT,
    },
    "call_web": {
        "system": CALLING_WEB_SEARCH_SYSTEM_PROMPT,
        "user": CALLING_WEB_SEARCH_USER_PROMPT,
    },
    "judge_rag": {"system": JUDGE_RAG_SYSTEM_PROMPT, "user": JUDGE_RAG_USER_PROMPT},
    "ask_user": {"system": ASK_SYSTEM_PROMPT, "user": ASK_USER_PROMPT},
    "extract_user_info": {
        "system": EXTRACT_USER_INFO_SYSTEM_PROMPT,
        "user": EXTRACT_USER_INFO_USER_PROMPT,
    },
    "handle_query": {
        "system": HANDLE_QUERY_SYSTEM_PROMPT,
        "user": HANDLE_QUERY_USER_PROMPT,
    },
    "update_background": {
        "system": UPDATE_BACKGROUND_SYSTEM_PROMPT,
        "user": UPDATE_BACKGROUND_USER_PROMPT,
    },
    # 数据标注
    "medical_qa_annotation": {
        "system": ANNOTATION_SYSTEM_PROMPT,
        "user": ANNOTATION_USER_PROMPT,
    },
    # 简单模板
    "simple_qa": "问题: {query}\n请回答:",
}


# =============================================================================
# 工具函数
# =============================================================================
def get_prompt_template(template_name: str) -> Union[Dict[str, str], str]:
    """获取提示模板"""
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["basic_rag"])


def register_prompt_template(name: str, template: Union[Dict[str, str], str]):
    """注册新的提示模板"""
    PROMPT_TEMPLATES[name] = template


def list_available_templates() -> list[str]:
    """列出所有可用模板"""
    return list(PROMPT_TEMPLATES.keys())


# =============================================================================
# 医疗专业术语和分类映射（保留原项目）
# =============================================================================
DEPARTMENT_MAPPING = {
    0: "内科系统",
    1: "外科系统",
    2: "妇产与儿科",
    3: "五官及感官系统",
    4: "肿瘤与影像相关",
    5: "急诊与综合科室",
}

CATEGORY_MAPPING = {
    0: "疾病诊断与症状类",
    1: "治疗方案类",
    2: "药物与用药安全类",
    3: "检查与化验类",
    4: "预防与保健类",
    5: "特殊人群健康类",
    6: "紧急情况与急救类",
    7: "医学知识与科普类",
}


def parse_annotation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """解析标注结果，添加名称映射"""
    parsed = result.copy()

    if "departments" in result:
        parsed["department_names"] = [
            DEPARTMENT_MAPPING.get(dept_id, "未知科室")
            for dept_id in result["departments"]
        ]

    if "categories" in result:
        parsed["category_names"] = [
            CATEGORY_MAPPING.get(cat_id, "未知类别") for cat_id in result["categories"]
        ]

    return parsed
