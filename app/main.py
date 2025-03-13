# FastAPI应用的入口，包含路由定义和数据库连接配置
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from .database import SessionLocal
from fastapi.responses import StreamingResponse
import re
import redis
import json
import os
from . import tools
import uvicorn
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/agent_stream")
async def agent_stream(session_id: str, question: str):
    async def event_generator():
        print(question)
        # Stage 1: Request received and initializing
        yield f"data: Request received, initializing\n\n"
        await asyncio.sleep(1)  # Simulate delay

        # Stage 2: Retrieve context from Redis
        context = get_context_from_redis(session_id)

        # Stage 3: Append the user question to the context
        context.append({"role": "user", "content": question})
        yield f"data: Context loaded, user question recorded, identifying intent\n\n"
        await asyncio.sleep(0.1)

        # Stage 4: Determine question category
        category = await determine_question_category(question)
        if category == 1:
            yield "data: Recognized as tool request, processing\n\n"
        else:
            yield "data: Recognized as knowledge request, processing\n\n"
        # yield f"data: Question classification complete, category: {category}.\n\n"
        await asyncio.sleep(1)

        # Stage 5: Process the question based on the category
        if category == 1:
            # Note: If there are multiple steps in processing, yield progress updates after each step
            yield f"data: Planning and executing tool invocation process\n\n"
            answer = await handle_tool_request(question, context)
        else:
            yield f"data: Retrieving knowledge base and generating answer\n\n"
            answer = await handle_knowledge_request(question, context)

        yield f"data: Answer generation complete\n\n"
        await asyncio.sleep(0.5)

        # Final stage: Append the assistant's answer to the context and save it
        context.append({"role": "assistant", "content": answer})
        save_context_to_redis(session_id, context)

        def process_answer(answer):
            # Remove <think>...</think> blocks
            processed = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
            return processed.strip()

        final_answer = process_answer(answer)
        # For each line in the final answer, yield a line prefixed with "data:"
        for line in final_answer.splitlines():
            yield f"data:{line}\n"
        yield f"data:[DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 配置 Redis
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
REDIS_EXPIRATION_TIME = 1800  # 上下文过期时间，单位：秒（30分钟）
def get_context_from_redis(session_id: str) -> list:
    """
    从 Redis 中获取上下文。
    
    :param session_id: 会话 ID
    :return: 上下文列表
    """
    context = redis_client.get(session_id)
    if context:
        return json.loads(context)
    
    # 如果上下文不存在，初始化默认值
    return [{"role": "system", "content": "You are a helpful assistant specialized in neuroscience databases."}]

def save_context_to_redis(session_id: str, context: list):
    """
    将上下文保存到 Redis, 并设置过期时间。
    
    :param session_id: 会话 ID
    :param context: 上下文列表
    """
    redis_client.set(session_id, json.dumps(context), ex=REDIS_EXPIRATION_TIME)

async def determine_question_category(question: str) -> int:
    """
    Determines the category of the question using an LLM.
    Categories:
    1 - Tool request
    2 - Knowledge request
    """
    prompt = (
        "Strictly respond with 1 or 2 based on:\n\n"
        "Category 1: Requests requiring database operations (search, update, or process data).\n"
        "Examples:\n"
        "- Find image of cell id 25213\n"
        "- Auto trace neuron ID 12345\n"
        "- Show statistics for brain region\n\n"
        "Category 2: Requests requiring scientific explanations.\n"
        "Examples:\n"
        "- Explain hippocampal neurons\n"
        "- What is SWC file format?\n\n"
        f"Query: \"{question}\"\n\n"
        "Output format: <number>"
    )

    # Call the LLM with the prompt
    response = await call_large_language_model(prompt)
    print("原始响应: \n\n", response)

    # Extract the category number from the response
    try:
        print("try to recognize the category: ")

        # 优先匹配<number>格式
        bracket_matches = re.findall(r'<([12])>', response)
        if bracket_matches:
            print("检测到尖括号格式:", bracket_matches)
            return int(bracket_matches[-1])

        # 次选方案：匹配独立数字
        digit_matches = re.findall(r'\b[12]\b', response)
        if digit_matches:
            print("检测到独立数字:", digit_matches)
            return int(digit_matches[-1])

        else:
            # If the response is not a valid category number, default to category 2
            print("未检测到有效数字，返回默认值2")
            return 2
        
    except ValueError as e:
        # If the response cannot be converted to an integer, default to category 2
        print(f"分类处理异常: {str(e)}")
        return 2

## Categories 1 - Tool Request
def generate_pipeline_dir() -> str:
    """
    生成一个以时间戳区分的pipeline目录，如:
      ./results/pipeline_20250101_2026
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    base_dir = './results'
    pipeline_subdir = f'pipeline_{timestamp}'
    pipeline_dir = os.path.join(base_dir, pipeline_subdir)
    os.makedirs(pipeline_dir, exist_ok=True)
    return pipeline_dir

async def handle_tool_request(question: str, context: list) -> str:
    """
    让 LLM 先生成一个【工具调用计划】，再根据计划依次调用并汇总结果。
    """
    # 1) 给大模型列出可用工具名称、功能、以及所需参数
    tools_description = """
You have these tools available:

1) search_by_id(cell_id: str, db: Session) -> str
   - Searches the database for the **image file path** associated with a given cell_id.
   - Use this tool when the request is specifically about retrieving the image (e.g., "show the image for cell ID 56692").
   - Returns a single image path (string) or an error message.

2) search_by_criteria(field: str, value: str, db: Session) -> List[str]
   - Searches the database for multiple image paths matching a given field and value.
   - For example, field="brain_region" with value="MFG", or field="dye" with value="Alexa488".
   - Returns a list of image paths or an error message.

3) preprocess_image(image_path_or_paths: Union[str, List[str]]) -> Union[str, List[str]]
  - Preprocesses one or multiple images using the Vaa3D imPreProcess plugin.
  - Returns the path(s) of the preprocessed image file(s) or an error message.

4) auto_tracing(image_path_or_paths: Union[str, List[str]]) -> Union[str, List[str]]
   - Automatically traces one or multiple images using Vaa3D, generating SWC file(s).
   - Returns the SWC path(s) or an error message.

5) postprocess_results(swc_path_or_paths: Union[str, List[str]]) -> Union[str, List[str]]
   - Performs resampling or other post-processing on one or multiple SWC file(s) using Vaa3D.
   - Returns the resampled SWC path(s) or an error message.

6) extract_features(swc_path_or_paths: Union[str, List[str]]) -> Union[str, List[str]]
   - Extracts features from one or multiple SWC file(s) using Vaa3D, generating CSV file(s).
   - Returns the path(s) to the CSV file(s) or an error message.

7) insert_data_from_csv(csv_path: str, db: Session) -> str
   - Reads a local CSV file and inserts each row into the database table.
   - Returns a success or error message.

8) delete_data(field: str, value: str, db: Session) -> str
   - Deletes records from the table where (field == value).
   - Example: delete_data(field="cell_id", value="12345") or delete_data(field="brain_region", value="MFG").

9) update_data(filter_field: str, filter_value: str, target_field: str, new_value: str, db: Session) -> str
   - Updates 'target_field' to 'new_value' for all rows where (filter_field == filter_value).
   - Example: update_data("cell_id", "12345", "dye_name", "Cascade Blue").

10) display_images(image_path_or_paths: Union[str, List[str]]) -> Union[str, List[str]]
    - Reads one or multiple .tif image files, converts them to PNG format in memory, and encodes them to Base64.
    - Returns Markdown formatted image tag(s) (e.g., ![Image](data:image/png;base64,...)) for direct visualization in the chat interface.

11) retrieve_data(filter_field: str, filter_value: str, db: Session) -> Union[List[models.HumanSingleCellTrackingTable], str]
    - Retrieves complete records from the HumanSingleCellTrackingTable that match the specified filter field and value.
    - Use this tool when the request is to "retrieve the data record", "find the data information", or similar, where details beyond the image path are needed.
    - Returns a list of matching records if found, or an appropriate error/informational message.
"""

    plan_prompt = f"""
You are a data-processing assistant. The user says:
"{question}"

{tools_description}

### Output Requirements:
- You MUST provide a valid JSON object with a top-level key "steps".
- "steps" is a list of objects, each with "tool_name" and "arguments".
- No extra keys at the top level.
- Do NOT wrap your output in markdown or add extra commentary.

### Examples of possible tasks and the correct JSON output:

EXAMPLE A:
User wants: "Find the image for cell 00123"
We want steps to do: search_by_id only

JSON output:
{{
  "steps": [
    {{
      "tool_name": "search_by_id",
      "arguments": {{
        "cell_id": "00123"
      }}
    }}
  ]
}}

EXAMPLE B:
User wants: "Auto trace the data for brain region MFG"
We want steps to do:
1) search_by_criteria(field=brain_region, value=MFG)
2) preprocess_image(image_path_or_paths=<the path from step1>)
3) auto_tracing(image_path_or_paths=<the paths from step2>)

JSON output:
{{
  "steps": [
    {{
      "tool_name": "search_by_criteria",
      "arguments": {{
        "field": "brain_region",
        "value": "MFG"
      }}
    }},
    {{
      "tool_name": "preprocess_image",
      "arguments": {{
        "image_path_or_paths": "PLACEHOLDER_FROM_STEP1",
      }}
    }},
    {{
      "tool_name": "auto_tracing",
      "arguments": {{
        "image_path_or_paths": "PLACEHOLDER_FROM_STEP2"
      }}
    }}
  ]
}}

EXAMPLE C:
User wants: "Compute the morphology features of cell 25213."
We want steps to do:
1) search_by_id(cell_id=25213)
2) preprocess_image(image_path_or_paths=<the path from step1>)
3) auto_tracing(image_path_or_paths=<the path from step2>)
4) extract_features(swc_path_or_paths=<the path from step3>)

JSON output:
{{
  "steps": [
    {{
      "tool_name": "search_by_id",
      "arguments": {{
        "cell_id": "25213"
      }}
    }},
    {{
      "tool_name": "preprocess_image",
      "arguments": {{
        "image_path_or_paths": "PLACEHOLDER_FROM_STEP1",
      }}
    }},
    {{
      "tool_name": "auto_tracing",
      "arguments": {{
        "image_path_or_paths": "PLACEHOLDER_FROM_STEP2"
      }}
    }},
    {{
      "tool_name": "extract_features",
      "arguments": {{
        "swc_path_or_paths": "PLACEHOLDER_FROM_STEP3"
      }}
    }}
  ]
}}

EXAMPLE D
User wants: "Display all the images of region MFG."
We want steps to do:
1) search_by_criteria(field=brain_region, value=MFG)
2) display_images(image_path_or_paths=<the paths from step1>)

JSON output:
{{
  "steps": [
    {{
      "tool_name": "search_by_criteria",
      "arguments": {{
        "field": "brain_region",
        "value": "MFG"
      }}
    }},
    {{
      "tool_name": "display_images",
      "arguments": {{
        "image_path_or_paths": "PLACEHOLDER_FROM_STEP1",
      }}
    }}
  ]
}}

EXAMPLE E:
User wants: "Insert data from local CSV 'C:/data/new_cells.csv' into the database"

JSON output:
{{
  "steps": [
    {{
      "tool_name": "insert_data_from_csv",
      "arguments": {{
        "csv_path": "C:/data/new_cells.csv"
      }}
    }}
  ]
}}

EXAMPLE F:
User wants: "Update dye name to 'Cascade Blue' for cell 12345"

JSON output:
{{
  "steps": [
    {{
      "tool_name": "update_data",
      "arguments": {{
        "filter_field": "cell_id",
        "filter_value": "12345",
        "target_field": "dye_name",
        "new_value": "Cascade Blue"
      }}
    }}
  ]
}}

END OF EXAMPLES

Now, based on the user's request: "{question}"

Please create the plan in exactly the same JSON structure shown above.
Only tools that appear in the tool description are allowed.
Output only the JSON (no extra commentary).
"""

    # 2) 让大模型输出 JSON 格式的调用计划
    plan_response = await call_large_language_model_with_context(plan_prompt, context)
    matches = re.search(r"```json\s*(\{[\s\S]*\})\s*```", plan_response)
    if matches:
        plan_response = matches.group(1)
        print("plan_response: ", plan_response)

    # 3) 从 plan_response 中解析 JSON
    try:
        print("try to read the JSON from plan_response...")
        plan_dict = json.loads(plan_response)
        print("plan_dict:", plan_dict)
        steps = plan_dict.get("steps", [])
        print("steps:", steps)
    except json.JSONDecodeError:
        return (
            "I could not parse a tool plan from your request. "
            "Please make sure the request is clear, or try again."
        )

    # 4) 依次执行每一步工具
    pipeline_dir = generate_pipeline_dir()  # 生成 pipeline_dir
    print("DEBUG pipeline_dir:", pipeline_dir)

    db = SessionLocal()  # 假设在同模块或全局已定义
    result_text = []

    # 用于存储每一步的输出结果，索引 i 对应步骤 i
    step_outputs = [None] * len(steps)
    print("step_outputs: ", step_outputs)

    # 需要使用 pipeline_dir 的工具函数列表
    pipeline_based_tools = [
        "preprocess_image", 
        "auto_tracing", 
        "postprocess_results",
        "extract_features"
    ]

    for i, step in enumerate(steps):
        tool_name = step.get("tool_name")
        args = step.get("arguments", {})

        # 数据库相关工具，加上 db
        if tool_name in ["search_by_id", "search_by_criteria", 
                         "insert_data_from_csv", "retrieve_data", "delete_data", "update_data"]:
            args["db"] = db

        # 对需要 pipeline_dir 的工具自动注入
        if tool_name in pipeline_based_tools:
            args["pipeline_dir"] = pipeline_dir

        # 检查 TOOLS_REGISTRY
        if tool_name not in tools.TOOLS_REGISTRY:
            result_text.append(f"Step {i+1}: Unknown tool '{tool_name}'. Skipped.")
            continue

        # 占位符替换逻辑 
        # 先扫描 arguments，如果出现 PLACEHOLDER_FROM_STEPx 等，就用前一步骤的结果替换
        for arg_key, arg_val in args.items():
            if not isinstance(arg_val, str):
                continue  # 只有字符串才需要替换

            # 如果大模型写的是 "PLACEHOLDER_FROM_STEP1"、"PLACEHOLDER_FROM_STEP2"...  
            match = re.search(r"PLACEHOLDER_FROM_STEP(\d+)", arg_val)
            if match:
                step_number = int(match.group(1)) - 1  # 大模型可能从1开始计数，Python列表从0开始
                if 0 <= step_number < i:  # 确保这个step_number已执行
                    args[arg_key] = step_outputs[step_number]

            # 如果大模型写的是 "PLACEHOLDER_FROM_SEARCH_PATH"
            # 可再单独写一段逻辑，比如：
            elif arg_val == "PLACEHOLDER_FROM_SEARCH_PATH":
                # 找到最近一次 search_by_id 的结果
                search_step_idx = None
                for j in range(i-1, -1, -1):
                    if steps[j].get("tool_name") == "search_by_id":
                        search_step_idx = j
                        break
                if search_step_idx is not None:
                    args[arg_key] = step_outputs[search_step_idx]

        # 检查工具是否存在
        if tool_name not in tools.TOOLS_REGISTRY:
            result_text.append(f"Step {i+1}: Unknown tool '{tool_name}'. Skipped.")
            continue

        tool_func = tools.TOOLS_REGISTRY[tool_name]

        try:
            output = tool_func(**args)
            step_outputs[i] = output  # 记录本步的执行结果
            result_text.append(f"Step {i+1}: Called {tool_name} => {output}\n\n")
        except Exception as e:
            step_outputs[i] = None
            result_text.append(f"Step {i+1}: Error calling {tool_name}: {str(e)}\n\n")

    db.close()

    final_report = "\n".join(result_text)
    return f"Tool invocation plan executed.\n\n{final_report}"

## Categories 2 - Professional or general questions
async def handle_knowledge_request(question: str, context: list) -> str:
    """
    处理专业或知识性问题：
    1. 先从知识库中检索相关信息。
    2. 将检索结果与用户问题一起构造 prompt。
    3. 调用大模型生成答案，并返回答案。
    """
    # Step 1: 检索知识库 (假设有一个函数 search_knowledge_base)
    retrieved_docs = search_knowledge_base(question)  # 返回列表，比如 ['文档1摘要', '文档2摘要']
    
    # Step 2: 构造增强 prompt，将检索结果嵌入 prompt 中
    knowledge_context = "\n\n".join(retrieved_docs)
    prompt = (
        "You are an expert in neuroscience with access to a knowledge base. "
        "Based on the following retrieved information and the user's question, "
        "provide a professional answer.\n\n"
        "Retrieved Information:\n"
        f"{knowledge_context}\n\n"
        "User Question:\n"
        f"{question}\n\n"
        "Answer:"
    )
    
    # Step 3: 调用大模型生成回答
    answer = await call_large_language_model_with_context(prompt, context)
    
    return answer


from sentence_transformers import SentenceTransformer, util
# 初始化模型（建议在程序启动时加载一次）
model = SentenceTransformer('./all-MiniLM-L6-v2')

def load_knowledge_docs_with_embeddings() -> List[dict]:
    try:
        with open("knowledge_docs_with_emb.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        return docs
    except Exception as e:
        print(f"Error loading knowledge_docs_with_emb.json: {e}")
        return []
    
def search_knowledge_base(question: str, top_k: int = 2) -> List[str]:
    """
    根据用户问题，从外部知识库中利用向量搜索检索相关文档摘要。
    使用预训练的 SentenceTransformer 模型计算用户问题和知识库文档的嵌入向量，
    然后返回最相关的 top_k 个文档内容。
    
    如果没有检索到足够相关的文档，则返回空列表。
    """
    docs = load_knowledge_docs_with_embeddings()  # 调用函数，获取文档列表
    query_embedding = model.encode(question, convert_to_tensor=True)
    
    # 计算余弦相似度
    cos_scores = [util.cos_sim(query_embedding, doc["embedding"]).item() for doc in docs]
    
    # 找到相似度最高的 top_k 个索引
    top_results = sorted(range(len(cos_scores)), key=lambda i: cos_scores[i], reverse=True)[:top_k]
    
    threshold = 0.4
    results = []
    for idx in top_results:
        if cos_scores[idx] >= threshold:
            results.append(docs[idx]["content"])
    
    print("results of search knowledge base:", results)
    
    return results


## Call LLMs
from openai import AsyncOpenAI 

async def call_large_language_model(prompt: str) -> str:
    """
    调用本地Ollama部署的DeepSeek模型进行分类任务
    
    :param prompt: 用户输入文本
    :return: 模型生成的分类结果
    """
    client = AsyncOpenAI(
        base_url='http://10.192.20.27:8888/v1',  # Ollama的OpenAI兼容API地址[5,7](@ref)
        api_key='ollama'  # 本地部署无需真实key[5](@ref)
    )

    try:
        response = await client.chat.completions.create(
            model="deepseek-r1:14b",  # 需与本地安装模型名称一致[1,3](@ref)
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,  # 限制输出长度适合分类任务[8](@ref)
            temperature=0.1,  # 确保输出确定性[8](@ref)
            # stream=False  # 如需流式响应可设为True
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error: {str(e)}"

async def call_large_language_model_with_context(prompt: str, context: list) -> str:
    """
    调用本地Ollama部署的DeepSeek模型进行多轮对话
    
    :param prompt: 用户当前提问
    :param context: 对话上下文（格式示例：[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助您？"}]）
    :return: 模型生成的回复
    """
    # 初始化OpenAI客户端（需安装openai>=1.0）
    client = AsyncOpenAI(
        base_url='http://10.192.20.27:8888/v1',  # Ollama的OpenAI兼容API地址
        api_key='ollama'  # 本地部署无需真实key，但参数必须存在
    )

    # 构建消息列表（保留完整对话上下文）
    messages = context + [{"role": "user", "content": prompt}]

    try:
        response = await client.chat.completions.create(
            model="deepseek-r1:14b",  # 模型名称 需与Ollama安装的模型一致
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            stream=False  # 如需流式响应可设为True
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"调用失败：{str(e)}"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
