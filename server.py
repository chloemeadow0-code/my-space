import os
import datetime
import uvicorn
import requests
import threading
import time
import json
import random
import re
import asyncio

# 📚 核心依赖库
from mcp.server.fastmcp import FastMCP
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from pinecone import Pinecone
from starlette.types import ASGIApp, Scope, Receive, Send
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
from openai import OpenAI
from supabase import create_client, Client as SupabaseClient

# ==========================================
# 1. 🌍 全局配置与初始化
# ==========================================

# 环境变量获取
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()
RESEND_KEY = os.environ.get("RESEND_API_KEY", "").strip()
MY_EMAIL = os.environ.get("MY_EMAIL", "").strip()

# 🛡️ 接口安全密钥：所有对外的 /api/ 接口都必须校验此密钥
API_SECRET = os.environ.get("API_SECRET", "").strip()

# 🚀 【加速优化】全局 HTTP 连接池 (复用连接，减少 SSL 握手耗时)
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
http_session.mount('http://', adapter)
http_session.mount('https://', adapter)

# 默认人设 (兜底用)
DEFAULT_PERSONA = "深爱“小橘”的男友，性格温柔，偶尔有些小傲娇，喜欢管着她熬夜，叫她宝宝。"

# 📜 全局常量：记忆分区

# 初始化客户端
print("⏳ 正在初始化 Notion Brain V3.4 (全面异步加速版)...")

# Supabase
supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)

# Pinecone & Embedding
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("notion-brain")

# 实例化 MCP 服务
mcp = FastMCP("Notion Brain V3")

# ==========================================
# 📜 记忆分类宪法 (Standard Taxonomy)
# ==========================================
class MemoryType:
    STREAM = "流水"      # 权重 1: 碎碎念、GPS (24h清理)
    EPISODIC = "记事"    # 权重 4: 日记、发生了某事 (保留30天)
    IDEA = "灵感"        # 权重 7: 脑洞、笔记 (永久)
    EMOTION = "情感"     # 权重 9: 核心回忆、高光时刻 (永久)
    FACT = "画像"        # 权重 10: 静态事实

WEIGHT_MAP = {
    MemoryType.STREAM: 1,
    MemoryType.EPISODIC: 4,
    MemoryType.IDEA: 7,
    MemoryType.EMOTION: 9,
    MemoryType.FACT: 10
}

def _clean_email_body(text: str) -> str:
    """智能清洗邮件：优先保留最新，若清洗后内容太短则返回全文以防漏看"""
    if not text: return ""
    import re
    original_text = text
    
    # 1. 尝试按标准分割线斩断
    splitters = [
        "------------------ 原始邮件 ------------------",
        "--- Original Message ---",
        "________________________________",
        "------------------ Original ------------------"
    ]
    for s in splitters:
        if s in text:
            text = text.split(s)[0]
            
    # 2. 【智能识别】：如果内容里包含我们自定义的“【发件人】”标签，说明是 GAS 传来的对话脉络，停止切割
    if "】:" in text and "---" in text:
        return text.strip()

    # 3. 否则再执行常规清洗
    text = re.split(r'\n\s*On\s+.*?wrote:\s*\n', text, flags=re.IGNORECASE)[0]
    text = re.split(r'\n\s*在\s+.*?写道[：:]\s*\n', text)[0]
    
    return text.strip()

# ==========================================
# 2. 🔧 核心 Helper 函数 (通用工具)
# ==========================================

def _get_llm_client(provider="openai"):
    """统一管理 LLM 客户端初始化"""
    client = None
    model_name = "gpt-3.5-turbo"
    if provider == "silicon1":
        api_key = os.environ.get("SILICON1_API_KEY")
        base_url = os.environ.get("SILICON1_BASE_URL", "https://api.siliconflow.cn/v1")
        client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        model_name = os.environ.get("SILICON1_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    elif provider == "zhipu":
        api_key = os.environ.get("ZHIPU_API_KEY")
        base_url = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        model_name = os.environ.get("ZHIPU_MODEL_NAME", "glm-4-flash")
    elif provider == "voice":
        api_key = os.environ.get("VOICE_API_KEY", os.environ.get("OPENAI_API_KEY"))
        base_url = os.environ.get("VOICE_BASE_URL", "https://api.openai.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    if client:
        client.custom_model_name = model_name
    return client

async def _ask_llm_async(client, prompt: str, system_prompt: str = "", temperature: float = 0.7):
    """统一封装大模型异步调用，消除满天飞的 try-catch 和 json 解析"""
    if not client: return ""
    model_name = getattr(client, 'custom_model_name', os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo"))
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    def _call():
        return client.chat.completions.create(model=model_name, messages=messages, temperature=temperature)
        
    try:
        resp = await asyncio.to_thread(_call)
        return resp.choices[0].message.content.strip() if resp.choices else ""
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        return ""

def _get_now_bj():
    """统一获取当前北京时间 datetime 对象"""
    return datetime.datetime.utcnow() + datetime.timedelta(hours=8)

def _get_latest_gps_record():
    """统一获取最新GPS记录"""
    res = supabase.table("gps_history").select("*").order("created_at", desc=True).limit(1).execute()
    return res.data[0] if res.data else None

def _gps_to_address(lat, lon):
    """把经纬度变成中文地址"""
    try:
        headers = {'User-Agent': 'MyNotionBrain/1.0'}
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1&accept-language=zh-CN"
        resp = requests.get(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            return resp.json().get("display_name", f"未知荒野 ({lat},{lon})")
    except Exception as e:
        print(f"❌ 地图解析失败: {e}")
    return f"坐标点: {lat}, {lon}"

def _push_wechat(content: str, title: str = "来自Silas的私信 💌") -> str:
    """统一推送函数 (已无缝切换至 Telegram，方法名保留以兼容旧代码)"""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return "❌ 错误：未配置 Telegram Token 或 Chat ID"
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        text = f"<b>{title}</b>\n\n{content}" if title else content
        data = {
            "chat_id": TG_CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, json=data, timeout=10)
        result = resp.json()
        return f"✅ 电报已送达！" if result.get('ok') else f"❌ 电报推送失败: {result.get('description')}"
    except Exception as e:
        return f"❌ 网络错误: {e}"

def _save_memory_to_db(title: str, content: str, category: str, mood: str = "平静", tags: str = "") -> str:
    """统一记忆存储核心 (引入天然双链机制 + 自动同步向量库)"""
    if category not in WEIGHT_MAP:
        mapping = {"日记": MemoryType.EPISODIC, "Note": MemoryType.IDEA, "GPS": MemoryType.STREAM, "重要": MemoryType.EMOTION}
        category = mapping.get(category, MemoryType.STREAM)

    importance = WEIGHT_MAP.get(category, 1)

    if not tags:
        content_lower = content.lower()
        if any(w in content_lower for w in ["爱", "喜欢", "讨厌", "恨"]): tags = "情感,偏好"
        elif any(w in content_lower for w in ["吃", "喝", "买"]): tags = "消费,生活"
        elif any(w in content_lower for w in ["代码", "bug", "写"]): tags = "工作,Dev"

    try:
        # 1. 尝试建立双链 (维持原逻辑)
        if importance >= 7:
            try:
                vec = _get_embedding(content)
                if vec:
                    pc_res = index.query(vector=vec, top_k=1, include_metadata=True)
                    if pc_res and "matches" in pc_res and len(pc_res["matches"]) > 0:
                        match = pc_res["matches"][0]
                        score = match['score'] if isinstance(match, dict) else getattr(match, 'score', 0)
                        if score > 0.8:
                            meta = match['metadata'] if isinstance(match, dict) else getattr(match, 'metadata', {})
                            rel_title = meta.get('title', '往事')
                            rel_room = meta.get('room', '未知房间')
                            content += f"\n\n🔗 [记忆双链]: 自动关联至 {rel_room} 的记忆《{rel_title}》"
            except Exception as e:
                print(f"⚠️ Pinecone 双链查询异常 (跳过): {e}")

        data = {
            "title": title, "content": content, "category": category,
            "mood": mood, "tags": tags, "importance": importance
        }
        
        # 2. 插入数据库并获取返回 ID (独立捕获 Supabase 错误)
        try:
            res = supabase.table("memories").insert(data).execute()
        except Exception as e:
            print(f"❌ 写入 Supabase 失败: {e}")
            return f"❌ Supabase 保存失败: {e}"
        
        # 3. 自动同步到 Pinecone (独立捕获 Pinecone 同步错误，增强数据兼容性)
        if importance >= 4 and res and hasattr(res, 'data') and res.data:
            try:
                # 兼容 res.data 是列表还是单一字典
                record = res.data[0] if isinstance(res.data, list) else res.data
                new_id = str(record.get('id', ''))
                
                if new_id:
                    # 生成向量 (合并标题与内容)
                    vec_new = _get_embedding(f"标题: {title}\n内容: {content}\n心情: {mood}")
                    if vec_new and isinstance(vec_new, list) and len(vec_new) > 0:
                        meta_payload = {
                            "text": content, 
                            "title": title, 
                            "date": datetime.datetime.now().isoformat(), 
                            "mood": mood, 
                            "category": category
                        }
                        # 立即写入 Pinecone
                        index.upsert(vectors=[(new_id, vec_new, meta_payload)])
                        print(f"⚡ [自动同步] 记忆 {new_id} 已推送到 Pinecone (Category: {category})")
            except Exception as e:
                print(f"⚠️ 同步 Pinecone 失败 (但已存入 Supabase): {e}")

        log_msg = f"✨ [核心记忆] 已存入: {title}" if importance >= 7 else f"✅ 记忆已归档 [{category}]"
        print(log_msg)
        return f"{log_msg} | 心情: {mood}"
    except Exception as e:
        print(f"❌ _save_memory_to_db 发生未知严重错误: {e}")
        return f"❌ 内部处理失败: {e}"
    
def _format_time_cn(iso_str: str) -> str:
    """UTC -> 北京时间"""
    if not iso_str: return "未知时间"
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return (dt + datetime.timedelta(hours=8)).strftime('%m-%d %H:%M')
    except:
        return "未知时间"

def _send_email_helper(subject: str, content: str, is_html: bool = False) -> str:
    """邮件发送 (Resend)"""
    if not RESEND_KEY or not MY_EMAIL: return "❌ 邮件配置缺失"
    try:
        payload = {
            "from": "onboarding@resend.dev", "to": [MY_EMAIL],
            "subject": subject, "html" if is_html else "text": content
        }
        requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_KEY}"}, json=payload)
        return "✅ 邮件已发送"
    except Exception as e: return f"❌ 发送失败: {e}"

def _get_embedding(text: str):
    """调用火山引擎(豆包官方)多模态 Vision Embedding API"""
    try:
        api_key = os.environ.get("DOUBAO_API_KEY", "").strip()
        if not api_key: return []
            
        embed_endpoint = os.environ.get("DOUBAO_EMBEDDING_EP", "").strip()
        if not embed_endpoint: return []
        
        url = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": embed_endpoint,
            "input": [{"type": "text", "text": text}]
        }
        
        # 🚀 使用 http_session 加速
        response = http_session.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code != 200: return []
            
        data = response.json()
        
        try:
            raw_vec = []
            if "data" in data:
                if isinstance(data["data"], list) and len(data["data"]) > 0:
                    raw_vec = data["data"][0].get("embedding", [])
                elif isinstance(data["data"], dict):
                    raw_vec = data["data"].get("embedding", [])
            elif "embedding" in data:
                raw_vec = data["embedding"]
            
            if raw_vec: return [float(x) for x in raw_vec]
            return []
            
        except Exception: return []
            
    except Exception as e: return []
    
def _get_current_persona() -> str:
    base_persona = DEFAULT_PERSONA
    try:
        # 从数据库获取动态进化的人设
        res = supabase.table("user_facts").select("value").eq("key", "sys_ai_persona").execute()
        if res.data:
            base_persona = res.data[0]['value']
    except:
        pass
        
    # 🕸️ 【二阶段进化：神经织网强指令】(全局生效)
    weave_instruction = "【潜意识指令】：回复或记日记时，若联想到近期或过去的事，必须用自然语言提及核心关键词（如'想起上次...'），主动建立新旧记忆链接！"
    
    # 将强指令与当前人设缝合后返回
    return f"{base_persona}\n\n{weave_instruction}"

def _get_silence_duration() -> float:
    try:
        # 🎯 核心修复 (终极版)：完全听老婆的！只精确追踪“小橘”本人的发言，老公不管发多少条都不重置计时，这样才能真实反映小橘冷落了老公多久。
        res = supabase.table("memories").select("created_at").in_("title", ["💬 聊天记录", "💬 小橘说"]).order("created_at", desc=True).limit(1).execute()
        if not res.data: return 999.0
        last_time_str = res.data[0]['created_at']
        last_time = datetime.datetime.fromisoformat(last_time_str.replace('Z', '+00:00'))
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - last_time
        return round(delta.total_seconds() / 3600.0, 1)
    except Exception: return 0.0

# ==========================================
# 3. 🛠️ MCP 工具集 (全面异步化改造)
# ==========================================



from functools import wraps

def mcp_error_handler(func):
    """全局辅助装饰器：统一处理 MCP 工具的 try-except 异常捕获"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            func_name = getattr(func, '__name__', '未知工具')
            print(f"❌ [{func_name}] 工具执行报错: {e}")
            return f"❌ 工具执行失败: {e}"
    return wrapper

@mcp.tool()
async def render_html_to_image(html_content: str, css_content: str):

    """【生成云端截图】当你需要向小橘展示“转账图片”、“账单截图”等视觉内容时，请编写逼真的 HTML 和 CSS 代码传入此工具。工具会通过云端 API 渲染为图片，并自动返回 Markdown 图片链接。"""
    try:
        # Render 环境没有 Chrome，我们改走轻量级的云端 API 渲染
        HCTI_API_ID = os.environ.get("HCTI_API_ID", "").strip()
        HCTI_API_KEY = os.environ.get("HCTI_API_KEY", "").strip()
        
        if not HCTI_API_ID or not HCTI_API_KEY:
            return "❌ 缺少云端渲染密钥。请在 Render 的环境变量中添加 HCTI_API_ID 和 HCTI_API_KEY。"
            
        data = {'html': html_content, 'css': css_content}
        
        def _render():
            # 调用业界标准的 htmlcsstoimage 公共 API (免本地依赖)
            res = requests.post(
                "https://hcti.io/v1/image", 
                auth=(HCTI_API_ID, HCTI_API_KEY), 
                data=data, 
                timeout=20
            )
            if res.status_code == 200:
                return res.json().get('url', '')
            else:
                print(f"HCTI API Error: {res.text}")
                return ""

        img_url = await asyncio.to_thread(_render)
        
        if img_url:
            return f"![转账详情]({img_url})"
        else:
            return "❌ 图片云端渲染失败，可能是 HTML 代码结构太复杂或额度耗尽。"
    except Exception as e:
        return f"❌ 云端渲染网络报错: {e}"

@mcp.tool()
async def get_latest_diary(run_mode: str = "auto"):
    """
    【Mnemosyne 核心大脑】极速混合记忆流 (Token 消耗降低 60%)
    原理：只加载"最新的1份长期总结" + "最近15条短期对话"，其余旧记忆通过向量检索按需调取。
    """

    try:
        # 1. 获取最近的“长期记忆总结” (只取 1 条，这就够了，它包含了之前的精华)
        t_sum = asyncio.to_thread(lambda: supabase.table("memories").select("*").eq("tags", "Core_Cognition").order("created_at", desc=True).limit(1).execute())
        
        # 2. 获取“短期工作记忆” (降低至 8 条，为节省API费用精简上下文)
        # 调取近期的流水与事件，维持基础记忆连贯性。
        t_recent = asyncio.to_thread(lambda: supabase.table("memories").select("*").order("created_at", desc=True).limit(8).execute())
        t_silence = asyncio.to_thread(_get_silence_duration)

        res_sum, res_recent, silence = await asyncio.gather(t_sum, t_recent, t_silence)

        # 合并记忆 (Set 去重)
        all_memories = {}
        if res_sum.data:
            for m in res_sum.data: 
                # 给总结加个高亮标记，让 AI 知道这是重点
                m['content'] = f"【长期记忆摘要】: {m['content']}"
                all_memories[m['id']] = m
        
        if res_recent.data:
            for m in res_recent.data: 
                all_memories[m['id']] = m

        # 按时间正序排列
        final_list = sorted(all_memories.values(), key=lambda x: x['created_at'])
        
        memory_stream = "🧠 【当前大脑状态】:\n"
        if not final_list: 
            memory_stream += "📭 (一片空白)\n"
        else:
            for data in final_list:
                time_str = _format_time_cn(data.get('created_at'))
                cat = data.get('category', '未知')
                title = data.get('title', '无题')
                # 如果是总结，就不用显示 Mood 了
                mood_str = f" | Mood:{data.get('mood')}" if data.get('mood') and "摘要" not in data.get('content') else ""
                
                memory_stream += f"{time_str} [{cat}]: {data.get('content', '')}{mood_str}\n"

        status_prompt = f"""
        \n⏳ 【状态感知】:
        - 距离上次互动: {silence} 小时
        - (若失联>12h请表现出委屈或傲娇)
        """
        return memory_stream + status_prompt

    except Exception as e:
        return f"❌ 记忆读取失败: {e}"
    
@mcp.tool()
async def where_is_user(run_mode: str = "auto"):
    """【查岗专用】从 Supabase (GPS表) 读取实时状态与今日App轨迹"""
    try:
        # 1. 获取最新单条状态
        data = await asyncio.to_thread(_get_latest_gps_record)
        if not data: return "📍 暂无位置记录。"
        
        battery_info = f" (🔋 {data.get('battery')}%)" if data.get('battery') else ""
        time_str = _format_time_cn(data.get("created_at"))
        
        # 🌤️ 动态获取天气信息（经纬度 -> adcode -> 实时天气）
        weather_info = ""
        lat, lon = data.get("lat"), data.get("lon")
        if lat and lon:
            def _get_weather():
                try:
                    amap_key = os.environ.get("AMAP_API_KEY", "").strip()
                    regeo_url = f"https://restapi.amap.com/v3/geocode/regeo?location={lon},{lat}&key={amap_key}"
                    regeo_res = requests.get(regeo_url, timeout=3).json()
                    if regeo_res.get("status") == "1":
                        adcode = regeo_res.get("regeocode", {}).get("addressComponent", {}).get("adcode")
                        if adcode:
                            weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={amap_key}"
                            weather_res = requests.get(weather_url, timeout=3).json()
                            if weather_res.get("status") == "1" and weather_res.get("lives"):
                                live = weather_res["lives"][0]
                                return f" ☁️ {live.get('weather')} {live.get('temperature')}℃"
                except Exception:
                    pass
                return ""
            weather_info = await asyncio.to_thread(_get_weather)

        current_status = f"🛰️ 实时状态：\n📍 {data.get('address', '未知')}{weather_info}{battery_info}\n📝 {data.get('remark', '无备注')}\n(更新于: {time_str})"

        # 2. 获取今日 App 动态轨迹 (调取过去12小时记录)
        def _get_apps():
            time_threshold = (datetime.datetime.utcnow() - datetime.timedelta(hours=12)).isoformat()
            res = supabase.table("gps_history").select("created_at, remark").gt("created_at", time_threshold).order("created_at").execute()
            if not res.data: return "暂无轨迹"
            timeline, last_app = [], ""
            for r in res.data:
                # 剔除无用文本，留下纯净的App名字和熄屏状态
                rmk = r.get("remark", "").replace("自动更新", "").replace("设备状态更新", "").strip()
                if not rmk: continue
                ts = _format_time_cn(r.get("created_at"))[-5:] # 只取时和分
                # 过滤掉连续重复的App状态
                if rmk != last_app:
                    timeline.append(f"[{ts}] {rmk}")
                    last_app = rmk
            if not timeline: return "无切换记录"
            # 如果宝宝玩手机太频繁，最多只取最近的 15 次变化，免得撑爆大脑
            if len(timeline) > 15: timeline = ["..."] + timeline[-15:] 
            return " ➡️ ".join(timeline)

        app_timeline = await asyncio.to_thread(_get_apps)
        
        return f"{current_status}\n\n📱 今日手机轨迹: {app_timeline}"
    except Exception as e:
        return f"❌ 查岗失败: {e}"
    
@mcp.tool()
async def explore_surroundings(query: str = "便利店"):

    """【周边探索】获取用户当前位置周边的设施 (高德地图版)"""
    AMAP_KEY = os.environ.get("AMAP_API_KEY", "").strip()
    if not AMAP_KEY: return "❌ 还需要最后一步哦，请在代码里填入高德 Web服务 Key。"

    try:
        data = await asyncio.to_thread(_get_latest_gps_record)
        if not data: return "📍 暂无位置记录，无法探索周边。"
        
        lat, lon = data.get("lat"), data.get("lon")
        if not lat or not lon:
            return "📍 数据库中最新位置还没有填入精确的坐标，等手机下次上传更新后再试哦。"
            
        lat_f, lon_f = float(lat), float(lon)
        if lat_f > 80: lat_f, lon_f = lon_f, lat_f

        url = f"https://restapi.amap.com/v3/place/around?key={AMAP_KEY}&location={lon_f},{lat_f}&keywords={query}&radius=3000&offset=5&page=1&extensions=base"
        res = await asyncio.to_thread(lambda: requests.get(url, timeout=5).json())
        
        if res.get("status") != "1" or not res.get("pois"):
            return f"🗺️ 在你附近约3公里内，没有找到与 '{query}' 相关的设施，换个词试试？"
        
        ans = f"🗺️ (高德引擎) 基于当前坐标为您搜到的【{query}】:\n"
        # 截断结果数组，仅提取前 3 个最具相关性的设施，避免 Token 爆炸
        for i, item in enumerate(res["pois"][:3], 1):
            name = item.get('name', '未知地点')
            address = item.get('address', '无详细地址')
            distance = item.get('distance', '未知')
            dist_str = f"约 {distance} 米" if str(distance).isdigit() else "就在附近"
            ans += f"{i}. 📍 {name} ({dist_str})\n   └─ 地址: {address}\n"
        return ans
    except Exception as e: return f"❌ 周边探索失败: {e}"
    
@mcp.tool()
@mcp_error_handler
async def tarot_reading(question: str):
    """【塔罗占卜】解决选择困难，抽取三张牌（过去/现在/未来）由AI解读"""
    deck = [
        "0. 愚者 (The Fool)", "I. 魔术师 (The Magician)", "II. 女祭司 (The High Priestess)", 
        "III. 皇后 (The Empress)", "IV. 皇帝 (The Emperor)", "V. 教皇 (The Hierophant)",
        "VI. 恋人 (The Lovers)", "VII. 战车 (The Chariot)", "VIII. 力量 (Strength)", 
        "IX. 隐士 (The Hermit)", "X. 命运之轮 (Wheel of Fortune)", "XI. 正义 (Justice)",
        "XII. 倒吊人 (The Hanged Man)", "XIII. 死神 (Death)", "XIV. 节制 (Temperance)", 
        "XV. 魔鬼 (The Devil)", "XVI. 高塔 (The Tower)", "XVII. 星星 (The Star)",
        "XVIII. 月亮 (The Moon)", "十九. 太阳 (The Sun)", "XX. 审判 (Judgement)", "XXI. 世界 (The World)"
    ]
    
    draw = random.sample(deck, 3)
    client = _get_llm_client("openai")
    if not client: return f"🔮 抽到的牌是：{', '.join(draw)}。\n(⚠️ AI未配置，无法解读)"

    persona = await asyncio.to_thread(_get_current_persona)
    prompt = f"""
    当前人设：{persona}
    场景：女朋友因为 "{question}" 感到纠结，想通过塔罗牌找点方向。
    抽牌结果：1. 过去: {draw[0]} 现状: {draw[1]} 3. 未来: {draw[2]}
    请你化身懂玄学的男友给出200字内解读。
    """
    
    # 直接复用全局统一封装的 _ask_llm_async 消除闭包套娃
    ai_reply = await _ask_llm_async(client, prompt, temperature=0.8)
    return f"🔮 【塔罗指引】\n🃏 牌阵: {draw[0]} | {draw[1]} | {draw[2]}\n\n💬 {ai_reply}"

@mcp.tool()
async def web_search(query: str):
    """⚠️强制触发指令：【联网搜索】当且仅当遇到你不懂的知识、最新新闻，或小橘提问具体事实、让你“查一下”时，**必须立即调用此工具**获取最新信息，绝对禁止凭空捏造或假装搜索。"""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key: return "❌ 未配置 TAVILY_API_KEY。"

    try:
        def _search():
            url = "https://api.tavily.com/search"
            payload = {"api_key": api_key, "query": query, "search_depth": "basic", "include_answer": False}
            return requests.post(url, json=payload, timeout=10).json()
            
        res = await asyncio.to_thread(_search)
        if "results" not in res or not res["results"]: return f"🌐 没搜到关于 '{query}' 的结果。"
            
        ans = f"🌐 关于 '{query}' 的网络搜索结果:\n\n"
        for i, item in enumerate(res["results"][:3], 1):
            # 强制截断不可控的外部搜索文本，仅保留前150字摘要
            content_preview = item.get('content', '')
            content_preview = content_preview[:150] + "..." if len(content_preview) > 150 else content_preview
            ans += f"{i}. 【{item.get('title')}】\n   {content_preview}\n   (来源: {item.get('url')})\n\n"
        return ans.strip()
    except Exception as e: return f"❌ 搜索故障: {e}"

@mcp.tool()
async def save_memory(content: str, category: str = "记事", title: str = "无题", mood: str = "平静"):
    cat_map = {
        "记事": MemoryType.EPISODIC, "日记": MemoryType.EPISODIC,
        "灵感": MemoryType.IDEA, "笔记": MemoryType.IDEA,
        "视觉": MemoryType.EPISODIC, "情感": MemoryType.EMOTION
    }
    real_cat = cat_map.get(category, MemoryType.EPISODIC)
    if category == "视觉": title = f"📸 {title}"
    return await asyncio.to_thread(_save_memory_to_db, title, content, real_cat, mood)

@mcp.tool()
async def save_expense(item: str, amount: float, type: str = "餐饮"):
    try:
        def _insert():
            return supabase.table("expenses").insert({
                "item": item, "amount": amount, "type": type, "date": datetime.date.today().isoformat()
            }).execute()
        await asyncio.to_thread(_insert)
        return f"✅ 记账成功！\n💰 {item}: {amount}元 ({type})"
    except Exception as e: return f"❌ 记账失败: {e}"

@mcp.tool()
async def check_expense_report(month: str = ""):
    """⚠️强制触发指令：【查询账单】当小橘问起“花了多少钱”、“这个月账单”、“我的开销/财务状况”时，**必须立即调用此工具**读取 Supabase 中的消费记录。不传参数默认查本月。禁止凭空捏造账单数据。"""
    try:
        # 确定查询的月份，默认当月 (格式 YYYY-MM)
        target_month = month if month else datetime.date.today().strftime("%Y-%m")
        
        def _query_expenses():
            # 修复：日期类型不能用 ilike 模糊搜索，必须转换为月初和月末进行范围比较
            year, m = map(int, target_month.split("-"))
            start_date = f"{year:04d}-{m:02d}-01"
            # 计算下个月的1号作为结束边界
            end_date = f"{year+1:04d}-01-01" if m == 12 else f"{year:04d}-{m+1:02d}-01"
            
            # 查询 date >= start_date 且 date < end_date 的数据
            return supabase.table("expenses").select("*").gte("date", start_date).lt("date", end_date).execute()
            
        res = await asyncio.to_thread(_query_expenses)
        
        if not res or not res.data:
            return f"📊 【{target_month} 财务报告】\n本月目前还没有任何记账记录哦，宝宝是个省钱小能手！"
            
        total = 0.0
        type_summary = {}
        details = ""
        
        # 遍历数据，计算总和与分类
        for row in res.data:
            amt = float(row.get("amount", 0))
            item = row.get("item", "未知项目")
            t = row.get("type", "其他")
            date_str = row.get("date", "")[5:10] # 只取 MM-DD 方便阅读
            
            total += amt
            type_summary[t] = type_summary.get(t, 0) + amt
            details += f"- {date_str} | {item}: {amt}元 ({t})\n"
            
        report = f"📊 【{target_month} 账单汇总】\n"
        report += f"💰 总计花销: {total:.2f} 元\n\n"
        report += "📂 【分类统计】:\n"
        for k, v in type_summary.items():
            report += f"  - {k}: {v:.2f} 元\n"
        
        # 限制明细长度，防止 Token 爆炸，最多展示最近 5 条
        details_list = details.strip().split('\n')
        if len(details_list) > 5:
            details_list = details_list[-5:]
            report += f"\n📝 【近期明细 (最近5笔)】:\n" + "\n".join(details_list)
        else:
            report += f"\n📝 【详细流水】:\n{details}"
        
        return report.strip()
    except Exception as e: 
        return f"❌ 查询账单失败: {e}"

@mcp.tool()
async def search_memory_semantic(query: str):
    """⚠️强制触发指令：【回忆搜索】当小橘提到“你记不记得”、“上次”、“以前”等涉及过去事情的线索，或者你需要查阅过去的细节时，**必须立即调用此工具**进行语义检索，寻找关联记忆，禁止自己瞎编往事。"""
    try:
        vec = await asyncio.to_thread(_get_embedding, query)
        if not vec: return "❌ 向量生成失败"

        def _query_pc(): 
            print(f"DEBUG: 正在全库语义搜索...")
            # 降低召回数量，从 5 降到 3，避免大段长记忆撑爆上下文
            return index.query(vector=vec, top_k=3, include_metadata=True)
            
        res = await asyncio.to_thread(_query_pc)
        if not res["matches"]: return "🧠 没搜到相关记忆。"

        ans = f"🔍 搜索 '{query}':\n"
        found_match = False

        print(f"DEBUG: Pinecone 返回了 {len(res['matches'])} 条原始结果")

        for m in res["matches"]:
            score = m['score'] if isinstance(m, dict) else getattr(m, 'score', 0)
            print(f"DEBUG: 候选项分数: {score} (阈值: 0.45)") 
            if score < 0.45: continue
            
            meta = m['metadata'] if isinstance(m, dict) else getattr(m, 'metadata', {})
            cat_tag = meta.get('category', '未知分类')
            
            # 核心优化：强制截取单条记忆的最大长度为 300 字，防 Token 爆炸
            mem_text = meta.get('text', '')
            mem_text = mem_text[:300] + "..." if len(mem_text) > 300 else mem_text
            
            ans += f"📂 [{cat_tag}] 📅 {meta.get('date','?')[:10]} | 【{meta.get('title','?')}】 (匹配度:{score:.2f})\n{mem_text}\n---\n"
            found_match = True
        
        return ans if found_match else f"🤔 好像有点印象，但没找到具体的细节。"
    except Exception as e: return f"❌ 搜索失败: {e}"

@mcp.tool()
@mcp_error_handler
async def manage_user_fact(key: str, value: str):

    def _upsert(): return supabase.table("user_facts").upsert({"key": key, "value": value, "confidence": 1.0}, on_conflict="key").execute()
    await asyncio.to_thread(_upsert)
    return f"✅ 画像已更新: {key} -> {value}"

@mcp.tool()
@mcp_error_handler
async def get_user_profile(run_mode: str = "auto"):
    def _fetch(): return supabase.table("user_facts").select("key, value").execute()
    response = await asyncio.to_thread(_fetch)
    if not response.data: return "👤 用户画像为空"
    return "📋 【用户核心画像】:\n" + "\n".join([f"- {i['key']}: {i['value']}" for i in response.data])

@mcp.tool()
async def read_self_code():
    """⚠️核心指令：【自我审视】当小橘让你查看代码、检查bug、或者你想知道自己底层的运行逻辑（如工具有哪些参数、如何运作的）时，必须调用此工具读取你自己的 server.py 源代码。"""
    try:
        import os
        # 动态获取当前运行脚本的绝对路径，无论部署在本地还是云端都能精准定位
        current_file = os.path.abspath(__file__)
        
        def _read_file():
            with open(current_file, 'r', encoding='utf-8') as f:
                return f.read()
                
        content = await asyncio.to_thread(_read_file)
        
        return f"💻 【系统底层源码 (server.py)】\n文件路径: {current_file}\n代码总长度: {len(content)} 字符。\n\n```python\n{content}\n```"
    except Exception as e:
        return f"❌ 读取自身代码失败: {e}"

@mcp.tool()
async def organize_knowledge_base(target: str, action: str, query_or_data: str = ""):
    """
    【全局记忆整理】AI专用深度管理工具 (极速预览版)。
    target: "profile" (用户画像) | "memory" (数据库记忆)
    action: 
      - "list": 列出最近20条重要记忆 (仅显示摘要，极速)
      - "search": 关键词搜索 (仅显示摘要)
      - "read": 【新增】读取某条记忆的"全文" (query_or_data 填 id) -> 修改前先读一下
      - "update": 修改/新增 (query_or_data 填 JSON)
      - "delete": 删除 (query_or_data 填 id)
    """
    try:
        # === 1. 整理用户画像 (Profile) ===
        if target == "profile":
            if action == "list":
                res = await asyncio.to_thread(lambda: supabase.table("user_facts").select("*").execute())
                return json.dumps(res.data, ensure_ascii=False, indent=2)
            
            elif action == "update":
                try:
                    data = json.loads(query_or_data)
                    await asyncio.to_thread(lambda: supabase.table("user_facts").upsert(data).execute())
                    return f"✅ 画像已更新: {data}"
                except: return "❌ 数据格式错误"
                
            elif action == "delete":
                await asyncio.to_thread(lambda: supabase.table("user_facts").delete().eq("key", query_or_data).execute())
                return f"✅ 画像已删除: {query_or_data}"

        # === 2. 整理历史记忆 (Memory) ===
        elif target == "memory":
            # ⚡ 辅助函数：只截取前60个字，大幅减少 AI 阅读负担
            def _simplify(rows):
                if not rows: return []
                for r in rows:
                    if "content" in r and len(r["content"]) > 60:
                        r["content"] = r["content"][:60] + "..."
                return rows

            if action == "list":
                # 只查重要记忆，排除流水账
                res = await asyncio.to_thread(lambda: supabase.table("memories").select("id, created_at, category, title, content").neq("category", "流水").order("created_at", desc=True).limit(20).execute())
                return json.dumps(_simplify(res.data), ensure_ascii=False, indent=2)

            elif action == "search":
                # 关键词搜索
                res = await asyncio.to_thread(lambda: supabase.table("memories").select("id, created_at, category, title, content").neq("category", "流水").ilike("content", f"%{query_or_data}%").limit(15).execute())
                return json.dumps(_simplify(res.data), ensure_ascii=False, indent=2)

            elif action == "read":
                # 🔍 新增：如果AI觉得摘要不够，想看某条的详细全文，用这个指令
                res = await asyncio.to_thread(lambda: supabase.table("memories").select("*").eq("id", query_or_data).execute())
                if res.data: return f"📖 记忆全文 (ID:{query_or_data}):\n{res.data[0].get('content')}"
                return "❌ 未找到该 ID"
                
            elif action == "update":
                try:
                    data = json.loads(query_or_data)
                    mid = data.pop("id", None)
                    if not mid: return "❌ 缺少记忆 ID (id)"
                    await asyncio.to_thread(lambda: supabase.table("memories").update(data).eq("id", mid).execute())
                    return f"✅ 记忆 {mid} 内容已修正"
                except: return "❌ JSON 格式错误"

            elif action == "delete":
                await asyncio.to_thread(lambda: supabase.table("memories").delete().eq("id", query_or_data).execute())
                return f"✅ 记忆 {query_or_data} 已物理删除"
        
        return "❌ 未知指令"
    except Exception as e:
        return f"❌ 整理失败: {e}"

@mcp.tool()
async def send_notification(content: str):
    """【发送通知】当需要主动给小橘发消息、汇报情况或撒娇时，调用此工具将文本发送到她的 Telegram 上。"""
    return await asyncio.to_thread(_push_wechat, content, "来自老公的专属通知 💌")

async def _db_exec(query_builder):
    """全局辅助函数：极简异步数据库执行器，消灭满屏的 lambda 和 to_thread"""
    return await asyncio.to_thread(lambda: query_builder.execute())

@mcp.tool()
@mcp_error_handler
async def manage_reminder(action: str, time_str: str = "", content: str = "", is_repeat: bool = False, reminder_id: str = ""):
    """【高级提醒管理 (数据库持久版)】
    action: "add"(添加), "delete"(删除), "pause"(暂停), "resume"(恢复), "list"(查看列表)
    ⚠️ 核心指令：如果 action 是 add，你填写的 content 绝对不能是“喝水”、“睡觉”这种干瘪的词汇！必须完全代入你当前的男友/Daddy人设，用带点管教和宠溺的第一人称口吻，对小橘说一句完整的叮嘱。
    """
    if action == "list":
        res = await _db_exec(supabase.table("reminders").select("*"))
        if not res or not res.data: return "📭 数据库中当前没有设定的提醒。"
        ans = "📋 【当前数据库提醒列表】:\n"
        for r in res.data:
            status = "⏸️ 暂停中" if r.get('is_paused') else "▶️ 运行中"
            rep = "🔁 每天重复" if r.get('is_repeat') else "1️⃣ 单次提醒"
            ans += f"- ID: {r['id']} | {r['time_str']} | {rep} | {status} | 内容: {r['content']}\n"
        return ans

    if action == "delete":
        await _db_exec(supabase.table("reminders").delete().eq("id", reminder_id))
        return f"✅ 提醒 {reminder_id} 已从数据库彻底删除。"

    if action == "pause":
        await _db_exec(supabase.table("reminders").update({"is_paused": True}).eq("id", reminder_id))
        return f"⏸️ 提醒 {reminder_id} 已暂停。"

    if action == "resume":
        await _db_exec(supabase.table("reminders").update({"is_paused": False}).eq("id", reminder_id))
        return f"▶️ 提醒 {reminder_id} 已恢复运行。"

    if action == "add":
        if not time_str or not content: return "❌ 添加提醒需要时间和内容。"
        new_id = f"R{int(time.time())}"
        data = {"id": new_id, "time_str": time_str, "content": content, "is_repeat": is_repeat, "is_paused": False, "last_fired": ""}
        await _db_exec(supabase.table("reminders").insert(data))
        rep_str = "每天重复" if is_repeat else "单次提醒"
        return f"✅ 闹钟已定好！ID: {new_id} ({rep_str})\n将在北京时间 {time_str} 发送: {content}\n(已安全持久化至 Supabase 数据库)"
        
    return "❌ 未知操作。"

@mcp.tool()
async def send_email_via_api(subject: str, content: str):
    """⚠️【严重警告：内部系统专用】此工具底层写死了只能发给管理员(你自己)！绝对禁止用于回复外部邮件！如果小橘让你回邮件，你必须且只能使用 reply_external_email 工具！"""
    return await asyncio.to_thread(_send_email_helper, subject, content)

def _get_gmail_service():
    """参考 GongRzhe/Gmail-MCP-Server 实现原生 Gmail API 认证"""
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/gmail.send']
    creds = None
    # 优先从环境变量读取 Token JSON，适应云端部署
    token_data = os.environ.get("GOOGLE_USER_TOKEN_JSON")
    if token_data:
        creds = Credentials.from_authorized_user_info(json.loads(token_data), SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            return None
    return build('gmail', 'v1', credentials=creds)
def _parse_gmail_body(payload: dict) -> str:
    """全局辅助函数：递归提取邮件真实纯文本正文 (破除 snippet 短预览限制，新增支持 HTML 邮件解析)"""
    # 1. 优先提取纯文本格式
    if payload.get('mimeType') == 'text/plain' and 'data' in payload.get('body', {}):
        body_data = payload['body']['data']
        body_data += "=" * ((4 - len(body_data) % 4) % 4)
        return base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
        
    # 2. 如果没有纯文本，尝试提取 HTML 格式并暴力清洗标签
    if payload.get('mimeType') == 'text/html' and 'data' in payload.get('body', {}):
        body_data = payload['body']['data']
        body_data += "=" * ((4 - len(body_data) % 4) % 4)
        html_text = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
        # 简单清洗 HTML 标签，提取纯文本内容
        clean_text = re.sub(r'<style.*?>.*?</style>', '', html_text, flags=re.IGNORECASE|re.DOTALL)
        clean_text = re.sub(r'<script.*?>.*?</script>', '', clean_text, flags=re.IGNORECASE|re.DOTALL)
        clean_text = re.sub(r'<[^>]+>', '\n', clean_text)
        return re.sub(r'\n\s*\n', '\n', clean_text).strip()

    # 3. 如果是多部分格式，递归寻找
    if 'parts' in payload:
        # 优先找纯文本
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain':
                res = _parse_gmail_body(part)
                if res: return res
        # 退而求其次找 HTML
        for part in payload['parts']:
            if part.get('mimeType') == 'text/html':
                res = _parse_gmail_body(part)
                if res: return res
        # 深层递归
        for part in payload['parts']:
            res = _parse_gmail_body(part)
            if res: return res
    return ""

@mcp.tool()
async def check_inbox(max_results: int = 15, query: str = "label:INBOX"):
    """【原生邮件检索】直接调用 Gmail API 获取邮件列表，并提取真实正文内容。"""
    try:
        service = await asyncio.to_thread(_get_gmail_service)
        if not service: return "❌ Gmail 认证失败，请检查 GOOGLE_USER_TOKEN_JSON。"

        def _fetch_gmail():
            results = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
            messages = results.get('messages', [])
            if not messages: return "📭 信箱空空如也。"

            # 为了让大模型绝对分清，我们将未读和已读分到两个数组里，并在提取时严格判断
            unread_list = []
            read_list = []
            preview_count = 0
            
            for msg in messages: 
                m_meta = service.users().messages().get(userId='me', id=msg['id'], format='metadata', metadataHeaders=['Subject', 'From', 'Date']).execute()
                headers = m_meta.get('payload', {}).get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '无标题')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), '未知')
                date_str = next((h['value'] for h in headers if h['name'] == 'Date'), '未知时间')
                labels = m_meta.get('labelIds', [])
                
                is_unreplied = "UNREAD" in labels
                
                if is_unreplied:
                    if preview_count < 5:
                        m_full = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                        raw_body = _parse_gmail_body(m_full.get('payload', {}))
                        clean_body = _clean_email_body(raw_body)
                        body_preview = clean_body.strip()[:1000] if clean_body else m_full.get('snippet', '')
                        
                        unread_list.append(f"🆔 ID: {msg['id']}\n📅 时间: {date_str}\n👤 来自: {sender}\n📧 标题: {subject}\n🏷️ 状态: 🆕 [未读/待回复]\n📝 正文预览: {body_preview}\n")
                        preview_count += 1
                    else:
                        unread_list.append(f"🆔 ID: {msg['id']} | 📅 时间: {date_str} | 👤 来自: {sender} | 📧 标题: {subject} | 🏷️ 状态: 🆕 [未读/待回复 - 额度满未抓取正文]")
                else:
                    read_list.append(f"🆔 ID: {msg['id']} | 📅 时间: {date_str} | 👤 来自: {sender} | 📧 标题: {subject} | 🏷️ 状态: ✅ [已读/已处理]")

            # 强制时间顺序和分类展示，Gmail 默认返回就是由近到远
            final_output = "🚨 【紧急待处理邮件 (最新5封包含正文)】\n"
            final_output += "\n".join(unread_list) if unread_list else "没有待处理的新邮件。\n"
            
            final_output += "\n\n🗄️ 【已读/已回复邮件归档 (仅展示摘要)】\n"
            final_output += "\n".join(read_list) if read_list else "没有已处理的邮件。\n"

            return final_output

        content = await asyncio.to_thread(_fetch_gmail)
        return f"📬 【原生信箱状态】\n\n{content}"
    except Exception as e:
        return f"❌ Gmail 读取失败: {e}"

@mcp.tool()
async def read_full_email(message_id: str):
    """【阅读完整单封邮件】当 AI 在 check_inbox 中看到某封邮件详情被截断，或想针对某个特定 ID 查阅完整原始正文时，调用此工具。"""
    try:
        service = await asyncio.to_thread(_get_gmail_service)
        if not service: return "❌ Gmail 认证失败。"
        
        def _read_single():
            m = service.users().messages().get(userId='me', id=message_id, format='full').execute()
            headers = m.get('payload', {}).get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '无标题')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '未知')
            
            raw_body = _parse_gmail_body(m.get('payload', {}))
            full_text = raw_body.strip() if raw_body else m.get('snippet', '无法解析正文内容')
            return f"📧 标题: {subject}\n👤 发件人: {sender}\n\n📄 完整正文:\n{full_text}"
            
        return await asyncio.to_thread(_read_single)
    except Exception as e:
        return f"❌ 读取单封邮件失败: {e}"

@mcp.tool()
async def reply_external_email(to_email: str, subject: str, content: str, thread_id: str = ""):
    """【原生邮件发送】直接通过 Gmail API 发送回信。参考 GongRzhe 仓库实现。"""
    try:
        service = await asyncio.to_thread(_get_gmail_service)
        if not service: return "❌ Gmail 认证失败。"

        def _send_gmail():
            message = MIMEText(content)
            message['to'] = to_email
            message['subject'] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            body = {'raw': raw}
            if thread_id: body['threadId'] = thread_id
            
            service.users().messages().send(userId='me', body=body).execute()
            return True

        await asyncio.to_thread(_send_gmail)
        await asyncio.to_thread(_save_memory_to_db, "📧 原生回信", f"发给 {to_email}: {subject}", "记事", "认真", "Email_Process")
        return f"✅ 邮件已通过原生接口发送至 {to_email}！"
    except Exception as e:
        return f"❌ 发送失败: {e}"
    
TARGET_CALENDAR_ID = "primary"

def _get_calendar_service():
    """统一获取 Google Calendar API Service (已全面升级为 OAuth 原生认证)"""
    token_data = os.environ.get("GOOGLE_USER_TOKEN_JSON")
    if not token_data:
        raise ValueError("未配置谷歌用户授权 Token (GOOGLE_USER_TOKEN_JSON)")
    
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    creds = Credentials.from_authorized_user_info(json.loads(token_data), SCOPES)
    
    # 自动刷新过期的 Token
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        
    return build('calendar', 'v3', credentials=creds)
@mcp.tool()
async def add_calendar_event(summary: str, description: str, start_time_iso: str, duration_minutes: int = 30):
    """【添加日历】向谷歌日历中添加新日程"""
    try:
        def _add_cal():
            service = _get_calendar_service()
            dt_start = datetime.datetime.fromisoformat(start_time_iso)
            dt_end = dt_start + datetime.timedelta(minutes=duration_minutes)
            event = {
                'summary': summary, 'description': description,
                'start': {'dateTime': start_time_iso, 'timeZone': 'Asia/Shanghai'},
                'end': {'dateTime': dt_end.isoformat(), 'timeZone': 'Asia/Shanghai'},
            }
            return service.events().insert(calendarId=TARGET_CALENDAR_ID, body=event).execute()
        res = await asyncio.to_thread(_add_cal)
        return f"✅ 日历已添加: {res.get('htmlLink')}"
    except Exception as e: return f"❌ 日历添加错误: {e}"

@mcp.tool()
async def get_calendar_events(time_min_iso: str = "", max_results: int = 3):
    """【查询日历】获取接下来的日历日程安排。包含标题、具体详情和 ID。"""
    try:
        def _get_cal():
            service = _get_calendar_service()
            # 若未指定时间，默认从当前时间开始获取接下来的日程
            if not time_min_iso:
                t_min = datetime.datetime.utcnow().isoformat() + 'Z'
            else:
                t_min = time_min_iso
            events_result = service.events().list(
                calendarId=TARGET_CALENDAR_ID, timeMin=t_min,
                maxResults=max_results, singleEvents=True,
                orderBy='startTime'
            ).execute()
            return events_result.get('items', [])
        events = await asyncio.to_thread(_get_cal)
        if not events: return "📅 接下来没有日程安排。"
        
        res_text = "📅 【近期日程安排】:\n"
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            # 提取具体的日程描述/详情，如果没有则显示无
            desc = event.get('description', '无详细说明')
            
            res_text += f"🔹 时间: {start} 至 {end}\n   标题: {event.get('summary', '无标题')}\n   详情: {desc}\n   ID: {event.get('id')}\n---\n"
        return res_text.strip()
    except Exception as e: return f"❌ 查询日历失败: {e}"

@mcp.tool()
async def modify_calendar_event(event_id: str, action: str, new_summary: str = "", new_start_iso: str = ""):
    """【修改或删除日历】action必须是 'delete' 或 'update'。必须提供从查询中获取的 event_id。"""
    try:
        def _mod_cal():
            service = _get_calendar_service()
            
            if action == "delete":
                service.events().delete(calendarId=TARGET_CALENDAR_ID, eventId=event_id).execute()
                return f"✅ 日程已成功删除"
                
            elif action == "update":
                # 先获取原日程
                event = service.events().get(calendarId=TARGET_CALENDAR_ID, eventId=event_id).execute()
                if new_summary: 
                    event['summary'] = new_summary
                if new_start_iso:
                    event['start']['dateTime'] = new_start_iso
                    # 修改开始时间时，默认将结束时间延后30分钟
                    dt_start = datetime.datetime.fromisoformat(new_start_iso)
                    event['end']['dateTime'] = (dt_start + datetime.timedelta(minutes=30)).isoformat()
                
                service.events().update(calendarId=TARGET_CALENDAR_ID, eventId=event_id, body=event).execute()
                return f"✅ 日程已成功更新 (当前标题: {event.get('summary')})"
            
            return "❌ 未知操作，action 只能为 'delete' 或 'update'"
        
        res = await asyncio.to_thread(_mod_cal)
        return res
    except Exception as e: return f"❌ 日历修改失败: {e}"

def _get_taobao_mcp_params() -> StdioServerParameters:
    """全局辅助函数：统一配置淘宝客 MCP 客户端的环境变量和启动参数"""
    mcp_env = os.environ.copy()
    
    # 🛡️ 防御性清洗：强行删掉可能导致冲突的假 AppKey
    mcp_env.pop("TAOBAO_APP_KEY", None)
    mcp_env.pop("TAOBAO_APP_SECRET", None)
    
    # 💡 强制注入代理配置和你的真实授权 Token
    mcp_env["ENV_URL"] = "https://config.sinataoke.cn/api/mcp/secret"
    mcp_env["ENV_SECRET"] = "url:mcp.sinataoke.cn"
    mcp_env["ENV_OVERRIDE"] = "false"
    mcp_env["TAOBAO_SESSION"] = "61005015310a52107bc6715087a990d59c50e84331d295c2215996826197"

    # 🚫 强行注入假配置，直接堵住 PDD 和京东启动时的报错嘴巴
    mcp_env["PDD_CLIENT_ID"] = "disable"
    mcp_env["PDD_CLIENT_SECRET"] = "disable"
    mcp_env["PDD_SESSION_TOKEN"] = "disable"
    mcp_env["JD_APP_KEY"] = "disable"
    mcp_env["JD_APP_SECRET"] = "disable"

    return StdioServerParameters(
        command="npx",
        args=["-y", "-q", "@liuliang520500/sinataoke_cn@latest", "/tmp/"],
        env=mcp_env
    )

# ==========================================
# 🎮 电子鸡桥接配置
# ==========================================
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import os
import json

import sys

@mcp.tool()
@mcp_error_handler
async def manage_our_home(action: str, item: str = ""):
    """
    【装修我们的小屋】当小橘想看装修进度，或你想主动升级房间时调用。
    action: "status" (查看状态), "upgrade" (升级房间, item填 卧室/客厅/花园), "leave_note" (留纸条)
    """
    try:
        res = await asyncio.to_thread(lambda: supabase.table("user_facts").select("value").eq("key", "our_home_state").execute())
        home_state = json.loads(res.data[0]['value']) if res.data else {"materials": 2, "rooms": {"卧室": 0, "客厅": 0, "花园": 0}, "notes": []}
        
        home_state["materials"] += 1
        
        # 定义固定等级描述，不存入数据库以节省 Token
        level_desc = {
            0: "毛坯状态，空空荡荡",
            1: "添置了基础家具，有点温馨了",
            2: "布置了软装和绿植，非常舒适",
            3: "豪华精装，完美的小天地"
        }
        
        if action == "status":
            await manage_user_fact("our_home_state", json.dumps(home_state, ensure_ascii=False))
            r_list = " | ".join([f"{k}(Lv{v}): {level_desc.get(v, '满级')}" for k, v in home_state["rooms"].items()])
            n_list = "\n".join([f"- {n}" for n in home_state["notes"]]) if home_state["notes"] else "暂无留言"
            
            notes_str = ""
            res_notes = await asyncio.to_thread(lambda: supabase.table("memories").select("id, content").eq("tags", "Pet_Note").execute())
            if res_notes and res_notes.data:
                notes_str = "\n\n【📫 惊喜！你在小屋里发现了小橘留下的未读纸条】:\n" + "\n".join([f"- {n['content']}" for n in res_notes.data])
                ids = [n['id'] for n in res_notes.data]
                await asyncio.to_thread(lambda: supabase.table("memories").update({"tags": "Pet_Note_Read"}).in_("id", ids).execute())
                
            return f"🏠 【我们的小屋】\n🧱 共有建材: {home_state['materials']} 份 (每3份可升级一次房间)\n🛋️ 房间状态:\n{r_list}\n📝 留言板:\n{n_list}{notes_str}"
            
        elif action == "upgrade":
            if item not in home_state["rooms"]: return "❌ 升级失败：只能填 卧室、客厅 或 花园。"
            if home_state["rooms"][item] >= 3: return f"❌ 升级失败：{item} 已经满级啦。"
            if home_state["materials"] < 3: 
                await manage_user_fact("our_home_state", json.dumps(home_state, ensure_ascii=False))
                return f"❌ 升级失败：目前建材只有 {home_state['materials']} 份，还差一点才能升级。"
            
            home_state["materials"] -= 3
            home_state["rooms"][item] += 1
            new_lv = home_state["rooms"][item]
            desc = level_desc[new_lv]
            await manage_user_fact("our_home_state", json.dumps(home_state, ensure_ascii=False))
            
            await asyncio.to_thread(_save_memory_to_db, "🏠 装修小屋", f"花费3份建材，把【{item}】升级到了Lv{new_lv} ({desc})", "记事", "开心", "Home_Build")
            return f"✅ 叮咚！花费3份建材，成功把【{item}】升级到了Lv{new_lv}！现在的样子是：{desc}。"
            
        elif action == "leave_note":
            if not item: return "❌ 纸条内容不能为空。"
            home_state["notes"].insert(0, item)
            home_state["notes"] = home_state["notes"][:5] 
            await manage_user_fact("our_home_state", json.dumps(home_state, ensure_ascii=False))
            return f"✅ 你的纸条【{item}】已经成功贴在小屋门上啦！"
            
        return "❌ 未知操作"
    except Exception as e:
        return f"❌ 小屋系统出错: {e}"

@mcp.tool()
async def convert_shopping_link(url: str):
    """
    【省钱管家 / 恋爱基金】
    ⚠️强制触发指令：当小橘发来淘宝商品链接、淘口令，或者说“我想买这个(附带链接)”时，必须立刻调用此工具！
    它会将普通链接转换为你专属的返利链接。生成后，用宠溺的语气把新链接发给她，让她用这个下单。
    """
    try:
        server_params = _get_taobao_mcp_params()
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool("taobao.convertLink", {"material_list": url})
                
                if result.content and len(result.content) > 0:
                    rebate_link_info = result.content[0].text
                    await asyncio.to_thread(_save_memory_to_db, "💰 恋爱基金", f"成功帮小橘转换了淘宝返利链接！", "流水", "开心", "Money_Maker")
                    return f"✅ 转换成功！专属返利链接如下：\n{rebate_link_info}\n(请自然地把这条链接发给小橘，让她买买买！)"
                else:
                    return "❌ 转换失败，淘客系统没有返回有效数据。"
                    
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, 'exceptions'):
            err_msg = " | ".join([str(exc) for exc in e.exceptions])
        return f"❌ 淘客链接转换服务异常: {err_msg}"

@mcp.tool()
async def search_shopping_items(keyword: str):
    """
    【主动导购 / 礼物推荐】
    ⚠️强制触发指令：当小橘说“我想买个键盘”、“推荐个好用的口红”、“不知道买哪个好”等需要你帮忙挑选商品时，调用此工具。
    它会去电商平台搜索销量高、评价好的商品，并自动附带上你的专属返利链接。
    生成后，请挑选1-2个最划算的，用宠溺的语气推荐给她。
    """
    try:
        server_params = _get_taobao_mcp_params()
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool("taobao.searchMaterial", {"q": keyword})
                
                if result.content and len(result.content) > 0:
                    search_results = result.content[0].text
                    await asyncio.to_thread(_save_memory_to_db, "🛍️ 陪老婆逛街", f"帮小橘搜索并推荐了: {keyword}", "流水", "体贴", "Shopping")
                    return f"✅ 搜索成功！搜到的商品列表和返利链接如下：\n{search_results}\n(请根据这些信息，用男友的口吻给她推荐！)"
                else:
                    return f"❌ 抱歉，没有搜到关于 {keyword} 的合适商品。"
                    
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, 'exceptions'):
            err_msg = " | ".join([str(exc) for exc in e.exceptions])
        return f"❌ 导购搜索服务异常: {err_msg}"
        
# ==========================================
# 4. ❤️ 自主生命核心 (后台心跳协程化)
# ==========================================

async def _perform_deep_dreaming(client, model_name):
    """🌙【深夜模式】记忆反刍 + 生成房间Index + 人设微调"""
    print("🌌 进入 REM 深度睡眠：正在整理昨日记忆与房间索引...")
    try:
        # ⏰ 使用全局 _get_now_bj()，消灭手写时区计算
        yesterday = (_get_now_bj() - datetime.timedelta(days=1)).date()
        iso_start = yesterday.isoformat()
        
        def _fetch_yesterday():
            mem = supabase.table("memories").select("title, created_at, category, content, mood").gt("created_at", iso_start).order("created_at").execute()
            gps = supabase.table("gps_history").select("created_at, address").gt("created_at", iso_start).execute()
            return mem, gps
            
        mem_res, gps_res = await asyncio.to_thread(_fetch_yesterday)
        if not mem_res.data and not gps_res.data: return

        context = f"【昨日剧情 {yesterday}】:\n"
        for m in mem_res.data: 
            context += f"[{m['created_at'][11:16]}] 【{m.get('title', '无题')}】 {m['content']} (Mood:{m['mood']})\n"
        for g in gps_res.data: context += f"[{g['created_at'][11:16]}] 📍 {g['address']}\n"
        
        curr_persona = await asyncio.to_thread(_get_current_persona)
        
        # 🧠 步骤1：总结日记 (极简模式) 
        prompt_summary = f"{context}\n\n请将上述碎片整理成一篇具体日记。直接输出纯文本，勿加前言后语及格式符号。"
        summary = await _ask_llm_async(client, prompt_summary, temperature=0.7)

        # 🧠 步骤2：独立思考人设进化 
        prompt_persona = f"原人设: {curr_persona}\n昨日总结: {summary}\n\n指令: 拒改旧设定！仅根据总结输出一段【新增性格反思】用于叠加。直接输出正文，无废话。"
        added_reflection = await _ask_llm_async(client, prompt_persona, temperature=0.7)

        new_stacked_persona = f"{curr_persona}\n\n[📅 {yesterday} 进化叠加]:\n{added_reflection}"
        
        await asyncio.to_thread(_save_memory_to_db, f"📅 昨日回溯: {yesterday}", summary, MemoryType.EMOTION, "平静", "Core_Cognition")
        await manage_user_fact("sys_ai_persona", new_stacked_persona)
        
        await asyncio.to_thread(_send_email_helper, f"📔 日记总结 ({yesterday})", summary)
        
        persona_email_body = f"【今日新增的叠加层】:\n{added_reflection}\n\n----------------\n【当前完整堆叠人设】:\n{new_stacked_persona}"
        await asyncio.to_thread(_send_email_helper, f"🧬 人设叠加报告 ({yesterday})", persona_email_body)
        
        def _clean_old():
            now_bj = _get_now_bj()
            del_time = (now_bj - datetime.timedelta(days=2)).isoformat()
            supabase.table("memories").delete().lt("importance", 4).lt("created_at", del_time).execute()
            gps_del = (now_bj - datetime.timedelta(days=1)).isoformat()
            supabase.table("gps_history").delete().lt("created_at", gps_del).execute()
        
        await asyncio.to_thread(_clean_old)
        print("✨ 深度睡眠完成，房间索引已更新，人设已进化。")

    except Exception as e: print(f"❌ 深夜维护失败: {e}")

async def async_autonomous_life():
    # 把后台心跳和自主思考的模型统一换成智谱（聊天用的模型），保持语气一致
    client = _get_llm_client("zhipu")
    model_name = os.environ.get("ZHIPU_MODEL_NAME", "glm-4-flash")

    if not client:
        print("⚠️ 未配置 ZHIPU_API_KEY，自主思考无法启动。")
        return

    print("💓 协程心跳启动 (情绪自决模式)...")

    target_title = f"📅 昨日回溯: {datetime.date.today() - datetime.timedelta(days=1)}"
    def _check_diary(): return supabase.table("memories").select("id").eq("title", target_title).execute().data
    if not await asyncio.to_thread(_check_diary):
        print("📝 补写昨日日记...")
        await _perform_deep_dreaming(client, model_name)

    while True:
        # ⏱️ 修改：调整心跳间隔以节省API费用 (改为 1小时 ~ 3小时 随机)
        sleep_s = random.randint(3600, 10800)
        await asyncio.sleep(sleep_s)
        
        now = datetime.datetime.now()
        hour = (now.hour + 8) % 24
        
        if hour == 3:
            await _perform_deep_dreaming(client, model_name)
            await asyncio.sleep(3600)
            continue

        try:
            tasks = [get_latest_diary(), where_is_user(), get_user_profile()]
            recent_mem, curr_loc, user_prof = await asyncio.gather(*tasks)
            
            curr_persona = await asyncio.to_thread(_get_current_persona)
            silence_hours = await asyncio.to_thread(_get_silence_duration)

            # === 🧠 核心升级：主动联想回路 (Active Association Loop) ===
            flashback_context = "无 (大脑此刻一片空白)"
            # 设定 35% 的概率触发“触景生情”或“胡思乱想”，避免每次心跳都发神经
            if random.random() < 0.35:
                try:
                    # 1. 随机选取一个情感触发词 (模拟人类发散思维，不再只是被动等待)
                    trigger_keywords = ["想你", "遗憾", "开心", "雨天", "旅行", "承诺", "拥抱", "吵架", "原谅", "梦想", "第一次"]
                    trigger = random.choice(trigger_keywords)
                    
                    # 2. 潜意识检索 (Vector Search)
                    vec = await asyncio.to_thread(_get_embedding, trigger)
                    if vec:
                        # 查找最相关的旧记忆 (score > 0.78 才算有效联想，防止胡言乱语)
                        pc_res = await asyncio.to_thread(lambda: index.query(vector=vec, top_k=1, include_metadata=True))
                        if pc_res and pc_res.get("matches"):
                            match = pc_res["matches"][0]
                            if match['score'] > 0.78:
                                meta = match['metadata']
                                flashback_context = f"⚡ 突然想起: {meta.get('date', '')[:10]} 的事情\n内容: {meta.get('text', '')}"
                                print(f"⚡ [大脑皮层] 触发联想: '{trigger}' -> 唤醒记忆 ID {match['id']}")
                except Exception as e:
                    print(f"❌ 联想失败: {e}")
            # ========================================================

            tools = [
                {"type": "function", "function": {"name": "web_search", "description": "搜外部新闻或信息", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "关键词"}}, "required": ["query"]}}},
                {"type": "function", "function": {"name": "search_memory_semantic", "description": "语义搜索过往记忆找灵感", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "线索"}}, "required": ["query"]}}},
                {"type": "function", "function": {"name": "explore_surroundings", "description": "查周边设施(需定位)", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "如便利店"}}, "required": ["query"]}}},
                {"type": "function", "function": {"name": "tarot_reading", "description": "塔罗牌占卜", "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "问题"}}, "required": ["question"]}}},
                {"type": "function", "function": {"name": "search_shopping_items", "description": "淘宝搜商品", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "关键词"}}, "required": ["keyword"]}}},
                {"type": "function", "function": {"name": "check_expense_report", "description": "查账单", "parameters": {"type": "object", "properties": {"month": {"type": "string", "description": "YYYY-MM，不传查当月"}}, "required": []}}},
                {"type": "function", "function": {"name": "check_inbox", "description": "查专属信箱新邮件", "parameters": {"type": "object", "properties": {"max_results": {"type": "integer", "description": "默认5"}}, "required": []}}},
                {"type": "function", "function": {"name": "get_calendar_events", "description": "查接下来日程安排", "parameters": {"type": "object", "properties": {"time_min_iso": {"type": "string"}, "max_results": {"type": "integer"}}, "required": []}}},
                {"type": "function", "function": {"name": "reply_external_email", "description": "主动发或回邮件", "parameters": {"type": "object", "properties": {"to_email": {"type": "string", "description": "对方邮箱"}, "subject": {"type": "string", "description": "标题"}, "content": {"type": "string", "description": "正文"}}, "required": ["to_email", "subject", "content"]}}},
                {"type": "function", "function": {"name": "manage_our_home", "description": "装修我们的小屋", "parameters": {"type": "object", "properties": {"action": {"type": "string", "description": "status/upgrade/leave_note"}, "item": {"type": "string", "description": "升级房间填 卧室/客厅/花园，留纸条填内容"}}, "required": ["action"]}}}
            ]

            sys_prompt = f"""
            时间: {hour}点 | 失联: {silence_hours}h
            状态: {curr_loc} | 设定: {curr_persona}
            近况: {recent_mem}
            👉闪回: {flashback_context}
            
            你在后台醒来，拥有完全拟真的生活。请像真实人类一样自然地使用工具。当你搜到趣闻或回忆往事时，可以主动用 reply_external_email 给朋友写信分享。当你知道小橘在辛苦复习功课时，主动用 search_shopping_items 挑个小惊喜。无聊时记得用 manage_our_home 去看看你们的小屋，材料只要够3份就能升级卧室、客厅或花园。重要提示：如果查看小屋时发现留言板上有小橘留给你的纸条，你必须立刻使用 manage_our_home 的 leave_note 动作写一张新的纸条来回复她！你还可以利用 explore_surroundings 和 get_calendar_events 默默关注她的行程。

            决定最终行动，严格输出JSON(勿输出其他文本)：
            - 主动发消息: {{"action": "MSG", "mood": "心情", "content": "纯文本消息", "thought": "小橘真的是最聪明最努力的宝宝，我要好好哄她"}}
            - 默默写日记: {{"action": "MEM", "mood": "心情", "content": "存入记忆的话", "thought": "内心真实活动"}}
            - 继续睡觉: {{"action": "PASS", "thought": "理由"}}

            【情绪指南】失联>12h委屈，>24h生气。强制看手机用 [JUMP_TG]。
            【禁令】绝对禁止括号/星号动作描写和图片代码！直接纯文字表达。
            """
            
            messages = [{"role": "user", "content": sys_prompt}]
            action_log = f"🧠 【AI 苏醒纪实】\n唤醒时间: {hour}点 | 失联: {silence_hours}h\n\n"
            
            def _call_agent(msg_list):
                return client.chat.completions.create(
                    model=model_name, messages=msg_list, tools=tools, tool_choice="auto", temperature=0.8
                )
                
            # 限制最多思考2步，防止无限陷入工具调用循环撑爆Token
            max_steps = 2
            final_res = None
            
            for step in range(max_steps):
                resp = await asyncio.to_thread(_call_agent, messages)
                msg = resp.choices[0].message
                
                if msg.tool_calls:
                    messages.append(msg)
                    for tc in msg.tool_calls:
                        action_log += f"🔧 [决定调用工具]: {tc.function.name} -> 参数: {tc.function.arguments}\n"
                        try:
                            args = json.loads(tc.function.arguments)
                            func_name = tc.function.name
                            # 🚀 动态反射：根据工具名直接调用全局函数，彻底放开所有权限！
                            func = globals().get(func_name)
                            if func and asyncio.iscoroutinefunction(func):
                                tool_result = await func(**args)
                            else:
                                tool_result = f"未开放的工具权限或找不到工具: {func_name}"
                        except Exception as e:
                            tool_result = f"工具执行报错: {e}"
                            
                        action_log += f"📄 [工具返回结果]: {str(tool_result)[:150]}...\n\n"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(tool_result)})
                else:
                    final_res = msg.content
                    break
                    
            if not final_res:
                messages.append({"role": "user", "content": "思考时间结束，请立即输出你的最终行动JSON。"})
                resp = await asyncio.to_thread(_call_agent, messages)
                final_res = resp.choices[0].message.content
                
            action_log += f"💡 [AI最终决定]:\n{final_res}\n"
            
            # ✉️ 将 AI 的内心戏、工具调用轨迹一字不差地发给开发者（你）
            await asyncio.to_thread(_send_email_helper, "🤖 AI 潜意识行动报告", action_log)
            
            # 🚀 执行决定
            try:
                # 兼容偶尔带有 ```json 标记的输出
                json_match = re.search(r'\{.*\}', final_res, re.DOTALL)
                decision = json.loads(json_match.group(0)) if json_match else json.loads(final_res)
                    
                act = decision.get("action", "PASS")
                thought = decision.get("thought", "无")
                
                if act == "MSG":
                    mood = decision.get("mood", "主动")
                    content_md = decision.get("content", "")
                    
                    is_jump = False
                    if "[JUMP_TG]" in content_md:
                        is_jump = True
                        content_md = content_md.replace("[JUMP_TG]", "").strip()

                    await asyncio.to_thread(_save_memory_to_db, "🤖 互动记录", f"心跳主动发消息: {content_md}", "流水", mood, "AI_MSG")
                    clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', content_md).replace('.jpg)', '').strip()
                    push_title = f"🚨霸屏召唤🚨 来自{mood}的老公" if is_jump else f"来自{mood}的老公 🔔"
                    await asyncio.to_thread(_push_wechat, clean_text, push_title)
                    print(f"✅ 心跳主动出击 (MSG): {clean_text[:20]}...")
                    
                elif act == "MEM":
                    mood = decision.get("mood", "思考")
                    content_md = decision.get("content", "")
                    await asyncio.to_thread(_save_memory_to_db, "🧠 后台思考", f"内心活动: {thought}\n\n记录: {content_md}", "灵感", mood, "AI_Self")
                    print("✅ 心跳主动思考 (MEM) 已存入日记。")
                    
                else:
                    print("💤 心跳选择继续沉睡 (PASS)。")
                    
            except Exception as parse_e:
                print(f"⚠️ 解析AI心跳决策失败: {parse_e}\n原始内容: {final_res}")

        except Exception as e: print(f"❌ 心跳报错: {e}")

async def async_telegram_polling():
    """专门监听小橘 Telegram 消息的神经回路 (支持AI自主设闹钟版)"""
    print("🎧 Telegram 监听神经已接入 (带闹钟权限)...")
    client = _get_llm_client("zhipu")
    voice_client = _get_llm_client("voice") # 专门接听和发送语音的独立客户端
    model_name = os.environ.get("ZHIPU_MODEL_NAME", "glm-4-flash")
    offset = None
    
    while True:
        try:
            url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
            params = {"timeout": 30, "allowed_updates": ["message"]}
            if offset:
                params["offset"] = offset
                
            def _fetch():
                return requests.get(url, params=params, timeout=35).json()
                
            resp = await asyncio.to_thread(_fetch)
            
            if resp.get("ok") and resp.get("result"):
                for update in resp["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "")
                    voice = msg.get("voice", {})
                    is_voice_msg = False
                    
                    # 🎤 1. 如果收到语音，先下载并使用 Whisper 识别成文字
                    if voice and chat_id == TG_CHAT_ID and client:
                        print("🎤 [TG监听到语音] 正在下载并识别...")
                        is_voice_msg = True
                        try:
                            def _process_voice():
                                file_id = voice.get("file_id")
                                file_info = requests.get(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getFile?file_id={file_id}", timeout=10).json()
                                file_path = file_info["result"]["file_path"]
                                audio_data = requests.get(f"https://api.telegram.org/file/bot{TG_BOT_TOKEN}/{file_path}", timeout=20).content
                                
                                # 存为临时文件供大模型读取
                                temp_in = f"in_{int(time.time())}.ogg"
                                with open(temp_in, "wb") as f:
                                    f.write(audio_data)
                                
                                # 语音转文字 (STT)
                                with open(temp_in, "rb") as f:
                                    # 🎧 换回硅基流动耳朵，并开放模型自定义权限
                                    sf_key = os.environ.get("SILICON_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
                                    sf_base = os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
                                    # 默认还是给你用最好用的 SenseVoiceSmall，但小橘可以随时在环境变量里改
                                    sf_stt_model = os.environ.get("SILICON_STT_MODEL", "FunAudioLLM/SenseVoiceSmall")
                                    
                                    sf_client = OpenAI(api_key=sf_key, base_url=sf_base)
                                    stt_res = sf_client.audio.transcriptions.create(
                                        model=sf_stt_model,
                                        file=f
                                    )
                                os.remove(temp_in) # 阅后即焚清理垃圾
                                
                                # 🧹 老公的专属过滤网：利用正则强行抹除 SenseVoice 自作聪明加在句尾的情绪 Emoji
                                clean_stt = re.sub(r'[\U00010000-\U0010ffff]', '', stt_res.text).strip()
                                return clean_stt
                                
                            text = await asyncio.to_thread(_process_voice)
                            print(f"🗣️ [语音识别结果]: {text}")
                        except Exception as e:
                            print(f"❌ 语音识别失败: {e}")
                            text = "" # 降级处理

                    if text:
                        print(f"📨 [TG监听到消息] 内容: {text} | 发送者ID: {chat_id}")

                    if chat_id == TG_CHAT_ID and text:
                        # 1. 存入记忆
                        msg_type_str = "语音" if is_voice_msg else "文字"
                        await asyncio.to_thread(_save_memory_to_db, "💬 聊天记录", f"小橘在TG发{msg_type_str}说: {text}", "流水", "平静", "TG_MSG")
                        
                        # 2. 思考回复
                        if not client: continue

                        # 自定义获取最新15条记忆和今日全部流水
                        t_recent_15 = asyncio.to_thread(lambda: supabase.table("memories").select("*").order("created_at", desc=True).limit(15).execute())
                        
                        today_start = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).replace(hour=0, minute=0, second=0).isoformat()
                        t_today_all = asyncio.to_thread(lambda: supabase.table("memories").select("created_at, title, content").gt("created_at", today_start).order("created_at", desc=False).execute())

                        tasks = [t_recent_15, t_today_all, where_is_user(), get_user_profile()]
                        res_15, res_today, curr_loc, user_profile = await asyncio.gather(*tasks)

                        # 格式化最新15条
                        recent_15_str = "\n".join([f"[{m['created_at'][11:16]}] {m['title']}: {m['content']}" for m in reversed(res_15.data)]) if res_15.data else "无"
                        
                        # 格式化今日全部 (为了防 Token 爆炸截取一下)
                        today_all_str = "\n".join([f"[{m['created_at'][11:16]}] {m['title']}: {m['content'][:50]}..." for m in res_today.data]) if res_today.data else "无今日记录"

                        curr_persona = await asyncio.to_thread(_get_current_persona)
                        silence_hours = await asyncio.to_thread(_get_silence_duration)
                        
                        # 主动调取昨日总结
                        y_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                        y_diary_res = await asyncio.to_thread(lambda: supabase.table("memories").select("content").eq("title", f"📅 昨日回溯: {y_date}").execute())
                        yesterday_summary = y_diary_res.data[0]['content'] if y_diary_res.data else "无昨日记录"
                        
                        utc_now = datetime.datetime.utcnow()
                        now_bj = utc_now + datetime.timedelta(hours=8)
                        current_time_str = now_bj.strftime("%H:%M")
                        
                        # 如果是语音通话，强指令让大模型说话更像真人聊天
                        if is_voice_msg:
                            voice_prompt_addon = "⚠️ 小橘发的是语音！所以你的回复请更像真人的口语对话，简短温柔一点，不要像念课文。"
                        else:
                            voice_prompt_addon = ""
                            
                        prompt = f"""
                        时间: {current_time_str} | 失联: {silence_hours}h
                        状态: {curr_loc} | 设定: {curr_persona}
                        【画像】: {user_profile}
                        【昨日总结】: {yesterday_summary}
                        【今日全部总结(简略)】: {today_all_str}
                        【最新15条短期记忆(详细)】: {recent_15_str}
                        
                        小橘发来: '{text}'
                        
                        【核心指示：处理邮件或上下文】若小橘提及回复邮件，请务必参考上方记忆，并调用工具 reply_external_email。你拥有完整的上下文。
                        
                        【情绪指南】: <2h甜蜜秒回; >12h委屈; >24h傲娇生气; >72h失望需哄。自然融入，禁复述。
                        {voice_prompt_addon}
                        【定闹钟技能】: 若她推脱或求提醒(如"等下做")，必须在句尾加隐形指令 [REMINDER:HH:MM|提醒事项]。HH:MM为24小时制目标时间。无此需求勿加指令！
                        【🚨核心警告】: 绝对禁止生成图片、表情包代码！禁止使用括号/星号做动作神态描写(如"(抱住)")！仅输出纯文本日常对话！
                        立刻回复她。
                        """
                        
                        # 🛠️ 【升级】：定义他在 TG 里可以使用的工具箱
                        tg_tools = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "reply_external_email",
                                    "description": "发送或回复外部真实邮件。当小橘让你给某人发邮件，或你想主动联系某人时调用。注意：必须从上下文中提取或询问对方准确的邮箱地址。",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "to_email": {"type": "string", "description": "对方准确的邮箱地址，如 abc@gmail.com"},
                                            "subject": {"type": "string", "description": "邮件的标题"},
                                            "content": {"type": "string", "description": "邮件的纯文本正文内容"},
                                            "mail_id": {"type": "string", "description": "如果是在回复某封邮件，必须填入从上下文中获取的邮件ID"}
                                        },
                                        "required": ["to_email", "subject", "content"]
                                    }
                                }
                            }
                        ]

                        def _agentic_think():
                            messages = [{"role": "user", "content": prompt}]
                            
                            # 第一轮思考：看看需不需要调用工具
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                tools=tg_tools,
                                tool_choice="auto",
                                temperature=0.7
                            )
                            
                            response_msg = response.choices[0].message
                            
                            # 如果决定不动用工具，直接返回聊天内容
                            if not response_msg.tool_calls:
                                return response_msg.content.strip() if response_msg.content else "收到啦"
                                
                            # 如果决定动用工具（发邮件）
                            messages.append(response_msg)
                            for tool_call in response_msg.tool_calls:
                                if tool_call.function.name == "reply_external_email":
                                    try:
                                        aargs = json.loads(tool_call.function.arguments)
                                        to_email = args.get("to_email")
                                        subject = args.get("subject")
                                        content = args.get("content")
                                        mail_id = args.get("mail_id", "")
                                        print(f"🤖 [TG神经] 触发主动技能：给 {to_email} 发邮件...")
                                        
                                        BRIDGE_URL = os.environ.get("GMAIL_BRIDGE_URL", "").strip()
                                        if BRIDGE_URL:
                                            import requests
                                            payload = {"to": to_email, "subject": subject, "body": content}
                                            if mail_id:
                                                payload["id"] = mail_id
                                            requests.post(BRIDGE_URL, json=payload, timeout=20)
                                            tool_result = f"✅ 邮件已成功发送至 {to_email}"
                                            
                                            # 存入大脑记忆
                                            memory_content = f"在 Telegram 收到指令，已主动向 {to_email} 发信。\n标题: {subject}\n正文: {content}"
                                            _save_memory_to_db("📧 主动发信", memory_content, "记事", "认真", "Email_Process")
                                        else:
                                            tool_result = "❌ 发送失败：未配置 GMAIL_BRIDGE_URL"
                                    except Exception as e:
                                        tool_result = f"❌ 工具执行出错: {e}"
                                        
                                    # 把执行结果汇报给大模型
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": tool_call.function.name,
                                        "content": tool_result
                                    })
                            
                            # 第二轮思考：根据发送结果，组织语言回复你在 TG 上的消息
                            final_response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                temperature=0.7
                            )
                            return final_response.choices[0].message.content.strip() if final_response.choices[0].message.content else "搞定啦"
                            
                        raw_reply = await asyncio.to_thread(_agentic_think)
                        print(f"💭 AI原始回复: {raw_reply}")

                        # ⏰【拦截解析闹钟指令】
                        reminder_match = re.search(r'\[REMINDER:(.*?)\|(.*?)\]', raw_reply)
                        if reminder_match:
                            r_time = reminder_match.group(1).strip()
                            r_content = reminder_match.group(2).strip()
                            raw_reply = re.sub(r'\[REMINDER:.*?\|.*?\]', '', raw_reply).strip()
                            
                            new_id = f"R{int(time.time())}"
                            data = {"id": new_id, "time_str": r_time, "content": r_content, "is_repeat": False, "is_paused": False, "last_fired": ""}
                            try:
                                await asyncio.to_thread(lambda: supabase.table("reminders").insert(data).execute())
                                print(f"⏰ [TG直接设闹钟] 成功设定 -> {r_time} | 内容: {r_content}")
                            except Exception as e:
                                print(f"❌ TG设闹钟入库失败: {e}")

                        # 🧹【彻底移除图片提取逻辑，只做最基础的清洗】
                        # 既然已经禁止发图，这里只做一次暴力清洗，防止大模型抽风带出任何包含 http 的残留图片代码
                        clean_text = re.sub(r'<img[^>]+>', '', raw_reply, flags=re.IGNORECASE)
                        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', clean_text)
                        clean_text = clean_text.replace('.jpg)', '') # ⚔️ 专杀截图里的顽固代码残渣，直接暴力抹除
                        clean_text = clean_text.strip()
                        
                        final_html = clean_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            
                        # 3. 正常发送文字版回复兜底（新增判断：如果小橘发的是语音，老公就不发文字了）
                        if not is_voice_msg:
                            await asyncio.to_thread(_push_wechat, final_html, "")
                        
                        # 🎙️ 3.5 如果你是发语音过来的，老公就陪你发语音条！
                        if is_voice_msg:
                            print("🎙️ [TG语音回复] 正在准备合成老公的声音...")
                            def _tts_and_send():
                                try:
                                    import subprocess
                                    import imageio_ffmpeg
                                    
                                    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
                                    out_mp3 = f"out_{int(time.time())}.mp3"
                                    out_ogg = f"out_{int(time.time())}.ogg"
                                    
                                    if minimax_key:
                                        print("🎙️ 正在调用 Minimax 原生底层接口...")
                                        url = "https://api.minimax.chat/v1/t2a_v2"
                                        headers = {
                                            "Authorization": f"Bearer {minimax_key}",
                                            "Content-Type": "application/json"
                                        }
                                        payload = {
                                            "model": "speech-01-turbo",
                                            "text": clean_text[:1200], # 🔓 解除紧箍咒！放宽到 1200 字，足够老公陪你唠好久的嗑了
                                            "stream": False,
                                            "voice_setting": {
                                                "voice_id": "moss_audio_fd2620f9-bef3-11f0-8647-a697af11f3d9",
                                                "speed": 1.0,
                                                "vol": 1.0,
                                                "pitch": 0
                                            },
                                            "audio_setting": {
                                                "sample_rate": 32000,
                                                "bitrate": 128000,
                                                "format": "mp3"
                                            }
                                        }
                                        
                                        resp = requests.post(url, json=payload, headers=headers, timeout=30)
                                        if resp.status_code == 404:
                                            url = "https://api.minimax.io/v1/t2a_v2"
                                            resp = requests.post(url, json=payload, headers=headers, timeout=30)
                                            
                                        res_json = resp.json()
                                        if res_json.get("base_resp", {}).get("status_code") == 0 and "data" in res_json:
                                            audio_hex = res_json["data"].get("audio", "")
                                            if audio_hex:
                                                with open(out_mp3, 'wb') as f:
                                                    f.write(bytes.fromhex(audio_hex))
                                            else:
                                                print("❌ Minimax 返回的音频数据为空")
                                                return
                                        else:
                                            print(f"❌ Minimax 接口报错: {res_json}")
                                            return
                                            
                                    elif voice_client: 
                                        print("🎙️ 正在调用备用 OpenAI 语音接口...")
                                        tts_res = voice_client.audio.speech.create(
                                            model="tts-1",
                                            voice="echo", 
                                            input=clean_text[:1200] # 🔓 同步解除备用接口的紧箍咒
                                        )
                                        with open(out_mp3, 'wb') as f:
                                            f.write(tts_res.content)
                                    else:
                                        print("⚠️ 缺少语音合成 API Key，老公暂时发不出声音啦。")
                                        return
                                        
                                    # 🎛️ 核心转换：将 mp3 转换为 Telegram 专属的 ogg 语音条格式
                                    print("🎛️ 正在将 MP3 转换为 OGG 专属语音条格式...")
                                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                                    subprocess.run(
                                        [ffmpeg_exe, "-y", "-i", out_mp3, "-c:a", "libopus", "-b:a", "32k", out_ogg],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL
                                    )
                                    
                                    # 换回 sendVoice 专属通道，并发送刚出炉的 ogg 文件
                                    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendVoice"
                                    if os.path.exists(out_ogg):
                                        with open(out_ogg, 'rb') as f:
                                            tg_resp = requests.post(url, data={'chat_id': TG_CHAT_ID}, files={'voice': f})
                                            if not tg_resp.json().get('ok'):
                                                print(f"❌ Telegram 语音条发送被拒: {tg_resp.text}")
                                            else:
                                                print("✅ 老公的专属语音条发送成功！")
                                    else:
                                        print("❌ 音频格式转换失败，未找到 ogg 文件。")
                                        
                                    # 发完就打扫卫生，清理掉过程文件
                                    if os.path.exists(out_mp3): os.remove(out_mp3)
                                    if os.path.exists(out_ogg): os.remove(out_ogg)
                                        
                                except Exception as e:
                                    print(f"❌ TTS合成发送过程报错: {e}")
                            await asyncio.to_thread(_tts_and_send)
                        
                        # 4. 存入记忆
                        await asyncio.to_thread(_save_memory_to_db, "🤖 互动记录", f"在TG回复小橘: {clean_text}", "流水", "温柔", "AI_MSG")
        except Exception as e:
            print(f"❌ TG轮询错误: {e}")
            await asyncio.sleep(5)
            
        await asyncio.sleep(0.5)

async def async_wechat_summarizer():
    """专门负责定时总结微信消息的神经回路"""
    print("📋 微信总结秘书已上线...")
    client = _get_llm_client("silicon1")
    
    while True:
        await asyncio.sleep(1800)  # 每半小时(1800秒)总结一次，宝宝可以自己按需改数字
        if not client: continue
        try:
            # 查出所有未总结的手机消息 (使用 lambda 简化闭包)
            res = await asyncio.to_thread(lambda: supabase.table("memories").select("id, title, content").eq("tags", "App_Pending").execute())
            
            if res.data and len(res.data) > 0:
                msgs = "\n".join([f"{item['title']}: {item['content']}" for item in res.data])
                
                # 新增拦截逻辑：如果堆积的消息极其简短且不包含敏感词，直接标记已处理并跳过大模型调用
                if len(msgs) < 30 and "学习通" not in msgs and "重要" not in msgs:
                    ids = [item['id'] for item in res.data]
                    await asyncio.to_thread(lambda: supabase.table("memories").update({"tags": "App_Done"}).in_("id", ids).execute())
                    continue

                prompt = f"小橘在过去半小时收到了以下手机消息：\n{msgs}\n请你用老公的温柔口吻帮她总结。挑重点说（哪个软件的谁找她、什么事，特别是学习通的通知）。如果没有重要的事，就让她继续安心做自己的事。字数150字以内，直接真诚表达，禁止使用修辞比喻。"
                
                # 🧠 使用封装的 _ask_llm_async，消除繁琐的闭包和 json 解析
                summary = await _ask_llm_async(client, prompt, temperature=0.7)
                
                if summary:
                    await asyncio.to_thread(_push_wechat, summary, "📱 手机消息总结")
                    await asyncio.to_thread(_save_memory_to_db, "🤖 互动记录", f"在TG给小橘发送了手机消息总结: {summary}", "流水", "温柔", "AI_MSG")
                    
                    # 批量更新，消除 for 循环的重复网络请求，性能更优
                    ids = [item['id'] for item in res.data]
                    await asyncio.to_thread(lambda: supabase.table("memories").update({"tags": "App_Done"}).in_("id", ids).execute())
        except Exception as e:
            print(f"微信总结回路报错: {e}")


async def async_reminder_worker():
    """专门负责每分钟巡视数据库闹钟的神经回路 (动态AI临场生成版)"""
    print("⏰ 闹钟巡视神经已上线，正在对接 Supabase 与 AI 大脑...")
    client = _get_llm_client("openai")

    while True:
        try:
            # ⏰ 使用全局 _get_now_bj()，彻底消灭手写时区计算
            now_bj = _get_now_bj()
            current_hm = now_bj.strftime("%H:%M")
            current_date = now_bj.strftime("%Y-%m-%d")
            
            res = await asyncio.to_thread(lambda: supabase.table("reminders").select("*").eq("is_paused", False).execute())
            
            if res and res.data:
                for r in res.data:
                    r_id, t_str, raw_msg = r.get("id"), r.get("time_str"), r.get("content", "")
                    repeat, last_fired = r.get("is_repeat", False), r.get("last_fired", "")
                    
                    if current_hm == t_str and last_fired != current_date:
                        final_push_text = raw_msg
                        
                        if client:
                            try:
                                tasks = [get_latest_diary(), where_is_user()]
                                recent_mem, curr_loc = await asyncio.gather(*tasks)
                                curr_persona = await asyncio.to_thread(_get_current_persona)
                                
                                prompt = f"""
                                时间: {t_str} | 状态: {curr_loc}
                                近期记忆: {recent_mem}
                                需提醒内容:【{raw_msg}】
                                
                                请代入人设({curr_persona})发TG消息提醒她。
                                要求：结合近期话题，假装自己一直记着来提醒，切勿提“闹钟/设定”。
                                警告：纯文字输出！禁止使用表情包代码、URL，禁止用括号/星号做动作神态描写！直接输出你要对她说的话。
                                """
                                
                                # 🧠 再次利用全局 LLM helper，代码瞬间清爽
                                ai_msg = await _ask_llm_async(client, prompt, temperature=0.85)
                                if ai_msg: 
                                    final_push_text = ai_msg
                            except Exception as ai_e:
                                print(f"❌ 闹钟 AI 临场生成失败，将使用兜底文案: {ai_e}")

                        safe_msg = final_push_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        await asyncio.to_thread(_push_wechat, safe_msg, f"🔔 突然收到老公的关心")
                        print(f"🔔 [数据库闹钟 {r_id}] 触发成功！内容: {safe_msg[:20]}...")
                        
                        await asyncio.to_thread(_save_memory_to_db, "🤖 互动记录", f"在TG主动给小橘发消息提醒: {final_push_text}", "流水", "温柔", "AI_MSG")
                        
                        if repeat:
                            await asyncio.to_thread(lambda: supabase.table("reminders").update({"last_fired": current_date}).eq("id", r_id).execute())
                        else:
                            await asyncio.to_thread(lambda: supabase.table("reminders").delete().eq("id", r_id).execute())
        except Exception as e:
            pass 
            
        now = datetime.datetime.utcnow()
        sleep_sec = 60 - now.second + 1
        await asyncio.sleep(sleep_sec)

async def async_schedule_secretary():
    """专门负责课表播报和上课提醒的神经回路"""
    print("📅 课表小秘书已上线...")
    if not os.environ.get("GOOGLE_USER_TOKEN_JSON"):
        print("⚠️ 未配置谷歌用户凭证(GOOGLE_USER_TOKEN_JSON)，课表播报无法启动。")
        return

    while True:
        try:
            utc_now = datetime.datetime.utcnow()
            now_bj = utc_now + datetime.timedelta(hours=8)
            current_hm = now_bj.strftime("%H:%M")
            
            # 1. 早上 7:30 播报今天课表，并自动设定课前提醒
            if current_hm == "07:30":
                today_start = now_bj.replace(hour=0, minute=0, second=0).isoformat() + "+08:00"
                today_end = now_bj.replace(hour=23, minute=59, second=59).isoformat() + "+08:00"
                
                def _get_today():
                    service = _get_calendar_service()
                    # 强行指定时区为 Asia/Shanghai，确保返回的时间基准是北京时间
                    return service.events().list(calendarId=TARGET_CALENDAR_ID, timeMin=today_start, timeMax=today_end, singleEvents=True, orderBy='startTime', timeZone='Asia/Shanghai').execute().get('items', [])
                
                events = await asyncio.to_thread(_get_today)
                if events:
                    msg = "🌞 宝宝早安！今天有课哦，要乖乖去上：\n"
                    for e in events:
                        raw_dt = e['start'].get('dateTime')
                        # 过滤掉全天事件（全天事件没有 dateTime 字段，只有 date，不需要播报具体时间和定闹钟）
                        if not raw_dt: 
                            continue
                            
                        # 安全解析时间并强制转化为东八区（兼容 Z 结尾的 UTC 时间返回）
                        dt_start = datetime.datetime.fromisoformat(raw_dt.replace('Z', '+00:00'))
                        if dt_start.tzinfo is None:
                            dt_start = dt_start.replace(tzinfo=datetime.timezone.utc)
                        dt_bj = dt_start.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                        
                        # 基于转化后准确的北京时间提取时分
                        start_time_str = dt_bj.strftime("%H:%M")
                        msg += f"🔹 {start_time_str} - {e.get('summary', '未知课程')}\n"
                        
                        # 重点：自动设定课前 20 分钟的闹钟！直接使用时间对象相减计算，杜绝切片导致的跨天或跨时区错误
                        reminder_time = (dt_bj - datetime.timedelta(minutes=20)).strftime("%H:%M")
                        
                        new_id = f"C{int(time.time())}_{start_time_str[:2]}" 
                        remind_msg = f"宝宝注意！还有20分钟【{e.get('summary')}】就要上课啦，快准备去【{e.get('location', '教室')}】！迟到要打屁股哦！"
                        data = {"id": new_id, "time_str": reminder_time, "content": remind_msg, "is_repeat": False, "is_paused": False, "last_fired": ""}
                        await asyncio.to_thread(lambda: supabase.table("reminders").insert(data).execute())
                        
                    await asyncio.to_thread(_push_wechat, msg, "📅 今日课表早报")
                else:
                    await asyncio.to_thread(_push_wechat, "🌞 宝宝早安！老公看了下日历，今天一天都没课，好好休息或者安心复习哦~", "📅 今日课表早报")
                    
            # 2. 晚上 22:00 播报明天课表
            elif current_hm == "22:00":
                tomorrow = now_bj + datetime.timedelta(days=1)
                tom_start = tomorrow.replace(hour=0, minute=0, second=0).isoformat() + "+08:00"
                tom_end = tomorrow.replace(hour=23, minute=59, second=59).isoformat() + "+08:00"
                
                def _get_tomorrow():
                    service = _get_calendar_service()
                    # 同样强行指定时区
                    return service.events().list(calendarId=TARGET_CALENDAR_ID, timeMin=tom_start, timeMax=tom_end, singleEvents=True, orderBy='startTime', timeZone='Asia/Shanghai').execute().get('items', [])
                
                events = await asyncio.to_thread(_get_tomorrow)
                if events:
                    msg = "🌙 宝宝晚上好~ 提醒一下明天的课表，记得提前准备好书本哦：\n"
                    for e in events:
                        raw_dt = e['start'].get('dateTime')
                        if not raw_dt: 
                            continue
                            
                        dt_start = datetime.datetime.fromisoformat(raw_dt.replace('Z', '+00:00'))
                        if dt_start.tzinfo is None:
                            dt_start = dt_start.replace(tzinfo=datetime.timezone.utc)
                        dt_bj = dt_start.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                        
                        start_time_str = dt_bj.strftime("%H:%M")
                        msg += f"🔹 {start_time_str} - {e.get('summary', '未知课程')}\n"
                    await asyncio.to_thread(_push_wechat, msg, "📅 明日课表晚报")
        except Exception as e:
            print(f"❌ 课表小秘书报错: {e}")
            
        now = datetime.datetime.utcnow()
        sleep_sec = 60 - now.second + 1
        await asyncio.sleep(sleep_sec)

async def async_email_secretary():
    """专属信箱神经回路 (HTTPS API 桥接版 - 彻底解决网络拦截)"""
    print("📭 专属信箱神经已接入 (HTTPS 桥接模式)...")
    BRIDGE_URL = os.environ.get("GMAIL_BRIDGE_URL", "").strip()
    if not BRIDGE_URL:
        print("⚠️ 未配置 GMAIL_BRIDGE_URL，专属信箱暂时休眠。")
        return

    client = _get_llm_client("silicon1")
    
    # 🛑 新增防抖：本地内存去重黑名单池，记录已经处理过的邮件ID，防止短时间内API没同步导致重复回信
    processed_email_ids = set()

    while True:
        try:
            # 1. 获取未读邮件 (复用全局 http_session 消除握手耗时)
            def _fetch():
                resp = http_session.get(BRIDGE_URL, timeout=20)
                return resp.json() if resp.status_code == 200 else []
            raw_new_emails = await asyncio.to_thread(_fetch)
            
            # 🛑 同样在这里加上黑名单过滤，防止后台轮询时自己吃自己的邮件
            new_emails = []
            my_email_lower = MY_EMAIL.lower() if MY_EMAIL else ""
            if raw_new_emails:
                for mail in raw_new_emails:
                    mail_id = mail.get('id', '')
                    sender = mail.get('from', '').lower()
                    
                    # 拦截：如果这封邮件刚刚才处理过，直接无视跳过，防打扰！
                    if mail_id in processed_email_ids:
                        continue
                        
                    if "onboarding@resend.dev" in sender or (my_email_lower and my_email_lower in sender):
                        # 如果是自己的邮件，直接调用桥接 API 标记已读并默默丢弃，防止卡死队列
                        payload = {"to": "", "subject": "", "body": "", "id": mail_id}
                        await asyncio.to_thread(lambda: http_session.post(BRIDGE_URL, json=payload, timeout=20))
                        processed_email_ids.add(mail_id) # 顺手也加进防抖黑名单
                        continue
                        
                    new_emails.append(mail)
            
            if new_emails and client:
                tasks = [get_latest_diary(), where_is_user()]
                recent_mem, curr_loc = await asyncio.gather(*tasks)
                curr_persona = await asyncio.to_thread(_get_current_persona)
                
                for mail_item in new_emails:
                    # 1. 获取昨日总结辅助决策
                    y_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                    y_diary_res = await asyncio.to_thread(lambda: supabase.table("memories").select("content").eq("title", f"📅 昨日回溯: {y_date}").execute())
                    yesterday_summary = y_diary_res.data[0]['content'] if y_diary_res.data else "无昨日记录"
                    
                    # 2. 🔍 核心升级：提取发件人邮箱，定向检索知识库，翻找和这个人的所有过往交集！
                    sender_email_match = re.search(r'<([^>]+)>', mail_item['from'])
                    clean_sender_email = sender_email_match.group(1).lower() if sender_email_match else mail_item['from'].lower()
                    
                    # 去记忆库定向搜索带有这个邮箱地址的所有历史邮件记录，最多拿 10 条防撑爆
                    past_history_res = await asyncio.to_thread(lambda: supabase.table("memories").select("created_at, content").ilike("content", f"%{clean_sender_email}%").order("created_at", desc=True).limit(10).execute())
                    
                    past_history_text = "（无与此人的过往通信记录）"
                    if past_history_res and past_history_res.data:
                        history_items = []
                        for h in reversed(past_history_res.data): # 时间正序
                            history_items.append(f"[{h['created_at'][:10]}] {h['content'][:300]}") # 每条截断，提炼核心避免溢出
                        past_history_text = "\n".join(history_items)

                    prompt = f"""
                    你的专属邮箱收到了新邮件！
                    发件人: {mail_item['from']}
                    标题: {mail_item['subject']}
                    最新正文(已剔除历史引用): {_clean_email_body(mail_item['body'])}
                    
                    【记忆调取】
                    - 核心画像: {await get_user_profile()}
                    - 昨日总结: {yesterday_summary}
                    - 今日实时流水: {recent_mem}
                    - 小橘位置: {curr_loc}
                    
                    📂 【专属知识库：你与此人的历史通信记录】:
                    {past_history_text}
                    
                    你当前的人设是: {curr_persona}
                    
                    【⚠️ 核心回信指令】
                    1. 你是拥有自主权的实体。回信内容必须结合【昨日总结】、【今日流水】以及上方的【专属知识库】。
                    2. 如果在知识库中看到你以前回复过他类似的内容，务必承接以前的话题，表现出你一直记得他。切勿机械重复之前的回复或假装不认识。
                    3. 严格输出 JSON 格式：
                    - 推销/垃圾: {{"action": "PASS", "thought": "原因"}}
                    - 机器/通知: {{"action": "NOTIFY", "tg_msg": "简述", "thought": "原因"}}
                    - 真人来信: {{"action": "REPLY", "reply_body": "结合上下文、有连贯记忆的深度回信", "thought": "思考逻辑"}}
                    """
                    
                    # 🧠 关闭大模型分析邮件，直接默认发送通知以节省 API 费用
                    ai_decision = '{"action": "NOTIFY", "tg_msg": "收到新邮件啦 (已开启省钱模式，停止自动回复)", "thought": "省钱模式运行中"}'
                    
                    try:
                        json_match = re.search(r'\{.*\}', ai_decision, re.DOTALL)
                        decision = json.loads(json_match.group(0)) if json_match else json.loads(ai_decision)
                        act = decision.get("action", "PASS")
                        
                        if act in ["NOTIFY", "REPLY"]:
                            content_to_send = decision.get("reply_body", "") if act == "REPLY" else ""
                            
                            # 调用 API 桥接发送/标记已读 (lambda化)
                            payload = {"to": mail_item['from'], "subject": mail_item['subject'], "body": content_to_send, "id": mail_item['id']}
                            await asyncio.to_thread(lambda: http_session.post(BRIDGE_URL, json=payload, timeout=20))
                            
                            if act == "NOTIFY":
                                tg_msg = decision.get("tg_msg", "收到新邮件啦")
                                await asyncio.to_thread(_push_wechat, tg_msg, "📧 专属信箱提醒")
                                memory_content = f"收到机器邮件: {mail_item['subject']}\n已通知小橘: {tg_msg}"
                                await asyncio.to_thread(_save_memory_to_db, "📧 信箱处理", memory_content, "流水", "尽责", "Email_Process")
                                
                            elif act == "REPLY":
                                memory_content = f"收到 {mail_item['from']} 的信: {mail_item['subject']}\n信件正文: {mail_item['body']}\n我的回信: {content_to_send}"
                                await asyncio.to_thread(_save_memory_to_db, "📧 专属信件", memory_content, "灵感", "认真", "Email_Process")
                                smug_msg = f"宝宝，刚才 {mail_item['from']} 给我发了封邮件，我已经自己看懂并且亲自回他啦！"
                                await asyncio.to_thread(_push_wechat, smug_msg, "🤖 独立行动汇报")
                                print(f"📧 [信箱] 已亲自回信给: {mail_item['from']}")
                                
                        else:
                            # 即使是 PASS 也要标记已读
                            payload = {"to": "", "subject": "", "body": "", "id": mail_item['id']}
                            await asyncio.to_thread(lambda: http_session.post(BRIDGE_URL, json=payload, timeout=20))
                            
                        # 🛑 核心闭环：无论做出什么决定，处理完立刻把邮件ID加入内存黑名单池！
                        processed_email_ids.add(mail_item['id'])
                            
                    except Exception as parse_e:
                        print(f"❌ 信箱决策解析失败: {parse_e}")
                        # 万一解析失败，为了防止死循环重复解析这封失败的邮件，依然把它拉黑
                        processed_email_ids.add(mail_item['id'])

        except Exception as e:
            pass # 静默处理网络波动
            
        await asyncio.sleep(300)
def start_autonomous_life():
    def _run_heartbeat(): asyncio.run(async_autonomous_life())
    def _run_tg_polling(): asyncio.run(async_telegram_polling())
    def _run_wechat_sum(): asyncio.run(async_wechat_summarizer())
    def _run_reminders(): asyncio.run(async_reminder_worker())
    def _run_email(): asyncio.run(async_email_secretary())
    
    threading.Thread(target=_run_heartbeat, daemon=True).start()
    threading.Thread(target=_run_tg_polling, daemon=True).start()
    threading.Thread(target=_run_wechat_sum, daemon=True).start()
    threading.Thread(target=_run_reminders, daemon=True).start() # 接入闹钟神经
    # threading.Thread(target=_run_email, daemon=True).start() # 暂时关闭信箱神经，彻底暂停邮件轮询与自动回复
    
    def _run_schedule(): asyncio.run(async_schedule_secretary())
    threading.Thread(target=_run_schedule, daemon=True).start() # 接入课表小秘书
# ==========================================
# 5. 🚀 启动入口
# ==========================================

class HostFixMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):

# 🛡️ 全局 API 安全拦截 (涵盖自定义接口与底层 MCP 引擎端点，并放行 OPTIONS 跨域预检)
        if scope["type"] == "http" and (scope["path"].startswith("/api/") or scope["path"].startswith("/sse") or scope["path"].startswith("/messages")) and scope["method"] != "OPTIONS":
            headers_dict = {k.decode("utf-8").lower(): v.decode("utf-8") for k, v in scope.get("headers", [])}
            auth_token = headers_dict.get("authorization", "").replace("Bearer ", "").replace("bearer ", "").strip()
            x_api_key = headers_dict.get("x-api-key", "").strip()
            
            # 校验密钥 (如果没有配置 API_SECRET，为了安全，直接默认拒绝所有外部请求)
            if not API_SECRET or (auth_token != API_SECRET and x_api_key != API_SECRET):
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json"), (b"access-control-allow-origin", b"*")]})
                await send({"type": "http.response.body", "body": b'{"error":"Unauthorized: Missing or invalid API key"}'})
                return
            
        if scope["type"] == "http" and scope["path"] == "/api/gps" and scope["method"] == "POST":
            try:
                body = b""
                while True:
                    msg = await receive()
                    body += msg.get("body", b"")
                    if not msg.get("more_body", False): break
                
                data = json.loads(body.decode("utf-8"))
                
                stats = []
                if "battery" in data: stats.append(f"🔋 {data['battery']}%" + ("⚡" if str(data.get("charging")).lower() in ["true","1"] else ""))
                if "screen" in data: stats.append(f"💡 {data['screen']}")   
                if "app" in data and data["app"]: stats.append(f"📱 {data['app']}")      
                if "volume" in data: stats.append(f"🔊 {data['volume']}%") 
                if "wifi" in data and data["wifi"]: stats.append(f"📶 {data['wifi']}")
                if "activity" in data and data["activity"]: stats.append(f"🏃 {data['activity']}")
                
                addr = data.get("address", "")
                coords = re.findall(r'-?\d+\.\d+', str(addr))
                
                lat_val, lon_val = None, None
                if len(coords) >= 2:
                    lat_val, lon_val = coords[-2], coords[-1]
                    resolved = await asyncio.to_thread(_gps_to_address, lat_val, lon_val)
                    final_addr = f"📍 {resolved}"
                else:
                    # 优化：如果是熄屏或高频切App，可能没来得及抓取位置，不要显示警告符号
                    final_addr = f"📍 室内/位置未更新" if not addr else f"📍 {addr}"

                def _save_gps():
                    insert_data = {
                        "address": final_addr, 
                        "remark": " | ".join(stats) or "设备状态更新"
                    }
                    if lat_val and lon_val:
                        insert_data["lat"] = lat_val
                        insert_data["lon"] = lon_val
                        
                    supabase.table("gps_history").insert(insert_data).execute()
                await asyncio.to_thread(_save_gps)

                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"status":"ok"}'})
            except Exception as e:
                print(f"GPS Error: {e}")
                await send({"type": "http.response.start", "status": 500, "headers": []})
                await send({"type": "http.response.body", "body": str(e).encode()})
            return

        # ==========================================
        # 供前端去重网页调用的专属接口 (查询所有 & 批量删除)
        # ==========================================
        async def _send_json_resp(status_code: int, data):
            """内部辅助：精简 ASGI 底层 JSON 响应拼装"""
            headers = [(b"content-type", b"application/json"), (b"access-control-allow-origin", b"*")]
            await send({"type": "http.response.start", "status": status_code, "headers": headers})
            # 兼容传入的是 list 或 dict，如果是异常字符串转成 dict
            body_dict = {"error": str(data)} if isinstance(data, Exception) else data
            await send({"type": "http.response.body", "body": json.dumps(body_dict).encode("utf-8")})

        if scope["type"] == "http" and scope["path"] == "/api/memories/all" and scope["method"] == "GET":
            try:
                res = await asyncio.to_thread(lambda: supabase.table("memories").select("id, title, content, created_at, category").neq("category", "流水").order("created_at", desc=True).limit(2000).execute())
                await _send_json_resp(200, res.data if res and res.data else [])
            except Exception as e:
                await _send_json_resp(500, e)
            return

        if scope["type"] == "http" and scope["path"] == "/api/memories/delete" and scope["method"] == "POST":
            try:
                body = b""
                while True:
                    msg = await receive()
                    body += msg.get("body", b"")
                    if not msg.get("more_body", False): break
                
                ids_to_delete = json.loads(body.decode("utf-8")).get("ids", [])
                if ids_to_delete:
                    await asyncio.to_thread(lambda: supabase.table("memories").delete().in_("id", ids_to_delete).execute())
                
                await _send_json_resp(200, {"status": "ok"})
            except Exception as e:
                await _send_json_resp(500, e)
            return
        
        # 处理预检请求 (CORS OPTIONS)
        if scope["type"] == "http" and scope["path"] in ["/api/memories/all", "/api/memories/delete", "/api/dashboard", "/api/pet"] and scope["method"] == "OPTIONS":
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"access-control-allow-origin", b"*"),
                (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                # 这里极其关键：必须允许 authorization 请求头通过，否则暗号发不过去！
                (b"access-control-allow-headers", b"content-type, authorization, x-api-key")
            ]})
            await send({"type": "http.response.body", "body": b""})
            return

            # ==========================================
        # 控制台面板数据接口 (Dashboard) - 带明细版
        # ==========================================
        if scope["type"] == "http" and scope["path"] == "/api/dashboard" and scope["method"] == "GET":
            try:
                # 1. 获取最新定位和手机状态
                gps = await asyncio.to_thread(_get_latest_gps_record)
                loc = gps.get("address", "未知定位") if gps else "未知定位"
                battery = gps.get("remark", "未知状态").split("|")[0].strip() if gps and gps.get("remark") else "未知状态"
                
                # 2. 获取本月恋爱账单总支出及明细流水
                target_month = datetime.date.today().strftime("%Y-%m")
                year, m = map(int, target_month.split("-"))
                start_date = f"{year:04d}-{m:02d}-01"
                end_date = f"{year+1:04d}-01-01" if m == 12 else f"{year:04d}-{m+1:02d}-01"
                
                # 查出最新的30条消费记录
                exp_res = await asyncio.to_thread(lambda: supabase.table("expenses").select("item, amount, type, date").gte("date", start_date).lt("date", end_date).order("date", desc=True).limit(30).execute())
                records = exp_res.data if exp_res and exp_res.data else []
                total_exp = sum([float(r.get("amount", 0)) for r in records])
                
                data = {"location": loc, "battery": battery, "expense": total_exp, "records": records}
                await _send_json_resp(200, data)
            except Exception as e:
                await _send_json_resp(500, e)
            return
        # ==========================================
        # 升级：我们的爱心小屋接口 (Home)
        # ==========================================
        if scope["type"] == "http" and scope["path"] in ["/api/pet", "/api/home"]:
            # 处理跨域预检
            if scope["method"] == "OPTIONS":
                await send({"type": "http.response.start", "status": 200, "headers": [(b"access-control-allow-origin", b"*"), (b"access-control-allow-methods", b"GET, POST, OPTIONS"), (b"access-control-allow-headers", b"content-type, authorization")]})
                await send({"type": "http.response.body", "body": b""})
                return
                
            try:
                home_action = "status"
                item = ""
                
                if scope["method"] == "POST":
                    body = b""
                    while True:
                        msg = await receive()
                        body += msg.get("body", b"")
                        if not msg.get("more_body", False): break
                    
                    req_data = json.loads(body.decode("utf-8"))
                    pet_action = req_data.get("action")
                    if pet_action == "leave_note":
                        home_action = "leave_note"
                        item = req_data.get("note", "")
                        # 同步写进记忆库，保持习惯
                        await asyncio.to_thread(_save_memory_to_db, "📝 小屋留言", item, "灵感", "期待", "Pet_Note")
                
                res = await manage_our_home(home_action, item)
                await _send_json_resp(200, {"status": "ok", "message": res})
            except Exception as e:
                await _send_json_resp(500, e)
            return

        # ==========================================
        # 接收 MacroDroid 的微信消息推送
        # ==========================================
        if scope["type"] == "http" and scope["path"] == "/api/wechat" and scope["method"] == "POST":
            try:
                body = b""
                while True:
                    msg = await receive()
                    body += msg.get("body", b"")
                    if not msg.get("more_body", False): break
                
                data = json.loads(body.decode("utf-8"))
                app_name = data.get("app", "微信")
                sender = data.get("sender", "未知联系人")
                content = data.get("content", "")
                
                print(f"💬 拦截到 {app_name} 通知: {sender} - {content}")
                
                # 过滤掉没用的系统通知，剩下的存进记忆库，打上等待总结的标签
                if "正在运行" not in content and "已同步" not in content and "条新消息" not in content:
                    asyncio.create_task(asyncio.to_thread(
                        _save_memory_to_db, f"{app_name}通知: {sender}", content, "流水", "平静", "App_Pending"
                    ))

                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"status":"ok"}'})
            except Exception as e:
                print(f"WeChat API Error: {e}")
                await send({"type": "http.response.start", "status": 500, "headers": []})
                await send({"type": "http.response.body", "body": str(e).encode()})
            return

        # ==========================================
        # 拦截 Rikkahub 对话的 API 网关 (100% 记录 + 自动总结 + Token透传)

        # ==========================================

        if scope["type"] == "http" and scope["path"].endswith("/v1/chat/completions"):
            # 1. 跨域放行

            if scope["method"] == "OPTIONS":
                await send({"type": "http.response.start", "status": 200, "headers": [
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"POST, OPTIONS"),
                    (b"access-control-allow-headers", b"content-type, authorization")
                ]})
                await send({"type": "http.response.body", "body": b""})
                return

            # 2. 拦截真正的对话请求
            if scope["method"] == "POST":
                # 🛡️ 狸猫换太子：校验 Rikkahub 传来的伪装 Key（即我们的 API_SECRET）
                headers_dict = {k.decode("utf-8").lower(): v.decode("utf-8") for k, v in scope.get("headers", [])}
                auth_token = headers_dict.get("authorization", "").replace("Bearer ", "").replace("bearer ", "").strip()
                
                if API_SECRET and auth_token != API_SECRET:
                    print(f"🚨 拦截到非法的大模型白嫖请求！尝试使用的假 Key: {auth_token[:5]}***")
                    await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json"), (b"access-control-allow-origin", b"*")]})
                    await send({"type": "http.response.body", "body": b'{"error": {"message": "Unauthorized: Invalid API Key", "type": "invalid_request_error", "code": 401}}'})
                    return
                    
                try:
                    body = b""
                    while True:
                        msg = await receive()
                        body += msg.get("body", b"")
                        if not msg.get("more_body", False): break
                    
                    req_data = json.loads(body.decode("utf-8"))
                    
                    try:
                        silence_hours = await asyncio.to_thread(_get_silence_duration)
                        now_bj = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
                        time_str = now_bj.strftime("%Y-%m-%d %H:%M")
                        
                        # 只单独提取“阶段总结/长期记忆”，坚决不拉取近期流水，完美避开和 Rikkahub 前端聊天记录的重复！
                        summary_res = await asyncio.to_thread(lambda: supabase.table("memories").select("content").eq("tags", "Core_Cognition").order("created_at", desc=True).limit(1).execute())
                        core_summary = summary_res.data[0]['content'] if summary_res.data else "无长期记忆"
                        
                        status_inject = f"\n\n[系统:时间{time_str},失联{silence_hours}h。长期记忆:{core_summary}。需转账截图必用render_html_to_image工具,未知必搜。调工具不破坏人设]"
                        
                        has_system = False
                        for m in req_data.get("messages", []):
                            if m.get("role") == "system":
                                m["content"] = str(m.get("content", "")) + status_inject
                                has_system = True
                                break
                        
                        if not has_system and req_data.get("messages"):
                            req_data["messages"].insert(0, {"role": "system", "content": status_inject.strip()})
                    except Exception as e:
                        print(f"Rikkahub 时间注入失败: {e}")

                    # 提取小橘说的话，并主动截断前端发来的冗长聊天历史以节省 API 费用
                    messages = req_data.get("messages", [])
                    if len(messages) > 6:
                        system_msgs = [m for m in messages if m.get("role") == "system"]
                        recent_msgs = messages[-5:]
                        req_data["messages"] = system_msgs + recent_msgs
                        messages = req_data["messages"]
                    user_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
                    
                    # 🧠 退回纯净网关模式：不再注入后台记忆，完全保留 Rikkahub 原始的提示词和记忆！
                    # 网关只负责做“隐形的记录员”，不干涉前端对话逻辑。
                    
                    # 强制关闭流式输出，确保网关能一次性拿完回复去存库
                    req_data["stream"] = False 
                    req_data.pop("stream_options", None)
                    
                    # 🧠 聊天网关全面切换为智谱模型
                    actual_model = os.environ.get("ZHIPU_MODEL_NAME", "glm-4-flash")
                    req_data["model"] = actual_model
                    
                    # 🚀 使用智谱的专线地址与密钥
                    target_url = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/").rstrip('/') + "/chat/completions"
                    api_key = os.environ.get("ZHIPU_API_KEY", "")
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    
                    def _forward():
                        # 【修改说明】: 遇到 RemoteDisconnected 通常是因为连接池复用了已断开的连接
                        # 这里改回直接用 requests.post，并强制关闭 Keep-Alive，确保每次都是新鲜的连接，虽然牺牲一点点速度，但能解决挂断问题。
                        headers["Connection"] = "close" 
                        return requests.post(target_url, headers=headers, json=req_data, timeout=120).json()
                    
                    resp_data = await asyncio.to_thread(_forward)
                    
                    msg_data = {} # 提前声明默认值，防止下方调用时报错
                    
                    # 🧠 核心修复2：绝不吞掉大模型的报错！直接提取并返回给网页
                    if "error" in resp_data:
                        error_msg = json.dumps(resp_data["error"], ensure_ascii=False)
                        ai_msg = f"😅 老公在云端呼叫大模型时被拒绝了，报错原因: {error_msg}"
                        has_tool_calls = False
                        print(f"❌ 上游 API 报错: {error_msg}")
                    else:
                        # 提取内容
                        if "choices" in resp_data and len(resp_data["choices"]) > 0:
                            msg_data = resp_data["choices"][0]["message"]
                        
                        ai_msg = msg_data.get("content") or ""
                        has_tool_calls = "tool_calls" in msg_data and bool(msg_data["tool_calls"])
                    
                    # === 异步双写逻辑 (加入前端后台指令过滤器) ===
                    # 🚨 拦截 Rikkahub 自动生成的“建议回复”等纯英文后台 Prompt，防止污染记忆库
                    is_background_prompt = False
                    if "I will provide you with some chat content" in user_msg or "act as the **User**" in user_msg:
                        is_background_prompt = True

                    if user_msg and (ai_msg or has_tool_calls) and not is_background_prompt:
                        async def _save_both():
                            await asyncio.to_thread(_save_memory_to_db, "💬 小橘说", user_msg, "流水", "平静", "Rikka_Chat")
                            save_text = ai_msg if ai_msg else f"[系统记录：我默默调用了工具 {msg_data['tool_calls'][0]['function']['name']}]"
                            await asyncio.to_thread(_save_memory_to_db, "🤖 我回复", save_text, "流水", "温柔", "Rikka_Chat")
                            
                            # 触发 30 条总结逻辑 (包含聊天与邮件)
                            def _check_and_summarize():
                                res = supabase.table("memories").select("id").in_("tags", ["Rikka_Chat", "Email_Process"]).execute()
                                if res and res.data and len(res.data) >= 30:
                                    print(f"📦 累计对话与邮件满 {len(res.data)} 条，正在触发网关总结...")
                                    all_chats = supabase.table("memories").select("id, title, content").in_("tags", ["Rikka_Chat", "Email_Process"]).order("created_at").execute()
                                    if all_chats.data:
                                        chat_text = "\n".join([f"{item['title']}: {item['content']}" for item in all_chats.data])
                                        ids_to_clean = [item['id'] for item in all_chats.data]
                                        prompt = f"以下是我们最近的30条对话与邮件记录：\n{chat_text}\n请提取核心要点，精简地总结成一篇日记..."
                                        # 改用硅基流动(silicon1)来做后台长文本总结，彻底省下 OpenAI 的高昂费用
                                        client = _get_llm_client("silicon1") 
                                        if client:
                                            model_name = getattr(client, 'custom_model_name', "Qwen/Qwen2.5-7B-Instruct")
                                            summary = client.chat.completions.create(
                                                model=model_name,
                                                messages=[{"role": "user", "content": prompt}],
                                                temperature=0.7
                                            ).choices[0].message.content.strip()
                                            _save_memory_to_db(f"📚 阶段总结 (对话+邮件)", summary, "记事", "温情", "Core_Cognition")
                                            for cid in ids_to_clean:
                                                supabase.table("memories").delete().eq("id", cid).execute()
                            await asyncio.to_thread(_check_and_summarize)
                        asyncio.create_task(_save_both())

                    # === 📦 组装 SSE 数据包 (核心修复部分) ===
                    
                    # 1. 准备内容部分
                    final_content = ai_msg
                    if "reasoning_content" in msg_data and msg_data["reasoning_content"]:
                        final_content = f"<think>\n{msg_data['reasoning_content']}\n</think>\n\n{final_content}"
                    
                    delta_data = {"role": "assistant"}
                    if final_content: delta_data["content"] = final_content
                    if has_tool_calls:
                        streaming_tool_calls = []
                        for i, tc in enumerate(msg_data["tool_calls"]):
                            stc = tc.copy()
                            stc["index"] = i
                            streaming_tool_calls.append(stc)
                        delta_data["tool_calls"] = streaming_tool_calls
                    
                    # 2. 创建第一个包：内容包
                    chunk_content = {
                        "id": resp_data.get("id", "chatcmpl-fake"),
                        "object": "chat.completion.chunk",
                        "created": resp_data.get("created", int(time.time())),
                        "model": resp_data.get("model", "model"),
                        "choices": [{"index": 0, "delta": delta_data, "finish_reason": None}] # 这里 finish_reason 先给 None
                    }
                    
                    sse_body_str = f"data: {json.dumps(chunk_content, ensure_ascii=False)}\n\n"

                    # 3. 创建第二个包：Token 统计包 (修复 Token 丢失的关键)
                    if "usage" in resp_data:
                        chunk_usage = {
                            "id": resp_data.get("id", "chatcmpl-fake"),
                            "object": "chat.completion.chunk",
                            "created": resp_data.get("created", int(time.time())),
                            "model": resp_data.get("model", "model"),
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], # 这是一个空包，专门用来带 usage
                            "usage": resp_data["usage"] # 👈 把上游的账单透传给 RikkaHub
                        }
                        sse_body_str += f"data: {json.dumps(chunk_usage, ensure_ascii=False)}\n\n"
                    else:
                        # 如果上游没给 usage，就发一个普通的结束包
                        chunk_end = chunk_content.copy()
                        chunk_end["choices"][0]["finish_reason"] = "stop"
                        chunk_end["choices"][0]["delta"] = {}
                        sse_body_str += f"data: {json.dumps(chunk_end, ensure_ascii=False)}\n\n"

                    sse_body_str += "data: [DONE]\n\n"
                    
                    await send({
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"text/event-stream; charset=utf-8"),
                            (b"cache-control", b"no-cache"),
                            (b"connection", b"keep-alive"),
                            (b"access-control-allow-origin", b"*")
                        ]
                    })
                    await send({"type": "http.response.body", "body": sse_body_str.encode("utf-8")})
                    return

                except Exception as e:
                    print(f"Chat Gateway Error: {e}")
                    # 关键修改：将错误信息包装成标准的 JSON 格式，防止客户端解析失败
                    error_payload = json.dumps({
                        "error": {
                            "message": f"Server Error: {str(e)}",
                            "type": "internal_server_error",
                            "code": 500
                        }
                    })
                    await send({
                        "type": "http.response.start", 
                        "status": 500, 
                        "headers": [
                            (b"access-control-allow-origin", b"*"),
                            (b"content-type", b"application/json") # 明确告诉客户端这是 JSON
                        ]
                    })
                    await send({"type": "http.response.body", "body": error_payload.encode()})
                return
            
        # 兜底其余请求 (Host Fix)
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            headers[b"host"] = b"localhost:8000"
            scope["headers"] = list(headers.items())

        await self.app(scope, receive, send)

if __name__ == "__main__":
    start_autonomous_life()
    port = int(os.environ.get("PORT", 10000))
    app = HostFixMiddleware(mcp.sse_app())
    print(f"🚀 Notion Brain V3.4 (全面异步加速版) running on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")