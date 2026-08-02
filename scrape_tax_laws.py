"""
scrape_tax_laws.py  v4 — 纯 HTTP API 版本
-------------------------------------------
国家税务总局政策法规库 — 税法文件抓取工具。

v4 核心策略变更（针对 API 驱动渲染的网站）:
  1. **彻底放弃浏览器自动化**：网站列表数据通过 AJAX POST API 动态加载，
     HTML 页面的 <ul> 是空的，Playwright 会被 WAF 拦截。
  2. 直接调用后端 API https://www.chinatax.gov.cn/getFileListByCodeId
     获取 JSON 格式的法律条目列表（total: 75 部法律）。
  3. 详情页是静态 HTML，使用 requests + BeautifulSoup 提取内容。
  4. 支持翻页抓取所有条目。

v3 → v4 失败根因:
  - 网站使用 Elasticsearch 后端 + AJAX 动态渲染列表
  - Playwright 浏览器指纹被 WAF 识别 → 重定向到首页
  - 即便 HTTP 请求能拿到列表页 HTML，<ul> 也是空的（数据靠 JS 渲染）

用法:
    python scrape_tax_laws.py              # 抓取法律 (c100009)
    python scrape_tax_laws.py --dry-run    # 仅列出可抓取的条目
    python scrape_tax_laws.py --code c100010  # 抓取行政法规
    python scrape_tax_laws.py --code c100013  # 抓取部门公告
"""

import os
import re
import sys
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import requests
import yaml
from bs4 import BeautifulSoup

# ════════════════════════════════════════════════════════════════════
# 路径配置
# ════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "tax law documents"
YAML_PATH = BASE_DIR / "data_source.yaml"

# ════════════════════════════════════════════════════════════════════
# 日志
# ════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scrape_tax")

# ════════════════════════════════════════════════════════════════════
# API 端点
# ════════════════════════════════════════════════════════════════════
LIST_API = "https://www.chinatax.gov.cn/getFileListByCodeId"
LIST_PAGE_URL = "https://fgk.chinatax.gov.cn/zcfgk/c100009/listflfg_fg.html"

# ════════════════════════════════════════════════════════════════════
# 分类代码 → channelId 映射（从 HTML 页面 JS 中提取）
# ════════════════════════════════════════════════════════════════════
CODE_CHANNEL_MAP = {
    "c100009": "d34fa7ad03f84f4caed12f5c2beae099",  # 法律
    "c100010": "e1cd1569d1ea4a25a11041248925a081",  # 行政法规
    "c100011": "ab1e1ee5ca4c49668bcbd5b43b1f9efd",  # 税务部门规章
    "c100012": "7c1596815e1147d192265a5bbd0a6d2e",  # 财税文件
    "c100013": "30d2e80ce4e84071bdec2917543b686c",  # 税务规范性文件
}

CODE_NAMES = {
    "c100009": "法律",
    "c100010": "行政法规",
    "c100011": "税务部门规章",
    "c100012": "财税文件",
    "c100013": "税务规范性文件",
}

# 延迟范围（秒）
DELAY_API = (1.0, 3.0)
DELAY_DETAIL = (1.5, 4.0)

# 每页条目数
PAGE_SIZE = 10

# ════════════════════════════════════════════════════════════════════
# 财税相关关键词（只抓取与税收/财政直接相关的法律）
# ════════════════════════════════════════════════════════════════════
TAX_KEYWORDS = ['税', '发票', '财政', '征收管理', '税收']

# ════════════════════════════════════════════════════════════════════
# 现有法律关键词（去重）
# ════════════════════════════════════════════════════════════════════
_EXISTING_LAWS_KEYWORDS = [
    "财税2015_119号", "财税2018_64号", "财税2023_7号",
    "财税2023_12号", "财税2026_10号",
    "国发2018_41号", "国发2023_13号",
    "个人所得税法", "企业所得税法", "契税法", "增值税法",
]


# ════════════════════════════════════════════════════════════════════
# HTTP 会话
# ════════════════════════════════════════════════════════════════════

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return s


def api_headers() -> dict:
    return {
        "Origin": "https://fgk.chinatax.gov.cn",
        "Referer": LIST_PAGE_URL,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }


# ════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════

def human_delay(lo: float, hi: float, reason: str = ""):
    d = random.uniform(lo, hi)
    if reason:
        log.debug(f"  ⏳ {reason}: {d:.1f}s")
    time.sleep(d)


def safe_filename(title: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|\n\r\t]', '', title)
    cleaned = re.sub(r'\s+', '_', cleaned).strip('_')
    return cleaned


def strip_html(text: str) -> str:
    """去除 HTML 标签，保留纯文本。"""
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


# ════════════════════════════════════════════════════════════════════
# 有效性 & 元数据提取
# ════════════════════════════════════════════════════════════════════

# 有效性状态映射：将网站原文映射为标准化状态值
VALIDITY_MAP = {
    "全文有效": "现行有效",
    "现行有效": "现行有效",
    "全文废止": "已废止",
    "已废止": "已废止",
    "全文失效": "已失效",
    "已失效": "已失效",
    "部分失效": "部分失效",
    "已被修改": "已被修改",
    "尚未生效": "尚未生效",
}


def normalize_validity(raw: str) -> str:
    """将网站原始有效性文本映射为标准状态值。"""
    if not raw:
        return "待确认"
    raw = raw.strip()
    return VALIDITY_MAP.get(raw, raw)


def extract_meta_from_api(item: dict) -> dict:
    """从 API 返回的 domainMetaList 中提取元数据。

    返回 dict:
        - validity_raw: 时效性原文（如 "全文有效"）
        - validity: 标准化后的状态（如 "现行有效"）
        - date_published: 成文日期 (YYYY-MM-DD)
        - doc_number: 发文字号
        - abolish_date: 废止日期（如有）
        - effect_level: 效力等级（如 "法律"）
        - issuing_authority: 发文机关
        - tax_policy: 税务政策分类
    """
    meta = {
        "validity_raw": None,
        "validity": None,
        "date_published": None,
        "doc_number": None,
        "abolish_date": None,
        "effect_level": None,
        "issuing_authority": None,
        "tax_policy": None,
    }

    domain_list = item.get("domainMetaList", [])
    if not domain_list:
        return meta

    for domain in domain_list:
        for entry in domain.get("resultList", []):
            key = entry.get("key", "")
            value = (entry.get("value") or "").strip()
            if not value:
                continue

            if key == "aging":                 # 时效性
                meta["validity_raw"] = value
                meta["validity"] = normalize_validity(value)
            elif key == "writtendate":         # 成文日期
                meta["date_published"] = value[:10] if value else None
            elif key == "writtentext":         # 发文字号
                meta["doc_number"] = value
            elif key == "abolishdate":         # 废止日期
                meta["abolish_date"] = value[:10] if value else None
            elif key == "effectlevel":         # 效力等级
                meta["effect_level"] = value
            elif key == "writtendepartments":  # 发文机关
                meta["issuing_authority"] = value
            elif key == "taxpolicy":           # 税务政策分类
                meta["tax_policy"] = value

    # 如果有废止日期但未标明废止，标记为已废止
    if meta["abolish_date"] and not meta["validity_raw"]:
        meta["validity_raw"] = f"已废止(废止日期: {meta['abolish_date']})"
        meta["validity"] = "已废止"

    return meta


def extract_validity_from_page(soup) -> str | None:
    """从详情页 HTML 提取有效性状态（兜底方案）。

    查找模式:
      - <span class="xg">全文有效</span>   (fgk.chinatax.gov.cn)
      - 页面文本中的 "时效性：XXX" / "有效性：XXX"
    """
    # 方案1: fgk 网站的专用 class
    xg = soup.find("span", class_="xg")
    if xg:
        text = xg.get_text(strip=True)
        if text and len(text) <= 20:
            return normalize_validity(text)

    # 方案2: 全文搜索常见模式
    body = soup.get_text()
    for pattern in [
        r'时效性[：:]\s*(.{1,20})',
        r'有效性[：:]\s*(.{1,20})',
        r'效力状态[：:]\s*(.{1,20})',
    ]:
        m = re.search(pattern, body)
        if m:
            return normalize_validity(m.group(1).strip())

    return None


def extract_date_from_page(soup) -> str | None:
    """从详情页 HTML 提取成文日期（兜底方案）。

    查找模式:
      - <span class="date">成文日期：2025-10-28</span>
    """
    date_span = soup.find("span", class_="date")
    if date_span:
        text = date_span.get_text(strip=True)
        m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if m:
            return m.group(1)
    return None


# ════════════════════════════════════════════════════════════════════
# HTML 表格 → Markdown 转换（保留二维结构）
# ════════════════════════════════════════════════════════════════════

def _bs4_table_to_markdown(table) -> str:
    """将 BeautifulSoup <table> 转为 Markdown 表格，保留行列关系。

    处理规则：
      - 遍历 <tr>，取 <th>/<td> 的文本
      - <br> 标签：转换为分号分隔，保留子行信息（防止多行内容被压缩成一维）
      - rowspan: 维护 column → remaining_rows 映射，自动在后续行插入占位
      - colspan: 同一行内重复填充 cell 文本
      - 行宽不一致时用空串补齐到最大列数
      - 单元格内换行全部压缩为空格
    """
    rows = table.find_all("tr")
    if not rows:
        return ""

    grid: list[list[str]] = []
    # col → 还需跨越的行数（rowspan 占位）
    rowspan_tracker: dict[int, int] = {}

    for _ri, tr in enumerate(rows):
        cells = tr.find_all(["th", "td"])
        row_data: list[str] = []
        col = 0

        for cell in cells:
            # 1) 跳过被前序 rowspan 占用的列
            while col in rowspan_tracker:
                # 从上一行同列取值作为占位内容
                prev_val = grid[-1][col] if grid and col < len(grid[-1]) else ""
                row_data.append(prev_val)
                rowspan_tracker[col] -= 1
                if rowspan_tracker[col] <= 0:
                    del rowspan_tracker[col]
                col += 1

            # 2) 提取 cell 文本（<br> → "；" 保留子行分隔）
            for br in cell.find_all("br"):
                br.replace_with("；")
            text = cell.get_text(strip=True)
            text = " ".join(text.split())

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # 3) 按 colspan 重复填充，按 rowspan 登记到 tracker
            for _ in range(colspan):
                row_data.append(text)
                if rowspan > 1:
                    rowspan_tracker[col] = rowspan - 1
                col += 1

        # 4) 行尾清空剩余的 rowspan 占位列
        while col in rowspan_tracker:
            prev_val = grid[-1][col] if grid and col < len(grid[-1]) else ""
            row_data.append(prev_val)
            rowspan_tracker[col] -= 1
            if rowspan_tracker[col] <= 0:
                del rowspan_tracker[col]
            col += 1

        grid.append(row_data)

    # 5) 确定列数 + 补齐
    ncols = max(len(r) for r in grid) if grid else 0
    if ncols == 0:
        return ""
    for row in grid:
        while len(row) < ncols:
            row.append("")

    # 6) 组装 Markdown
    lines: list[str] = []
    lines.append("| " + " | ".join(grid[0]) + " |")
    lines.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def build_dedup_set() -> set:
    dedup = set()
    for kw in _EXISTING_LAWS_KEYWORDS:
        dedup.add(kw)
    if DOCS_DIR.exists():
        for f in DOCS_DIR.iterdir():
            if f.suffix.lower() in (".txt", ".pdf"):
                dedup.add(f.stem)
    if YAML_PATH.exists():
        try:
            with open(YAML_PATH, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        fn = entry.get("file_name", "")
                        if fn:
                            dedup.add(Path(fn).stem)
                        for key in ("source_name", "policy_name"):
                            val = entry.get(key, "")
                            if val:
                                dedup.add(val.strip())
        except yaml.YAMLError:
            pass
    return dedup


def is_duplicate(text: str, dedup_set: set) -> bool:
    text_clean = strip_html(text).strip()
    if text_clean in dedup_set:
        return True
    for known in dedup_set:
        if len(known) < 4:
            continue
        if known in text_clean or text_clean in known:
            # 防止短字符串误杀长标题（如"环境保护税法" in "环境保护税法实施条例"）
            shorter = min(len(known), len(text_clean))
            longer = max(len(known), len(text_clean))
            if shorter / longer >= 0.80:
                return True
    return False


def extract_date_from_text(text: str) -> str | None:
    text = strip_html(text)
    m = re.search(r'(20[1-3]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])', text)
    if m:
        return m.group(0)
    m = re.search(r'(20[1-3]\d)年(\d{1,2})月(\d{1,2})日', text)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return None


def format_yaml_entry(entry: dict) -> str:
    lines = [f'- file_name: "{entry["file_name"]}"']
    for key in ("source_name", "doc_type", "status", "date_published",
                 "date_effective", "url", "applicable_scope",
                 "validity_source", "doc_number", "issuing_authority",
                 "tax_policy", "abolish_date"):
        if key not in entry:
            continue
        val = entry[key]
        if isinstance(val, list):
            items = ", ".join(f'"{v}"' for v in val)
            lines.append(f"  {key}: [{items}]")
        else:
            escaped = str(val).replace('"', '\\"')
            lines.append(f'  {key}: "{escaped}"')
    return "\n".join(lines)


def append_to_yaml(entries: list[dict]):
    if not entries:
        return
    with open(YAML_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        for i, entry in enumerate(entries):
            f.write(f"# 🤖 自动抓取 #{i + 1}\n")
            f.write(format_yaml_entry(entry))
            f.write("\n")
    log.info(f"✅ YAML 追加完成：{len(entries)} 条新记录 → {YAML_PATH}")


# ════════════════════════════════════════════════════════════════════
# API 调用
# ════════════════════════════════════════════════════════════════════

def fetch_law_list(session: requests.Session, code_name: str,
                   page: int = 1, size: int = PAGE_SIZE) -> list[dict]:
    """
    调用 getFileListByCodeId API 获取法律列表。

    返回: list[dict]，每个 dict 包含:
        - titleHtml: 标题（含 HTML）
        - url: 相对路径，如 /chinatax/...
        - publishedTimeStr: 发布日期字符串
        - subTitleHtml: 副标题
        - channelName: 分类名称
    """
    channel_id = CODE_CHANNEL_MAP.get(code_name)
    if not channel_id:
        log.error(f"❌ 未知分类代码: {code_name}")
        return []

    headers = api_headers()
    params = {
        "codeId": "",
        "channelId": channel_id,
        "page": str(page),
        "size": str(size),
        "relateSubChannels": "false",
    }

    try:
        resp = session.post(LIST_API, data=params, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"❌ API 请求失败 (page={page}): {e}")
        return []

    try:
        data = resp.json()
    except ValueError:
        log.error(f"❌ API 响应非 JSON: {resp.text[:200]}")
        return []

    if data.get("code") != 200:
        log.error(f"❌ API 返回错误: {data.get('code')}")
        return []

    results_data = data.get("results", {}).get("data", {})
    total = results_data.get("total", 0)
    items = results_data.get("results", [])

    log.info(f"  📊 API 返回: page={page}, 本页 {len(items)} 条, 总计 {total} 条")

    return items


def fetch_all_law_list(session: requests.Session, code_name: str,
                       max_pages: int = 20) -> list[dict]:
    """分页获取所有法律条目。"""
    all_items = []
    for page in range(1, max_pages + 1):
        log.info(f"📡 获取第 {page} 页...")
        items = fetch_law_list(session, code_name, page=page)
        if not items:
            if page == 1:
                log.error("❌ 第 1 页无数据，可能 API 异常")
            break
        all_items.extend(items)
        if len(items) < PAGE_SIZE:
            log.info(f"🏁 已获取全部 {len(all_items)} 条")
            break
        human_delay(*DELAY_API, "翻页间隔")
    return all_items


# ════════════════════════════════════════════════════════════════════
# 详情页抓取
# ════════════════════════════════════════════════════════════════════

def scrape_detail_page(session: requests.Session, entry: dict) -> tuple[str | None, dict]:
    """抓取详情页内容，同时提取页面元数据。

    返回: (content_text, page_meta_dict)
        content_text: 提取的正文内容，失败返回 None
        page_meta_dict: 从页面 HTML 提取的元数据（validity, date_published）
    """
    page_meta: dict = {"validity": None, "date_published": None}

    url_path = entry.get("url", "")
    title_html = entry.get("titleHtml", "未知标题")
    title_plain = strip_html(title_html)

    # 构造完整 URL
    # API 返回的 URL 是 www.chinatax.gov.cn，但实际内容在 fgk.chinatax.gov.cn
    if url_path.startswith("http"):
        url = url_path.replace("www.chinatax.gov.cn", "fgk.chinatax.gov.cn")
    elif url_path.startswith("/"):
        url = f"https://fgk.chinatax.gov.cn{url_path}"
    else:
        log.warning(f"  ⚠️ 无效 URL: {url_path}")
        return None, page_meta

    log.info(f"  🔍 {title_plain[:70]}")
    log.debug(f"     URL: {url}")

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        # 强制 UTF-8：服务器不返回 charset，requests 误判为 ISO-8859-1
        resp.encoding = "utf-8"
    except requests.RequestException as e:
        log.warning(f"  ⚠️ 请求失败: {e}")
        return None, page_meta

    human_delay(*DELAY_DETAIL, "详情页停留")

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── 从页面 HTML 提取元数据（兜底用）──
    page_validity = extract_validity_from_page(soup)
    if page_validity:
        page_meta["validity"] = page_validity
        log.info(f"    🏷️  页面有效性: {page_validity}")
    page_date = extract_date_from_page(soup)
    if page_date:
        page_meta["date_published"] = page_date

    # --- 尝试多种内容选择器，按优先级 ---
    content_selectors = [
        # TRS 系统常见容器
        {"id": "fontzoom"},
        {"class_": "TRS_Editor"},
        {"class_": "TRS_PreAppend"},
        {"id": "zoom"},
        # 通用文章容器
        {"class_": "article"},
        {"class_": "article-content"},
        {"class_": "article_con"},
        {"class_": "content"},
        {"class_": "detail-con"},
        {"id": "UCAP-CONTENT"},
        {"class_": "main-content"},
        # HTML5 语义标签
        {"name": "article"},
    ]

    best_text = None
    best_len = 0

    for sel in content_selectors:
        if "id" in sel:
            container = soup.find("div", id=sel["id"])
        elif "class_" in sel:
            container = soup.find("div", class_=sel["class_"])
        elif "name" in sel:
            container = soup.find(sel["name"])
        else:
            continue

        if container:
            # 去除脚本和样式
            for tag in container.find_all(["script", "style"]):
                tag.decompose()

            # ── 表格保留：将 <table> 转为 Markdown 后再提取文本 ──
            from bs4 import NavigableString
            for tbl in container.find_all("table"):
                md = _bs4_table_to_markdown(tbl)
                tbl.insert_after(NavigableString(f"\n\n{md}\n\n"))
                tbl.decompose()

            text = container.get_text(separator="\n")
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
            if len(text) > best_len:
                best_text = text
                best_len = len(text)

    if best_text and best_len > 200:
        # 清理常见的 UI/导航噪音行
        lines = best_text.split('\n')
        cleaned_lines = []
        skip_patterns = [
            r'^字体[：:]',
            r'^【[大中小]】$',
            r'^分享到',
            r'^分享$',
            r'^收藏$',
            r'^订阅$',
            r'^已推送',
            r'^个人中心',
            r'^订阅设置',
            r'^全文有效$',
            r'^语音播报',
            r'^语音播放',
            r'^扫一扫在手机打开当前页$',
            r'^注释$',
            r'^此稿件无标签',
            r'^中查看$',
            r'^中订阅更多$',
            r'^"个人中心',
            r'^"订阅设置',
        ]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append('')
                continue
            skip = False
            for pat in skip_patterns:
                if re.match(pat, stripped):
                    skip = True
                    break
            if not skip:
                cleaned_lines.append(stripped)
        best_text = '\n'.join(cleaned_lines)
        best_text = re.sub(r'\n{3,}', '\n\n', best_text).strip()
        log.info(f"    📝 提取成功 ({len(best_text)} 字符)")
        return best_text, page_meta

    # 兜底：body 文本（去除脚本、样式）
    body = soup.find("body")
    if body:
        for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = body.get_text(separator="\n")
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        if len(text) >= 100:
            log.warning(f"    ⚠️ 兜底提取 body ({len(text)} 字符)")
            return text, page_meta

    log.warning(f"    ⚠️ 未能提取内容")
    return None, page_meta


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════

def run_scrape(code_name: str, max_pages: int, dry_run: bool, max_items: int = 0):
    DOCS_DIR.mkdir(exist_ok=True)
    dedup_set = build_dedup_set()
    log.info(f"📊 去重集合: {len(dedup_set)} 条")
    log.info(f"📂 分类: {CODE_NAMES.get(code_name, code_name)} ({code_name})")

    session = create_session()

    # Step 1: 获取法律列表
    log.info(f"\n{'='*60}")
    log.info(f"📡 Step 1: 调用 API 获取法律列表")
    log.info(f"{'='*60}")

    all_items = fetch_all_law_list(session, code_name, max_pages=max_pages)
    log.info(f"\n📋 总计获取: {len(all_items)} 条 {CODE_NAMES.get(code_name, '')}")

    if not all_items:
        log.error("❌ 未获取到任何条目")
        return

    # Step 2: 筛选新条目
    new_items = []
    skipped_invalid = 0
    for item in all_items:
        title = strip_html(item.get("titleHtml", ""))
        # 财税关键词过滤
        if not any(kw in title for kw in TAX_KEYWORDS):
            log.info(f"  ⏭️  非财税，跳过: {title[:60]}")
            continue
        if is_duplicate(title, dedup_set):
            log.info(f"  ⏭️  已存在，跳过: {title[:60]}")
            continue
        # ── 有效性过滤：全文失效/废止的不抓取 ──
        api_meta = extract_meta_from_api(item)
        validity = api_meta.get("validity") or ""
        validity_raw = api_meta.get("validity_raw") or ""
        if validity in ("已失效", "已废止") or validity_raw in ("全文失效", "全文废止"):
            log.info(f"  🗑️  {validity_raw or validity}，跳过: {title[:60]}")
            skipped_invalid += 1
            continue
        new_items.append(item)

    log.info(f"🆕 财税相关新增条目: {len(new_items)} 条")
    if skipped_invalid:
        log.info(f"🗑️  因失效/废止自动跳过: {skipped_invalid} 条")

    if dry_run:
        log.info(f"\n{'='*60}")
        log.info(f"💡 DRY-RUN 模式 — 以下为可抓取条目:")
        log.info(f"{'='*60}")
        for i, item in enumerate(new_items):
            title = strip_html(item.get("titleHtml", ""))
            url = item.get("url", "")
            pub_date = item.get("publishedTimeStr", "")[:10]
            # 从 API 提取有效性
            api_meta = extract_meta_from_api(item)
            validity_label = api_meta.get("validity_raw") or "未知"
            log.info(f"  [{i+1}] {title[:80]}  ({pub_date})  [{validity_label}]")
        log.info(f"\n🏁 DRY-RUN 完成：{len(new_items)} 条可抓取")
        return

    if not new_items:
        log.info("\n🏁 无新条目需要抓取")
        return

    # Step 3: 抓取详情页
    log.info(f"\n{'='*60}")
    log.info(f"📑 Step 2: 抓取详情页内容")
    log.info(f"{'='*60}")

    results = []
    for i, item in enumerate(new_items):
        if max_items > 0 and len(results) >= max_items:
            log.info(f"🛑 已达最大条目数 {max_items}，停止抓取")
            break
        human_delay(*DELAY_DETAIL, "条目间停顿")
        log.info(f"\n[{i+1}/{len(new_items)}]")

        # ── 优先从 API 元数据提取有效性 ──
        api_meta = extract_meta_from_api(item)
        if api_meta["validity_raw"]:
            log.info(f"  🏷️  API时效性: {api_meta['validity_raw']} → {api_meta['validity']}")

        content, page_meta = scrape_detail_page(session, item)
        if content:
            title_html = item.get("titleHtml", "")
            title = strip_html(title_html)

            # ── 合并元数据：API 优先，页面兜底 ──
            pub_date = (
                api_meta["date_published"]
                or page_meta.get("date_published")
                or (item.get("publishedTimeStr", "") or "")[:10]
            )
            validity = (
                api_meta["validity"]
                or page_meta.get("validity")
                or "待确认"
            )
            validity_raw = api_meta.get("validity_raw") or page_meta.get("validity") or ""

            results.append({
                "title": title,
                "url": item.get("url", ""),
                "date": pub_date,
                "validity": validity,
                "validity_raw": validity_raw,
                "doc_number": api_meta.get("doc_number") or "",
                "effect_level": api_meta.get("effect_level") or "",
                "issuing_authority": api_meta.get("issuing_authority") or "",
                "tax_policy": api_meta.get("tax_policy") or "",
                "abolish_date": api_meta.get("abolish_date") or "",
                "content": content,
            })
            dedup_set.add(title)

    # Step 4: 保存
    log.info(f"\n{'='*60}")
    log.info(f"💾 Step 3: 保存文件")
    log.info(f"{'='*60}")

    yaml_entries = []
    for r in results:
        title = r["title"].strip()
        date_pub = r["date"] or datetime.now().strftime("%Y-%m-%d")
        validity_status = r.get("validity", "待确认")

        filename = safe_filename(title) + ".txt"
        filepath = DOCS_DIR / filename
        if filepath.exists():
            filename = safe_filename(title) + f"_{hash(r['url']) % 10000:04d}.txt"
            filepath = DOCS_DIR / filename

        filepath.write_text(r["content"], encoding="utf-8")
        log.info(f"💾 {filename} ({len(r['content'])} 字符)")

        entry = {
            "file_name": filename,
            "source_name": title,
            "doc_type": r.get("effect_level") or "国家法律",
            "status": validity_status,
            "date_published": date_pub,
            "date_effective": date_pub,
            "url": r["url"],
            "applicable_scope": ["待分类"],
        }
        # 附加元数据注释
        if r.get("validity_raw"):
            entry["validity_source"] = r["validity_raw"]
        if r.get("doc_number"):
            entry["doc_number"] = r["doc_number"]
        if r.get("issuing_authority"):
            entry["issuing_authority"] = r["issuing_authority"]
        if r.get("tax_policy"):
            entry["tax_policy"] = r["tax_policy"]
        if r.get("abolish_date"):
            entry["abolish_date"] = r["abolish_date"]
            entry["status"] = "已废止"

        yaml_entries.append(entry)

    append_to_yaml(yaml_entries)

    log.info(f"\n{'='*60}")
    log.info(f"🏁 抓取完成！新增 {len(results)} 条")
    log.info(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="国家税务总局政策法规库 — 税法抓取 v4 (纯 HTTP API)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出可抓取条目，不实际下载")
    parser.add_argument("--max-pages", type=int, default=20,
                        help="最大翻页数")
    parser.add_argument("--code", type=str, default="c100009",
                        choices=list(CODE_CHANNEL_MAP.keys()),
                        help="分类代码 (默认: c100009 法律)")
    parser.add_argument("--max-items", type=int, default=0,
                        help="最多抓取条目数 (0=不限制)")
    args = parser.parse_args()

    run_scrape(
        code_name=args.code,
        max_pages=args.max_pages,
        dry_run=args.dry_run,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
