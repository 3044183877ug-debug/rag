"""
refresh_validity.py — 轻量脚本：为已有的 data_source.yaml 条目回填有效性状态

只调用 API 列表接口（不重新下载详情页），根据 URL 匹配已有条目，更新 status 字段。
"""
import sys
import re
import yaml
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
YAML_PATH = BASE_DIR / "data_source.yaml"

# API 配置
LIST_API = "https://www.chinatax.gov.cn/getFileListByCodeId"
LIST_PAGE_URL = "https://fgk.chinatax.gov.cn/zcfgk/c100009/listflfg_fg.html"

CODE_CHANNEL_MAP = {
    "c100009": "d34fa7ad03f84f4caed12f5c2beae099",  # 法律
    "c100010": "f71d2ecc6a0d424a9db8e5aa17a0e3b3",  # 行政法规
    "c100011": "ab1e1ee5ca4c49668bcbd5b43b1f9efd",  # 税务部门规章
    "c100012": "7c1596815e1147d192265a5bbd0a6d2e",  # 财税文件
    "c100013": "30d2e80ce4e84071bdec2917543b686c",  # 税务规范性文件
}

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


def fetch_all_api_meta(code_name: str) -> dict[str, dict]:
    """获取指定分类下所有条目的 API 元数据，返回 {url末尾ID: meta} 映射。"""
    channel_id = CODE_CHANNEL_MAP.get(code_name)
    if not channel_id:
        print(f"未知分类: {code_name}")
        return {}

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": "https://fgk.chinatax.gov.cn",
        "Referer": LIST_PAGE_URL,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    })

    url_meta_map = {}
    for page in range(1, 20):
        params = {
            "codeId": "",
            "channelId": channel_id,
            "page": str(page),
            "size": "10",
            "relateSubChannels": "false",
        }
        try:
            resp = session.post(LIST_API, data=params, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"  API 请求失败 page={page}: {e}")
            break

        items = data.get("results", {}).get("data", {}).get("results", [])
        if not items:
            break

        for item in items:
            url = item.get("url", "")
            # 从 URL 提取唯一标识，如 c5193032
            m = re.search(r'/(c\d+)/content\.html', url)
            if m:
                content_id = m.group(1)
            else:
                content_id = url  # fallback

            meta = {}
            for domain in item.get("domainMetaList", []):
                for entry in domain.get("resultList", []):
                    key = entry.get("key", "")
                    value = (entry.get("value") or "").strip()
                    if key == "aging" and value:
                        meta["validity_raw"] = value
                        meta["validity"] = VALIDITY_MAP.get(value, value)
                    elif key == "writtendate" and value:
                        meta["date_published"] = value[:10]
                    elif key == "writtentext" and value:
                        meta["doc_number"] = value
                    elif key == "abolishdate" and value:
                        meta["abolish_date"] = value[:10]
                    elif key == "effectlevel" and value:
                        meta["effect_level"] = value
                    elif key == "writtendepartments" and value:
                        meta["issuing_authority"] = value
                    elif key == "taxpolicy" and value:
                        meta["tax_policy"] = value

            url_meta_map[content_id] = meta

        if len(items) < 10:
            break
        print(f"  已获取 page {page} ({len(items)} 条)")

    return url_meta_map


def main():
    if not YAML_PATH.exists():
        print("data_source.yaml 不存在")
        return

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    data = yaml.safe_load(content)
    if not isinstance(data, list):
        print("YAML 格式不是列表")
        return

    # 收集需要更新的条目
    to_update = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status", "")
        if "待确认有效性" in str(status) or "自动抓取" in str(status):
            to_update.append(entry)

    if not to_update:
        print("没有需要更新的条目（所有条目已有明确的有效性状态）")
        return

    print(f"找到 {len(to_update)} 条待确认有效性的条目")

    # 按分类代码对 URL 分组
    code_urls = {}
    for entry in to_update:
        url = entry.get("url", "")
        m = re.search(r'/zcfgk/(c\d+)/', url)
        code = m.group(1) if m else "c100009"
        if code not in code_urls:
            code_urls[code] = []
        code_urls[code].append(entry)

    # 逐分类获取 API 元数据
    for code, entries in code_urls.items():
        print(f"\n正在获取 {code} 的 API 元数据...")
        api_meta_map = fetch_all_api_meta(code)
        print(f"  获取到 {len(api_meta_map)} 条元数据")

        updated = 0
        for entry in entries:
            url = entry.get("url", "")
            m = re.search(r'/(c\d+)/content\.html', url)
            content_id = m.group(1) if m else None

            if content_id and content_id in api_meta_map:
                meta = api_meta_map[content_id]
                if meta.get("validity"):
                    old_status = entry.get("status", "")
                    entry["status"] = meta["validity"]
                    if meta.get("validity_raw"):
                        entry["validity_source"] = meta["validity_raw"]
                    if meta.get("date_published"):
                        entry["date_published"] = meta["date_published"]
                    if meta.get("doc_number"):
                        entry["doc_number"] = meta["doc_number"]
                    if meta.get("effect_level"):
                        entry["doc_type"] = meta["effect_level"]
                    if meta.get("issuing_authority"):
                        entry["issuing_authority"] = meta["issuing_authority"]
                    if meta.get("tax_policy"):
                        entry["tax_policy"] = meta["tax_policy"]
                    if meta.get("abolish_date"):
                        entry["abolish_date"] = meta["abolish_date"]
                        entry["status"] = "已废止"
                    print(f"  [OK] {entry.get('source_name', '')[:50]}: {old_status} -> {entry['status']}")
                    updated += 1
            else:
                print(f"  [WARN] 未匹配到 API 数据: {entry.get('source_name', '')[:50]}")

        print(f"  更新了 {updated}/{len(entries)} 条")

    # 写回 YAML（保留注释和格式）
    # 用简单方式：重建 YAML 文件
    lines = []
    for entry in data:
        if isinstance(entry, dict):
            lines.append(f'  - file_name: "{entry.get("file_name", "")}"')
            for key in ("source_name", "policy_name", "source", "source_number",
                         "doc_type", "status", "date_published", "date_effective",
                         "url", "applicable_scope", "topic",
                         "validity_source", "doc_number", "issuing_authority",
                         "tax_policy", "abolish_date"):
                if key not in entry:
                    continue
                val = entry[key]
                if isinstance(val, list):
                    items = ", ".join(f'"{v}"' for v in val)
                    lines.append(f"    {key}: [{items}]")
                else:
                    escaped = str(val).replace('"', '\\"')
                    lines.append(f'    {key}: "{escaped}"')
            # attachments
            if "attachments" in entry:
                lines.append(f"    attachments:")
                for att in entry["attachments"]:
                    lines.append(f'      - "{att}"')
            lines.append("")  # blank line between entries
        else:
            lines.append(str(entry))

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"\n[DONE] YAML 已更新: {YAML_PATH}")


if __name__ == "__main__":
    main()
