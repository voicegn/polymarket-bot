#!/usr/bin/env python3
"""
Polymarket Crypto Markets Scraper
从 https://polymarket.com/crypto 获取市场数据
"""

import requests
import json
import re
from datetime import datetime

def fetch_crypto_page():
    """获取 crypto 页面 HTML"""
    url = "https://polymarket.com/crypto"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers)
    return resp.text

def extract_next_data(html):
    """从 HTML 中提取 __NEXT_DATA__ JSON"""
    pattern = r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return None

def get_markets_via_api():
    """尝试通过可能的 API 获取市场"""
    endpoints = [
        "https://gamma-api.polymarket.com/events?tag_label=crypto&_limit=500",
        "https://gamma-api.polymarket.com/events?slug=crypto&_limit=500",
        "https://gamma-api.polymarket.com/markets?category=crypto&_limit=500",
    ]
    
    for url in endpoints:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    print(f"Found data at: {url}")
                    return data
        except Exception as e:
            print(f"Error: {url} - {e}")
    
    return None

def main():
    print("=" * 60)
    print("Polymarket Crypto Markets Scraper")
    print("=" * 60)
    
    # 方法 1: 尝试 API
    print("\n[1] Trying API endpoints...")
    api_data = get_markets_via_api()
    if api_data:
        print(f"Found {len(api_data)} markets via API")
        with open("data/crypto_markets_api.json", "w") as f:
            json.dump(api_data, f, indent=2)
    
    # 方法 2: 抓取页面
    print("\n[2] Fetching crypto page...")
    html = fetch_crypto_page()
    
    next_data = extract_next_data(html)
    if next_data:
        print("Found __NEXT_DATA__")
        with open("data/crypto_page_data.json", "w") as f:
            json.dump(next_data, f, indent=2)
        
        # 尝试提取市场数据
        try:
            page_props = next_data.get("props", {}).get("pageProps", {})
            markets = page_props.get("markets", [])
            events = page_props.get("events", [])
            
            print(f"Markets in page: {len(markets)}")
            print(f"Events in page: {len(events)}")
            
            if markets:
                with open("data/crypto_markets.json", "w") as f:
                    json.dump(markets, f, indent=2)
                    
        except Exception as e:
            print(f"Error parsing page data: {e}")
    else:
        print("No __NEXT_DATA__ found, page might be client-rendered")
        
    print("\n[3] Summary")
    print("-" * 40)
    print("需要进一步分析页面结构来获取完整市场数据")
    print("建议: 使用浏览器开发者工具抓取 Network 请求")

if __name__ == "__main__":
    main()
