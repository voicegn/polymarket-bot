#!/usr/bin/env python3
"""
Polymarket 简化回测脚本

由于 API 不提供历史价格数据，这里做一个简化的策略验证：
1. 获取已结算市场
2. 假设在市场价格偏离"合理值"时交易
3. 计算假设的收益

这不是严格回测，而是策略逻辑验证。
"""

import requests
import json
from decimal import Decimal
from collections import defaultdict

# Gamma API
GAMMA_URL = "https://gamma-api.polymarket.com"

def fetch_closed_markets(limit=100):
    """获取已结算市场"""
    url = f"{GAMMA_URL}/markets"
    params = {
        "closed": "true",
        "_limit": limit,
        "_sort": "volume:desc"
    }
    resp = requests.get(url, params=params)
    return resp.json()

def analyze_market(market):
    """分析单个市场"""
    try:
        outcomes = json.loads(market.get("outcomes", "[]"))
        prices = json.loads(market.get("outcomePrices", "[]"))
        
        if len(outcomes) != 2 or len(prices) != 2:
            return None
            
        yes_price = float(prices[0])
        no_price = float(prices[1])
        volume = float(market.get("volumeNum", 0))
        
        # 结算结果 (yes_price ≈ 1 表示 Yes 赢, ≈ 0 表示 No 赢)
        if yes_price > 0.9:
            resolution = "Yes"
        elif no_price > 0.9:
            resolution = "No"
        else:
            return None  # 未结算或分摊
            
        return {
            "id": market["id"],
            "question": market.get("question", "")[:60],
            "volume": volume,
            "resolution": resolution,
        }
    except:
        return None

def simulate_strategy(markets):
    """
    模拟简单策略：
    - 假设我们能在市场开盘时以 50% 买入
    - 持有到结算
    - 计算收益
    """
    total_trades = 0
    wins = 0
    losses = 0
    total_pnl = Decimal("0")
    
    print("\n📊 回测模拟 (简化版)")
    print("=" * 70)
    print("假设策略: 随机选择方向，模拟50%概率事件")
    print("=" * 70)
    
    results_by_category = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": Decimal("0")})
    
    for m in markets:
        if m is None:
            continue
            
        total_trades += 1
        
        # 简化: 假设我们买 Yes @ 0.50
        entry_price = Decimal("0.50")
        stake = Decimal("100")  # $100 per trade
        
        if m["resolution"] == "Yes":
            # Yes 赢了，我们赚
            pnl = stake * (Decimal("1") - entry_price) / entry_price
            wins += 1
        else:
            # No 赢了，我们亏
            pnl = -stake
            losses += 1
            
        total_pnl += pnl
        
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    print(f"\n总交易: {total_trades}")
    print(f"胜: {wins} | 负: {losses}")
    print(f"胜率: {win_rate:.1f}%")
    print(f"总 PnL: ${total_pnl:.2f}")
    print(f"平均每笔: ${total_pnl/total_trades:.2f}" if total_trades > 0 else "N/A")
    
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": float(total_pnl)
    }

def main():
    print("🔄 获取历史市场数据...")
    markets_raw = fetch_closed_markets(200)
    print(f"获取到 {len(markets_raw)} 个市场")
    
    markets = [analyze_market(m) for m in markets_raw]
    markets = [m for m in markets if m is not None]
    print(f"有效市场: {len(markets)}")
    
    # 显示一些市场样本
    print("\n📋 市场样本:")
    for m in markets[:10]:
        print(f"  [{m['resolution']:3}] {m['question']}... (Vol: ${m['volume']:,.0f})")
    
    # 运行模拟
    results = simulate_strategy(markets)
    
    print("\n" + "=" * 70)
    print("⚠️  注意: 这是简化模拟，不是真实回测")
    print("   真实回测需要:")
    print("   1. 历史价格数据 (需要付费或爬取)")
    print("   2. 模型预测概率 vs 市场价格比较")
    print("   3. 滑点和手续费计算")
    print("=" * 70)
    
    # 展示假设 LLM 策略的潜力
    print("\n🤖 LLM 策略潜力分析:")
    print("   如果 LLM 能提供 55% 准确率 (比随机高 5%):")
    
    edge = 0.05  # 5% edge
    trades_per_month = 50
    stake_per_trade = 100
    
    expected_profit = trades_per_month * stake_per_trade * edge * 2  # 2x because binary
    print(f"   - 每月 {trades_per_month} 笔交易 @ ${stake_per_trade}/笔")
    print(f"   - 预期月收益: ${expected_profit:.0f}")
    print(f"   - 年化: ${expected_profit * 12:.0f}")
    
    print("\n   如果 LLM 能提供 60% 准确率:")
    edge = 0.10
    expected_profit = trades_per_month * stake_per_trade * edge * 2
    print(f"   - 预期月收益: ${expected_profit:.0f}")
    print(f"   - 年化: ${expected_profit * 12:.0f}")

if __name__ == "__main__":
    main()
