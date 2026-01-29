#!/usr/bin/env python3
"""
LLM Market Analyzer - Uses DeepSeek to analyze Polymarket markets
Real LLM analysis vs simulated random analysis comparison
"""

import json
import random
import os
import urllib.request
from datetime import datetime
from pathlib import Path

# Load environment variables
def load_env():
    """Load .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env()

# Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
MODEL = 'deepseek-chat'

# Strategy Parameters
MIN_EDGE = 0.05          # 5% minimum edge
MIN_CONFIDENCE = 0.70    # 70% confidence threshold
MAX_KELLY = 0.15         # 15% max Kelly fraction
MAX_POSITION = 0.02      # 2% max position size
INITIAL_CAPITAL = 1000   # Starting USDC
MAX_MARKETS = 20         # Limit API calls

def fetch_markets():
    """Fetch active markets from Polymarket Gamma API"""
    url = "https://gamma-api.polymarket.com/markets?closed=false&limit=500"
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []

def call_deepseek(prompt, max_tokens=500):
    """Call DeepSeek API for analysis"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set")
    
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }
    
    payload = {
        'model': MODEL,
        'messages': [
            {
                'role': 'system',
                'content': '''You are an expert prediction market analyst. Analyze markets and provide probability estimates.

For each market, respond in this exact JSON format:
{
    "probability": <number 0-100>,
    "confidence": <number 0-100>,
    "reasoning": "<brief analysis>"
}

Be concise. Focus on factual analysis.'''
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_tokens': max_tokens,
        'temperature': 0.3
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            return result['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ''
        print(f"API Error: {e.code} - {error_body}")
        return None
    except Exception as e:
        print(f"Request Error: {e}")
        return None

def parse_llm_response(response_text):
    """Parse LLM response to extract probability and confidence"""
    try:
        # Try to extract JSON from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            data = json.loads(json_str)
            return {
                'probability': float(data.get('probability', 50)) / 100,
                'confidence': float(data.get('confidence', 50)) / 100,
                'reasoning': data.get('reasoning', '')
            }
    except:
        pass
    
    # Fallback: return moderate values
    return {
        'probability': 0.5,
        'confidence': 0.5,
        'reasoning': 'Failed to parse response'
    }

def parse_prices(market):
    """Extract Yes/No prices from market"""
    try:
        prices_str = market.get('outcomePrices', '[]')
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        if len(prices) >= 2:
            return float(prices[0]), float(prices[1])
    except:
        pass
    return None, None

def calculate_market_quality(market):
    """Score market quality 0-100"""
    score = 0
    volume = market.get('volumeNum', 0) or 0
    liquidity = market.get('liquidityNum', 0) or 0
    vol_24h = market.get('volume24hr', 0) or 0
    
    if volume > 1000000: score += 30
    elif volume > 100000: score += 20
    elif volume > 10000: score += 10
    
    if liquidity > 100000: score += 30
    elif liquidity > 10000: score += 20
    elif liquidity > 1000: score += 10
    
    if vol_24h > 10000: score += 20
    elif vol_24h > 1000: score += 10
    elif vol_24h > 100: score += 5
    
    spread = market.get('spread', 1) or 1
    if spread < 0.02: score += 20
    elif spread < 0.05: score += 15
    elif spread < 0.10: score += 10
    
    return score

def analyze_with_llm(market, yes_price, no_price):
    """Use DeepSeek to analyze a market"""
    question = market.get('question', 'Unknown')
    description = market.get('description', '')[:500]
    end_date = market.get('endDate', 'Unknown')
    volume = market.get('volumeNum', 0) or 0
    
    prompt = f"""Analyze this prediction market:

QUESTION: {question}

CONTEXT: {description}

CURRENT PRICES:
- YES: {yes_price:.1%} (market thinks {yes_price:.0%} chance)
- NO: {no_price:.1%}

END DATE: {end_date}
TOTAL VOLUME: ${volume:,.0f}

What is your estimated TRUE probability that YES wins?
Consider current events, logic, and any known information.
"""

    print(f"  Calling DeepSeek for: {question[:60]}...")
    response = call_deepseek(prompt)
    
    if response:
        parsed = parse_llm_response(response)
        parsed['raw_response'] = response
        return parsed
    else:
        return {
            'probability': yes_price,  # Fall back to market price
            'confidence': 0.3,
            'reasoning': 'API call failed',
            'raw_response': None
        }

def simulate_random_analysis(market, yes_price, no_price):
    """Random baseline analysis for comparison"""
    quality = calculate_market_quality(market)
    
    # Random with slight bias toward market price
    noise = random.gauss(0, 0.15)
    prob = max(0.05, min(0.95, yes_price + noise))
    conf = random.uniform(0.4, 0.9)
    
    return {
        'probability': prob,
        'confidence': conf,
        'reasoning': f'Random estimate with quality={quality}'
    }

def kelly_fraction(prob, odds, confidence):
    """Calculate Kelly bet size"""
    if odds <= 0:
        return 0
    q = 1 - prob
    edge = prob * odds - q
    if edge <= 0:
        return 0
    kelly = edge / odds
    adjusted = kelly * confidence * MAX_KELLY
    return min(adjusted, MAX_POSITION)

def generate_signal(market, analysis, yes_price, no_price):
    """Generate trading signal based on analysis"""
    est_prob = analysis['probability']
    confidence = analysis['confidence']
    
    signal = {
        'market_id': market.get('id'),
        'question': market.get('question', '')[:100],
        'yes_price': yes_price,
        'no_price': no_price,
        'estimated_prob': est_prob,
        'confidence': confidence,
        'action': 'HOLD',
        'direction': None,
        'edge': 0,
        'position_size': 0,
        'reason': ''
    }
    
    # Check YES opportunity
    if yes_price and 0.01 < yes_price < 0.99:
        yes_odds = 1 / yes_price - 1
        yes_edge = est_prob - yes_price
        
        if yes_edge >= MIN_EDGE and confidence >= MIN_CONFIDENCE:
            size = kelly_fraction(est_prob, yes_odds, confidence)
            if size > 0.001:
                signal['action'] = 'BUY'
                signal['direction'] = 'YES'
                signal['edge'] = yes_edge
                signal['position_size'] = size
                signal['reason'] = f"YES edge {yes_edge:.1%}"
    
    # Check NO opportunity
    if signal['action'] == 'HOLD' and no_price and 0.01 < no_price < 0.99:
        no_prob = 1 - est_prob
        no_odds = 1 / no_price - 1
        no_edge = no_prob - no_price
        
        if no_edge >= MIN_EDGE and confidence >= MIN_CONFIDENCE:
            size = kelly_fraction(no_prob, no_odds, confidence)
            if size > 0.001:
                signal['action'] = 'BUY'
                signal['direction'] = 'NO'
                signal['edge'] = no_edge
                signal['position_size'] = size
                signal['reason'] = f"NO edge {no_edge:.1%}"
    
    if signal['action'] == 'HOLD':
        max_edge = max(abs(est_prob - yes_price), abs((1-est_prob) - no_price)) if yes_price and no_price else 0
        signal['reason'] = f"No opportunity (edge={max_edge:.1%}, conf={confidence:.0%})"
    
    return signal

def simulate_outcome(signal, true_prob=None):
    """Simulate trade outcome"""
    if signal['action'] == 'HOLD':
        return 0, False
    
    # Use edge-adjusted probability for simulation
    edge = signal['edge']
    conf = signal['confidence']
    
    if signal['direction'] == 'YES':
        win_prob = signal['yes_price'] + (edge * conf * 0.5)
    else:
        win_prob = signal['no_price'] + (edge * conf * 0.5)
    
    win_prob = max(0.1, min(0.9, win_prob))
    won = random.random() < win_prob
    
    if won:
        price = signal['yes_price'] if signal['direction'] == 'YES' else signal['no_price']
        profit = (1 - price) / price
    else:
        profit = -1
    
    return profit * signal['position_size'], won

def run_analysis():
    """Main analysis loop"""
    print("=" * 70)
    print("POLYMARKET LLM ANALYSIS - DeepSeek vs Random Baseline")
    print("=" * 70)
    print(f"API Key: {'✓ Configured' if DEEPSEEK_API_KEY else '✗ Missing'}")
    print(f"Model: {MODEL}")
    print(f"Parameters: min_edge={MIN_EDGE:.0%}, confidence={MIN_CONFIDENCE:.0%}")
    print("-" * 70)
    
    # Fetch markets
    print("\nFetching markets from Polymarket...")
    markets = fetch_markets()
    print(f"Retrieved {len(markets)} markets")
    
    # Filter quality markets
    quality_markets = []
    for m in markets:
        if m.get('closed'):
            continue
        liquidity = m.get('liquidityNum', 0) or 0
        if liquidity < 1000:
            continue
        yes_p, no_p = parse_prices(m)
        if yes_p and no_p and 0.05 < yes_p < 0.95:
            quality = calculate_market_quality(m)
            m['_quality'] = quality
            m['_yes'] = yes_p
            m['_no'] = no_p
            quality_markets.append(m)
    
    # Sort by quality and take top N
    quality_markets.sort(key=lambda x: x['_quality'], reverse=True)
    selected_markets = quality_markets[:MAX_MARKETS]
    
    print(f"Selected {len(selected_markets)} high-quality markets for analysis")
    print("-" * 70)
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': MODEL,
            'min_edge': MIN_EDGE,
            'min_confidence': MIN_CONFIDENCE,
            'max_kelly': MAX_KELLY,
            'max_position': MAX_POSITION,
            'initial_capital': INITIAL_CAPITAL,
            'markets_analyzed': len(selected_markets)
        },
        'llm_analysis': [],
        'random_analysis': [],
        'comparison': {}
    }
    
    # Run both LLM and Random analysis
    llm_capital = INITIAL_CAPITAL
    random_capital = INITIAL_CAPITAL
    llm_trades = 0
    random_trades = 0
    llm_wins = 0
    random_wins = 0
    
    for i, market in enumerate(selected_markets):
        question = market.get('question', '')[:60]
        yes_price = market['_yes']
        no_price = market['_no']
        quality = market['_quality']
        
        print(f"\n[{i+1}/{len(selected_markets)}] Q={quality} | {question}...")
        
        # LLM Analysis
        llm_result = analyze_with_llm(market, yes_price, no_price)
        llm_signal = generate_signal(market, llm_result, yes_price, no_price)
        llm_pnl, llm_won = simulate_outcome(llm_signal)
        
        if llm_signal['action'] == 'BUY':
            pos_value = llm_capital * llm_signal['position_size']
            llm_pnl_usd = pos_value * llm_pnl
            llm_capital += llm_pnl_usd
            llm_trades += 1
            if llm_won:
                llm_wins += 1
            print(f"  LLM: {llm_signal['direction']} | prob={llm_result['probability']:.0%} | edge={llm_signal['edge']:.1%} | {'WIN' if llm_won else 'LOSS'} ${llm_pnl_usd:+.2f}")
        else:
            print(f"  LLM: HOLD | prob={llm_result['probability']:.0%} | {llm_signal['reason']}")
        
        # Random Analysis
        rand_result = simulate_random_analysis(market, yes_price, no_price)
        rand_signal = generate_signal(market, rand_result, yes_price, no_price)
        rand_pnl, rand_won = simulate_outcome(rand_signal)
        
        if rand_signal['action'] == 'BUY':
            pos_value = random_capital * rand_signal['position_size']
            rand_pnl_usd = pos_value * rand_pnl
            random_capital += rand_pnl_usd
            random_trades += 1
            if rand_won:
                random_wins += 1
            print(f"  RND: {rand_signal['direction']} | prob={rand_result['probability']:.0%} | edge={rand_signal['edge']:.1%} | {'WIN' if rand_won else 'LOSS'} ${rand_pnl_usd:+.2f}")
        else:
            print(f"  RND: HOLD")
        
        # Store results
        results['llm_analysis'].append({
            'market_id': market.get('id'),
            'question': market.get('question', ''),
            'quality': quality,
            'yes_price': yes_price,
            'no_price': no_price,
            'llm_probability': llm_result['probability'],
            'llm_confidence': llm_result['confidence'],
            'llm_reasoning': llm_result['reasoning'],
            'llm_raw_response': llm_result.get('raw_response'),
            'signal': llm_signal['action'],
            'direction': llm_signal['direction'],
            'edge': llm_signal['edge'],
            'position_size': llm_signal['position_size']
        })
        
        results['random_analysis'].append({
            'market_id': market.get('id'),
            'random_probability': rand_result['probability'],
            'signal': rand_signal['action'],
            'direction': rand_signal['direction'],
            'edge': rand_signal['edge']
        })
    
    # Summary
    llm_roi = (llm_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    random_roi = (random_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    results['comparison'] = {
        'llm': {
            'final_capital': llm_capital,
            'total_pnl': llm_capital - INITIAL_CAPITAL,
            'roi': llm_roi,
            'trades': llm_trades,
            'wins': llm_wins,
            'win_rate': llm_wins / llm_trades if llm_trades > 0 else 0
        },
        'random': {
            'final_capital': random_capital,
            'total_pnl': random_capital - INITIAL_CAPITAL,
            'roi': random_roi,
            'trades': random_trades,
            'wins': random_wins,
            'win_rate': random_wins / random_trades if random_trades > 0 else 0
        },
        'llm_advantage': llm_roi - random_roi
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'LLM (DeepSeek)':<20} {'Random Baseline':<20}")
    print("-" * 70)
    print(f"{'Final Capital':<25} ${llm_capital:<19.2f} ${random_capital:<19.2f}")
    print(f"{'Total P&L':<25} ${llm_capital - INITIAL_CAPITAL:<+18.2f} ${random_capital - INITIAL_CAPITAL:<+18.2f}")
    print(f"{'ROI':<25} {llm_roi:<+19.1%} {random_roi:<+19.1%}")
    print(f"{'Trades Taken':<25} {llm_trades:<19} {random_trades:<19}")
    print(f"{'Win Rate':<25} {llm_wins}/{llm_trades if llm_trades else 1:<17} {random_wins}/{random_trades if random_trades else 1:<17}")
    print("-" * 70)
    print(f"LLM Advantage: {llm_roi - random_roi:+.1%}")
    print("=" * 70)
    
    # Save results
    logs_dir = Path(__file__).parent.parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = logs_dir / f"llm_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    run_analysis()
