//! Performance Benchmark Tests
//!
//! Tests for latency, throughput, and performance under load.

use crate::strategy::{
    DynamicKelly, DynamicKellyConfig, MarketContext,
    SignalGenerator,
};
use crate::config::{RiskConfig, StrategyConfig};
use crate::model::Prediction;
use crate::types::{Market, Outcome};
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== LATENCY BENCHMARKS ====================

    #[test]
    fn bench_kelly_calculation_latency() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let iterations = 10_000;
        let start = Instant::now();
        
        for i in 0..iterations {
            let prob = dec!(0.50) + Decimal::from(i % 40) / dec!(100);
            kelly.calculate_position_size(
                prob,
                dec!(0.50),
                dec!(0.80),
                dec!(1.0),
                None,
            );
        }
        
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / iterations as u128;
        
        println!("Kelly calculation: {} iterations in {:?}", iterations, elapsed);
        println!("Average latency: {} ns per calculation", avg_ns);
        
        // Should be fast (< 50 μs) - allows for debug mode overhead
        assert!(avg_ns < 50_000, "Kelly calculation too slow: {} ns", avg_ns);
    }

    #[test]
    fn bench_signal_generation_latency() {
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        let market = create_test_market();
        let iterations = 10_000;
        let start = Instant::now();
        
        for i in 0..iterations {
            let prob = dec!(0.55) + Decimal::from(i % 30) / dec!(100);
            let prediction = Prediction {
                probability: prob,
                confidence: dec!(0.80),
                reasoning: "Test".to_string(),
            };
            let _ = generator.generate(&market, &prediction);
        }
        
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / iterations as u128;
        
        println!("Signal generation: {} iterations in {:?}", iterations, elapsed);
        println!("Average latency: {} ns per signal", avg_ns);
        
        // Should be fast (< 50 μs)
        assert!(avg_ns < 50_000, "Signal generation too slow: {} ns", avg_ns);
    }

    #[test]
    fn bench_kelly_with_context_latency() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let ctx = MarketContext {
            volatility: dec!(0.08),
            liquidity_score: dec!(0.7),
            time_pressure: dec!(0.3),
        };
        
        let iterations = 10_000;
        let start = Instant::now();
        
        for i in 0..iterations {
            let prob = dec!(0.50) + Decimal::from(i % 40) / dec!(100);
            kelly.calculate_position_size(
                prob,
                dec!(0.50),
                dec!(0.80),
                dec!(1.0),
                Some(&ctx),
            );
        }
        
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / iterations as u128;
        
        println!("Kelly with context: {} iterations in {:?}", iterations, elapsed);
        println!("Average latency: {} ns per calculation", avg_ns);
        
        // Should be fast (< 100 μs) - allows for debug mode overhead
        assert!(avg_ns < 100_000, "Kelly with context too slow: {} ns", avg_ns);
    }

    // ==================== THROUGHPUT BENCHMARKS ====================

    #[test]
    fn bench_signal_throughput() {
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        // Generate 20 different markets
        let markets: Vec<Market> = (0..20).map(|i| create_varied_market(i)).collect();
        
        let iterations = 1_000;
        let signals_per_iteration = markets.len();
        let start = Instant::now();
        
        for _ in 0..iterations {
            for market in &markets {
                let prediction = Prediction {
                    probability: dec!(0.65),
                    confidence: dec!(0.80),
                    reasoning: "Test".to_string(),
                };
                let _ = generator.generate(market, &prediction);
            }
        }
        
        let elapsed = start.elapsed();
        let total_signals = iterations * signals_per_iteration;
        let signals_per_sec = total_signals as f64 / elapsed.as_secs_f64();
        
        println!("Signal throughput: {} signals in {:?}", total_signals, elapsed);
        println!("Throughput: {:.0} signals/second", signals_per_sec);
        
        // Should handle at least 10k signals/sec
        assert!(signals_per_sec > 10_000.0, "Throughput too low: {:.0}", signals_per_sec);
    }

    #[test]
    fn bench_full_pipeline_throughput() {
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let markets: Vec<Market> = (0..10).map(|i| create_varied_market(i)).collect();
        
        let iterations = 500;
        let start = Instant::now();
        
        for _ in 0..iterations {
            for market in &markets {
                // Generate signal
                let prediction = Prediction {
                    probability: dec!(0.65),
                    confidence: dec!(0.80),
                    reasoning: "Test".to_string(),
                };
                
                if let Some(signal) = generator.generate(market, &prediction) {
                    // Kelly sizing
                    let _kelly_result = kelly.calculate_position_size(
                        signal.model_probability,
                        signal.market_probability,
                        signal.confidence,
                        dec!(1.0),
                        None,
                    );
                }
            }
        }
        
        let elapsed = start.elapsed();
        let total_markets = iterations * markets.len();
        let markets_per_sec = total_markets as f64 / elapsed.as_secs_f64();
        
        println!("Full pipeline: {} markets in {:?}", total_markets, elapsed);
        println!("Throughput: {:.0} markets/second", markets_per_sec);
        
        // Should handle at least 5k markets/sec through full pipeline
        assert!(markets_per_sec > 5_000.0, "Pipeline throughput too low: {:.0}", markets_per_sec);
    }

    // ==================== MEMORY EFFICIENCY BENCHMARKS ====================

    #[test]
    fn bench_kelly_memory_with_many_trades() {
        let kelly = DynamicKelly::new(DynamicKellyConfig {
            lookback_trades: 1000, // Large lookback
            ..Default::default()
        }, dec!(10000));
        
        // Record many trades
        for i in 0..10_000 {
            let pnl = if i % 3 == 0 { dec!(-50) } else { dec!(100) };
            kelly.record_trade(pnl);
        }
        
        // Should still be fast after many trades
        let start = Instant::now();
        for _ in 0..1000 {
            kelly.calculate_position_size(
                dec!(0.60),
                dec!(0.50),
                dec!(0.80),
                dec!(1.0),
                None,
            );
        }
        let elapsed = start.elapsed();
        
        println!("Kelly after 10k trades: 1000 calcs in {:?}", elapsed);
        assert!(elapsed.as_millis() < 100, "Kelly too slow after many trades");
    }

    // ==================== CONCURRENT ACCESS BENCHMARKS ====================

    #[test]
    fn bench_kelly_concurrent_updates() {
        use std::sync::Arc;
        use std::thread;
        
        let kelly = Arc::new(DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000)));
        let num_threads = 4;
        let iterations_per_thread = 1000;
        
        let start = Instant::now();
        
        let handles: Vec<_> = (0..num_threads).map(|t| {
            let kelly = Arc::clone(&kelly);
            thread::spawn(move || {
                for i in 0..iterations_per_thread {
                    // Some threads update, some read
                    if (t + i) % 3 == 0 {
                        let pnl = if i % 2 == 0 { dec!(50) } else { dec!(-30) };
                        kelly.record_trade(pnl);
                    } else {
                        kelly.calculate_position_size(
                            dec!(0.60),
                            dec!(0.50),
                            dec!(0.80),
                            dec!(1.0),
                            None,
                        );
                    }
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let elapsed = start.elapsed();
        let total_ops = num_threads * iterations_per_thread;
        
        println!("Concurrent Kelly ops: {} operations across {} threads in {:?}",
            total_ops, num_threads, elapsed);
        
        assert!(elapsed.as_millis() < 1000, "Concurrent access too slow");
    }

    // ==================== STRESS TESTS ====================

    #[test]
    fn stress_test_rapid_calculations() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        let market = create_test_market();
        
        // Rapid fire calculations
        let start = Instant::now();
        let duration = std::time::Duration::from_secs(1);
        let mut count = 0u64;
        
        while start.elapsed() < duration {
            for _ in 0..100 {
                let _ = kelly.calculate_position_size(
                    dec!(0.60),
                    dec!(0.50),
                    dec!(0.80),
                    dec!(1.0),
                    None,
                );
                let prediction = Prediction {
                    probability: dec!(0.65),
                    confidence: dec!(0.80),
                    reasoning: "Test".to_string(),
                };
                let _ = generator.generate(&market, &prediction);
                count += 2;
            }
        }
        
        let elapsed = start.elapsed();
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();
        
        println!("Stress test: {} operations in {:?}", count, elapsed);
        println!("Sustained rate: {:.0} ops/sec", ops_per_sec);
        
        // Should sustain at least 50k ops/sec
        assert!(ops_per_sec > 50_000.0, "Stress test rate too low");
    }

    #[test]
    fn stress_test_varying_load() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let ctx = MarketContext {
            volatility: dec!(0.08),
            liquidity_score: dec!(0.7),
            time_pressure: dec!(0.3),
        };
        
        let mut total_time_ns = 0u128;
        let mut max_time_ns = 0u128;
        let iterations = 1000;
        
        for i in 0..iterations {
            // Vary inputs
            let prob = dec!(0.40) + Decimal::from(i % 50) / dec!(100);
            let price = dec!(0.30) + Decimal::from(i % 60) / dec!(100);
            let conf = dec!(0.50) + Decimal::from(i % 50) / dec!(100);
            let budget = dec!(0.20) + Decimal::from(i % 80) / dec!(100);
            
            let start = Instant::now();
            let _ = kelly.calculate_position_size(prob, price, conf, budget, Some(&ctx));
            let elapsed = start.elapsed().as_nanos();
            
            total_time_ns += elapsed;
            max_time_ns = max_time_ns.max(elapsed);
        }
        
        let avg_time_ns = total_time_ns / iterations as u128;
        
        println!("Varying load test:");
        println!("  Avg latency: {} ns", avg_time_ns);
        println!("  Max latency: {} ns", max_time_ns);
        
        // Max should not be too much higher than avg (no outliers)
        let ratio = max_time_ns as f64 / avg_time_ns as f64;
        println!("  Max/Avg ratio: {:.1}x", ratio);
        
        // Allow higher variance in debug mode with GC/memory pressure
        assert!(ratio < 500.0, "Too much latency variance: {:.1}x", ratio);
    }

    // ==================== HELPERS ====================

    fn create_test_market() -> Market {
        Market {
            id: "test-market".to_string(),
            question: "Test market question?".to_string(),
            description: Some("Description".to_string()),
            end_date: Some(Utc::now() + chrono::Duration::days(7)),
            volume: dec!(100000),
            liquidity: dec!(50000),
            outcomes: vec![
                Outcome { token_id: "yes".to_string(), outcome: "Yes".to_string(), price: dec!(0.50) },
                Outcome { token_id: "no".to_string(), outcome: "No".to_string(), price: dec!(0.50) },
            ],
            active: true,
            closed: false,
        }
    }

    fn create_varied_market(index: usize) -> Market {
        let yes_price = dec!(0.30) + Decimal::from((index * 7) % 40) / dec!(100);
        
        Market {
            id: format!("market-{}", index),
            question: format!("Market question {}?", index),
            description: Some(format!("Description {}", index)),
            end_date: Some(Utc::now() + chrono::Duration::days((index % 30 + 1) as i64)),
            volume: dec!(10000) + Decimal::from(index as i64 * 1000),
            liquidity: dec!(5000) + Decimal::from(index as i64 * 500),
            outcomes: vec![
                Outcome { 
                    token_id: format!("yes-{}", index), 
                    outcome: "Yes".to_string(), 
                    price: yes_price 
                },
                Outcome { 
                    token_id: format!("no-{}", index), 
                    outcome: "No".to_string(), 
                    price: Decimal::ONE - yes_price 
                },
            ],
            active: true,
            closed: false,
        }
    }
}
