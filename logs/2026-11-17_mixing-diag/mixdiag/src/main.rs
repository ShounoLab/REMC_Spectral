//! 各スロットのエネルギー列の積分自己相関時間 τ_int（Sokal の窓，c = 6）を測る．
//! 使い方: mixdiag n data_seed K [seed]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use remc_spectral::model::{self, Peak, Prior};
use remc_spectral::remc;

fn tau_int(v: &[f64]) -> f64 {
    let n = v.len();
    let m = v.iter().sum::<f64>() / n as f64;
    let d: Vec<f64> = v.iter().map(|x| x - m).collect();
    let c0 = d.iter().map(|x| x * x).sum::<f64>() / n as f64;
    if c0 == 0.0 {
        return f64::NAN;
    }
    let mut tau = 1.0;
    for t in 1..n / 2 {
        let ct = d.iter().zip(&d[t..]).map(|(a, b)| a * b).sum::<f64>() / n as f64;
        tau += 2.0 * ct / c0;
        if t as f64 >= 6.0 * tau {
            return tau;
        }
    }
    f64::INFINITY // 窓が閉じない＝列が短すぎる
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a[1].parse().unwrap();
    let data_seed: u64 = a[2].parse().unwrap();
    let k: usize = a[3].parse().unwrap();
    let seed: u64 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(42);
    // main.rs と同じデータ・事前分布・設定
    let x: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 / (n - 1) as f64).collect();
    let truth = [
        Peak { a: 1.0, mu: 1.0, b: 1.0 / 0.15f64.powi(2) },
        Peak { a: 0.8, mu: 1.4, b: 1.0 / 0.15f64.powi(2) },
        Peak { a: 1.2, mu: 2.2, b: 1.0 / 0.20f64.powi(2) },
    ];
    let y = model::synthesize(&x, &truth, 0.1, &mut ChaCha8Rng::seed_from_u64(data_seed));
    let prior = Prior { a_rate: 1.0, mu_lo: 0.0, mu_hi: 3.0, b_shape: 2.0, b_rate: 2.0 / 30.0 };
    let cfg = remc::Config {
        n_replica: 40, beta_min: 1e-2, beta_max: 1e3, burn_in: 10_000, n_sample: 20_000,
        adapt_interval: 100, target_acc: 0.4, recompute_interval: 1000, thin: 10, seed,
        ladder_interval: 250,
    };
    let res = remc::run(&x, &y, k, &prior, &cfg);
    println!("l,beta,tau_int,ess,exch_to_next");
    for (l, (b, e)) in res.betas.iter().zip(&res.energies).enumerate() {
        let t = tau_int(e);
        let ex = res.exchange_rate.get(l).copied().unwrap_or(f64::NAN);
        println!("{l},{b:.4},{t:.1},{:.0},{ex:.3}", e.len() as f64 / t);
    }
}
