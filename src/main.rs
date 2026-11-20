//! ベイズ的スペクトル分解 + 交換モンテカルロ法（Nagata, Sugita & Okada 2012 の枠組み）
//! 雑音精度を逆温度とみなして F(K, β) を同時に求める（Tokuda, Nagata & Okada 2017 の考え方）
//!
//! 使い方: cargo run --release -- [K_max] [n_sample] [seed] [data_seed] [n] [n_replica]
//! seed は交換モンテカルロの乱数，data_seed は人工データの雑音の乱数に使う．
//! n はデータ点数（観測範囲 [0, 3] を等間隔に取る），n_replica はレプリカ数．
//! スレッド数は環境変数 RAYON_NUM_THREADS で指定（未指定なら論理コア数）．
//!
//! 出力: data.csv（人工データ），free_energy.csv（格子点ごとの F），
//! free_energy_fine.csv（格子点の間を再重み付けで補間した F(β)），model_selection.csv（K ごとの β̂ と F_min）．

use remc_spectral::{model, remc};

use model::{Gauss, PeakShape, Prior};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

/// free_energy_fine.csv で F(β) を求める β の点数（[β_1, β_max] を対数等間隔に取る）．
const N_FINE_CURVE: usize = 200;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let k_max: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let n_sample: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);
    let data_seed: u64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_120_401);
    let n: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(301);
    let n_replica: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(40);

    // ---- 人工データ: 3 本の重なったピーク，雑音 σ = 0.1（真の β = 100） ----
    let x: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 / (n - 1) as f64).collect();
    let truth = [
        Gauss {
            a: 1.0,
            mu: 1.0,
            b: 1.0 / 0.15f64.powi(2),
        },
        Gauss {
            a: 0.8,
            mu: 1.4,
            b: 1.0 / 0.15f64.powi(2),
        },
        Gauss {
            a: 1.2,
            mu: 2.2,
            b: 1.0 / 0.20f64.powi(2),
        },
    ];
    let sigma = 0.1;
    let mut data_rng = ChaCha8Rng::seed_from_u64(data_seed);
    let y = model::synthesize(&x, &truth, sigma, &mut data_rng);
    {
        let mut w = BufWriter::new(File::create("data.csv")?);
        writeln!(w, "x,y")?;
        for (xi, yi) in x.iter().zip(&y) {
            writeln!(w, "{xi},{yi}")?;
        }
    }

    // ---- 事前分布（デモ用の値．実データでは物理的な知見から設計する） ----
    let prior = Prior {
        a_rate: 1.0, // a ~ Exp(1)
        mu_lo: 0.0,  // μ ~ U(0, 3)（観測範囲内に限定）
        mu_hi: 3.0,
        b_shape: 2.0, // b ~ Gamma(2, rate)，平均 30（幅 ≈ 0.18）
        b_rate: 2.0 / 30.0,
    };

    let cfg = remc::Config {
        n_replica,
        beta_min: 1e-2,
        beta_max: 1e3,
        burn_in: n_sample / 2,
        n_sample,
        adapt_interval: 100,
        target_acc: 0.4,
        recompute_interval: 1000,
        thin: 10,
        seed,
        ladder_interval: 250,
    };

    println!(
        "threads = {}, N = {}, L = {}, sweeps = {} + {}",
        rayon::current_num_threads(),
        n,
        cfg.n_replica,
        cfg.burn_in,
        cfg.n_sample
    );
    println!(
        "true sigma = {sigma}  (true beta = {:.1}), seed = {seed}, data_seed = {data_seed}\n",
        1.0 / (sigma * sigma)
    );

    let mut csv = BufWriter::new(File::create("free_energy.csv")?);
    writeln!(csv, "K,beta,F,mean_nE,exchange_rate")?;
    let mut fine_csv = BufWriter::new(File::create("free_energy_fine.csv")?);
    writeln!(fine_csv, "K,beta,F")?;

    let mut best: Option<(usize, f64, f64)> = None; // (K, F_min, β̂)
    let mut results = Vec::new();

    for k in 1..=k_max {
        let t0 = Instant::now();
        let res = remc::run::<Gauss>(&x, &y, k, &prior, &cfg);
        let lz = remc::log_z(&res.betas, &res.energies);
        let f = remc::free_energy(&res.betas, &lz, n);

        // β について最小化（= 雑音分散の経験ベイズ推定）．格子点の間は再重み付けで補間する．
        // 事後平均の表示には，F が最小の格子点 l̂ のスロットのサンプルを使う．
        let (l_hat, _) = f
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let (beta_hat, f_min) = remc::minimize_free_energy(&res.betas, &res.energies, &lz, n);
        let (min_ex, max_ex) = res
            .exchange_rate
            .iter()
            .fold((1.0f64, 0.0f64), |(lo, hi), &r| (lo.min(r), hi.max(r)));

        println!(
            "K={k}: F_min = {f_min:9.3} at beta = {beta_hat:8.2} (sigma_hat = {:.4}), \
             exchange rate = [{min_ex:.3}, {max_ex:.3}], {:.1} s",
            1.0 / beta_hat.sqrt(),
            t0.elapsed().as_secs_f64()
        );

        // exchange_rate[l] はスロット l と l+1 の間の交換率（最後のスロットは NaN）
        let rows = res
            .betas
            .iter()
            .zip(&f)
            .zip(&res.energies)
            .enumerate()
            .skip(1);
        for (l, ((beta, fl), e)) in rows {
            let mean_e = e.iter().sum::<f64>() / e.len() as f64;
            let ex = res.exchange_rate.get(l).copied().unwrap_or(f64::NAN);
            writeln!(csv, "{k},{beta},{fl},{mean_e},{ex}")?;
        }

        // 格子点の間も含めた F(β) の曲線．β_0 = 0 では F が発散するので β_1 から取る．
        let (b_lo, b_hi) = (res.betas[1], *res.betas.last().unwrap());
        let (llo, lhi) = (b_lo.ln(), b_hi.ln());
        for i in 0..=N_FINE_CURVE {
            let beta = (llo + (lhi - llo) * i as f64 / N_FINE_CURVE as f64)
                .exp()
                .clamp(b_lo, b_hi);
            let fb = 0.5 * n as f64 * (2.0 * std::f64::consts::PI / beta).ln()
                - remc::log_z_at(&res.betas, &res.energies, &lz, beta);
            writeln!(fine_csv, "{k},{beta},{fb}")?;
        }

        let improved = match best {
            Some((_, f_best, _)) => f_min < f_best,
            None => true,
        };
        if improved {
            best = Some((k, f_min, beta_hat));
        }
        results.push((res, l_hat, f_min, beta_hat));
    }

    // ---- 最小値からの差 ΔF(K)．1 nat 程度の差はモデル選択の根拠としては弱い ----
    let f_best = best.unwrap().1;
    let mut sel_csv = BufWriter::new(File::create("model_selection.csv")?);
    writeln!(sel_csv, "K,beta_hat,sigma_hat,F_min,delta_F")?;
    println!("\nDelta F (K) = F_min(K) - min_K F_min(K):");
    for (k, (_, _, fm, bh)) in results.iter().enumerate() {
        println!("  K={}: {:8.3}", k + 1, fm - f_best);
        writeln!(
            sel_csv,
            "{},{bh},{},{fm},{}",
            k + 1,
            1.0 / bh.sqrt(),
            fm - f_best
        )?;
    }

    // ---- 選ばれたモデルの事後平均（β̂ のスロットのサンプルから） ----
    let (k_hat, _, beta_hat) = best.unwrap();
    let (res, l_hat, _, _) = &results[k_hat - 1];
    println!("\nargmin K = {k_hat}, beta_hat = {beta_hat:.2}");
    println!(
        "posterior mean ± sd at grid beta = {:.2} nearest to the F minimum (peaks sorted by center):",
        res.betas[*l_hat]
    );
    let s = &res.samples[*l_hat];
    for k in 0..k_hat {
        // パラメータ j の事後平均と標準偏差
        let stats: Vec<String> = Gauss::PARAM_NAMES
            .iter()
            .enumerate()
            .map(|(j, name)| {
                let v: Vec<f64> = s.iter().map(|p| p[k].get(j)).collect();
                let m = v.iter().sum::<f64>() / v.len() as f64;
                let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
                format!("{name} = {m:.3}±{sd:.3}")
            })
            .collect();
        println!("  peak {k}: {}", stats.join(", "));
    }
    println!(
        "acceptance ({}) at that grid beta: {:.2?}",
        Gauss::PARAM_NAMES.join(", "),
        res.acceptance[*l_hat]
    );
    println!("\nwrote free_energy.csv, free_energy_fine.csv, model_selection.csv");
    Ok(())
}
