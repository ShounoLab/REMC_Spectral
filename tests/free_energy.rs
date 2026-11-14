//! 結合テスト: K = 1 のベイズ自由エネルギーを数値積分と比較する
//!
//! K = 1 ならパラメータは (a, μ, b) の 3 次元なので，格子上の数値積分で
//! Z(β) = ∫ exp(-β nE(w)) φ(w) dw を直接計算できる．
//! 交換モンテカルロの推定値がこれと一致することを確かめる．

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use remc_spectral::model::{synthesize, Peak, Prior};
use remc_spectral::remc::{free_energy, log_z, run, Config};

fn log_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        m
    } else {
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}

/// 格子積分による log Z(β)（事前分布の正規化定数込み）
fn log_z_quadrature(x: &[f64], y: &[f64], prior: &Prior, betas: &[f64]) -> Vec<f64> {
    let (na, nmu, nb) = (400, 300, 300);
    let a_max = 4.0;
    let da = a_max / na as f64;
    let dmu = (prior.mu_hi - prior.mu_lo) / nmu as f64;
    let (lb_lo, lb_hi) = (0.05f64.ln(), 2000f64.ln());
    let dlb = (lb_hi - lb_lo) / nb as f64;
    let syy: f64 = y.iter().map(|v| v * v).sum();

    // 正規化された事前密度の対数
    let log_p_a = |a: f64| prior.a_rate.ln() - prior.a_rate * a;
    let log_p_mu = -(prior.mu_hi - prior.mu_lo).ln();
    // Gamma(shape=2): ln Γ(2) = 0．log b 格子なので Jacobian (+ ln b) を加える
    assert_eq!(prior.b_shape, 2.0);
    let log_p_lb = |b: f64| 2.0 * prior.b_rate.ln() + b.ln() - prior.b_rate * b + b.ln();

    let mut acc = vec![f64::NEG_INFINITY; betas.len()];
    for ib in 0..nb {
        let b = (lb_lo + (ib as f64 + 0.5) * dlb).exp();
        for imu in 0..nmu {
            let mu = prior.mu_lo + (imu as f64 + 0.5) * dmu;
            // SSE(a) = Syy - 2a Sgy + a² Sgg
            let (mut sgy, mut sgg) = (0.0, 0.0);
            for (&xi, &yi) in x.iter().zip(y) {
                let g = (-0.5 * b * (xi - mu).powi(2)).exp();
                sgy += g * yi;
                sgg += g * g;
            }
            for ia in 0..na {
                let a = (ia as f64 + 0.5) * da;
                let sse = syy - 2.0 * a * sgy + a * a * sgg;
                let lp = log_p_a(a) + log_p_mu + log_p_lb(b);
                for (s, &beta) in acc.iter_mut().zip(betas) {
                    *s = log_add(*s, -0.5 * beta * sse + lp);
                }
            }
        }
    }
    let log_cell = (da * dmu * dlb).ln();
    acc.iter().map(|s| s + log_cell).collect()
}

#[test]
fn k1_free_energy_matches_quadrature() {
    let n = 151;
    let x: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 / (n - 1) as f64).collect();
    let truth = [Peak {
        a: 1.0,
        mu: 1.3,
        b: 30.0,
    }];
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let y = synthesize(&x, &truth, 0.1, &mut rng);
    let prior = Prior {
        a_rate: 1.0,
        mu_lo: 0.0,
        mu_hi: 3.0,
        b_shape: 2.0,
        b_rate: 2.0 / 30.0,
    };

    let cfg = Config {
        n_replica: 24,
        beta_min: 1e-1,
        beta_max: 3e2,
        burn_in: 2_000,
        n_sample: 8_000,
        adapt_interval: 100,
        target_acc: 0.4,
        recompute_interval: 1_000,
        thin: 10,
        seed: 11,
    };
    let res = run(&x, &y, 1, &prior, &cfg);
    let f_mc = free_energy(&res.betas, &log_z(&res.betas, &res.energies), n);

    // 中間〜低温側のいくつかの β で比較
    let idx = [6, 12, 18, 23];
    let betas: Vec<f64> = idx.iter().map(|&l| res.betas[l]).collect();
    let lz_q = log_z_quadrature(&x, &y, &prior, &betas);
    let f_q = free_energy(&betas, &lz_q, n);

    for (j, &l) in idx.iter().enumerate() {
        let diff = (f_mc[l] - f_q[j]).abs();
        println!(
            "beta = {:8.3}: MC {:9.3}  quad {:9.3}  |diff| {:.3}",
            betas[j], f_mc[l], f_q[j], diff
        );
        assert!(diff < 0.3, "free energy mismatch at beta = {}", betas[j]);
    }
}
