//! レプリカ交換モンテカルロ法（交換モンテカルロ法）の本体
//!
//! 設計方針:
//!   - 「温度スロット」(Slot) と「状態」(Chain) を分ける．
//!     逆温度・ステップ幅・乱数生成器はスロットに固定し，交換時は Chain だけを swap する．
//!   - Chain の中身は Vec（ヒープへのポインタ＋長さ＋容量）なので，swap は O(1)．
//!     実質的に「温度ラベルの付け替え」と同じコストで済む．
//!   - Metropolis 更新はスロットごとに独立なので rayon で並列化．交換は O(L) の逐次処理．
//!   - 乱数はスロットごとに ChaCha のストリームを割り当てるので，
//!     スレッド数やスケジューリングに依存せず結果がビット単位で再現する．

use crate::model::{Peak, Prior, PARAMS};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// 温度間を移動する状態
pub struct Chain {
    pub peaks: Vec<Peak>,
    /// 各ピークの寄与 g_k(x_i)．K×N を行優先で格納（ピーク k の行が連続）．
    contrib: Vec<f64>,
    /// 残差 r_i = y_i - Σ_k g_k(x_i)
    resid: Vec<f64>,
    /// Σ_i r_i²．エネルギー nE_n(w) = sse / 2．
    sse: f64,
}

impl Chain {
    fn new(peaks: Vec<Peak>, x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        let mut c = Chain {
            contrib: vec![0.0; peaks.len() * n],
            resid: vec![0.0; n],
            sse: 0.0,
            peaks,
        };
        c.recompute(x, y);
        c
    }

    /// 残差と SSE を一から計算し直す．差分更新の丸め誤差の蓄積を定期的にリセットするため．
    fn recompute(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len();
        self.resid.copy_from_slice(y);
        // chunks_exact_mut: contrib を長さ n の行ごとに可変参照として切り出す
        for (row, p) in self.contrib.chunks_exact_mut(n).zip(&self.peaks) {
            for ((g, &xi), r) in row.iter_mut().zip(x).zip(self.resid.iter_mut()) {
                *g = p.eval(xi);
                *r -= *g;
            }
        }
        self.sse = self.resid.iter().map(|r| r * r).sum();
    }

    /// nE_n(w) = (1/2) Σ_i (y_i - f(x_i; w))²
    #[inline]
    pub fn energy(&self) -> f64 {
        0.5 * self.sse
    }
}

/// 温度スロット（動かないもの一式）
pub struct Slot {
    pub beta: f64,
    pub chain: Chain,
    rng: ChaCha8Rng,
    step: [f64; 3],
    acc: [u64; 3],
    tried: [u64; 3],
    /// 提案ピークの寄与を書き込む作業領域（更新ごとのメモリ確保を避ける）
    scratch: Vec<f64>,
}

impl Slot {
    fn new(beta: f64, chain: Chain, rng: ChaCha8Rng, step: [f64; 3]) -> Self {
        let n = chain.resid.len();
        Slot {
            beta,
            chain,
            rng,
            step,
            acc: [0; 3],
            tried: [0; 3],
            scratch: vec![0.0; n],
        }
    }

    /// 全パラメータを 1 回ずつ単一成分 Metropolis 更新する（1 スイープ）
    fn sweep(&mut self, x: &[f64], prior: &Prior) {
        let n = x.len();
        for k in 0..self.chain.peaks.len() {
            for &p in &PARAMS {
                let j = p as usize;
                self.tried[j] += 1;

                let old = self.chain.peaks[k].get(p);
                let z: f64 = self.rng.sample(StandardNormal);
                let new = old + self.step[j] * z;

                let d_prior = prior.log_density(p, new) - prior.log_density(p, old);
                if d_prior == f64::NEG_INFINITY {
                    continue; // 台の外（a ≤ 0 など）は即棄却
                }

                let mut cand = self.chain.peaks[k];
                cand.set(p, new);

                // 差分更新: ピーク k の寄与だけを差し替えた SSE を O(N) で計算
                //   r_i' = r_i + g_k^old(x_i) - g_k^new(x_i)
                let g_old = &self.chain.contrib[k * n..(k + 1) * n];
                let mut sse_new = 0.0;
                for (((s, &xi), &r), &go) in self
                    .scratch
                    .iter_mut()
                    .zip(x)
                    .zip(&self.chain.resid)
                    .zip(g_old)
                {
                    let g = cand.eval(xi);
                    *s = g;
                    let rn = r + go - g;
                    sse_new += rn * rn;
                }

                // 目標分布 ∝ exp(-β nE_n(w)) φ(w)
                let log_ratio = -0.5 * self.beta * (sse_new - self.chain.sse) + d_prior;
                if log_ratio >= 0.0 || self.rng.gen::<f64>().ln() < log_ratio {
                    // 受理．同じ構造体の異なるフィールドへの可変借用は同時に取れる
                    // （resid と contrib は disjoint なので borrow checker が許す）．
                    let row = &mut self.chain.contrib[k * n..(k + 1) * n];
                    for ((r, g), &gn) in self.chain.resid.iter_mut().zip(row).zip(&self.scratch) {
                        *r += *g - gn;
                        *g = gn;
                    }
                    self.chain.sse = sse_new;
                    self.chain.peaks[k] = cand;
                    self.acc[j] += 1;
                }
            }
        }
    }

    /// burn-in 中だけステップ幅を受理率に合わせて調整する（本サンプリング中は固定）
    fn adapt(&mut self, target: f64) {
        for j in 0..3 {
            if self.tried[j] == 0 {
                continue;
            }
            let rate = self.acc[j] as f64 / self.tried[j] as f64;
            self.step[j] *= if rate > target { 1.2 } else { 1.0 / 1.2 };
            self.step[j] = self.step[j].clamp(1e-8, 1e4);
            self.acc[j] = 0;
            self.tried[j] = 0;
        }
    }

    pub fn acceptance(&self) -> [f64; 3] {
        // std::array::from_fn（Rust 1.63+）: 添字から固定長配列を直接作る
        std::array::from_fn(|j| self.acc[j] as f64 / self.tried[j].max(1) as f64)
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub n_replica: usize,
    pub beta_min: f64,
    pub beta_max: f64,
    pub burn_in: usize,
    pub n_sample: usize,
    pub adapt_interval: usize,
    pub target_acc: f64,
    pub recompute_interval: usize,
    pub thin: usize,
    pub seed: u64,
    /// 逆温度列を調整する間隔（スイープ数）．0 なら調整しない．
    /// 調整は burn-in の前半だけで行い，後半は固定した列のもとで平衡化させる（[`adapt_ladder`]）．
    pub ladder_interval: usize,
}

/// [`adapt_ladder`] の利得 κ．交換率の偏差 0.1 につき間隔を約 10 % 伸縮させる．
const LADDER_GAIN: f64 = 1.0;

pub struct RemcResult {
    pub betas: Vec<f64>,
    /// `energies[l][t]`: スロット l の t 番目のサンプルの nE_n(w)
    pub energies: Vec<Vec<f64>>,
    /// `samples[l][t]`: 間引いたピーク列（μ の昇順に並べ替え済み）
    pub samples: Vec<Vec<Vec<Peak>>>,
    /// `exchange_rate[l]`: スロット l と l+1 の間の交換受理率
    pub exchange_rate: Vec<f64>,
    pub acceptance: Vec<[f64; 3]>,
}

/// 逆温度列: β_0 = 0（事前分布），β_1..β_{L-1} は β_min から β_max まで等比
pub fn ladder(l: usize, beta_min: f64, beta_max: f64) -> Vec<f64> {
    assert!(l >= 3);
    let gamma = (beta_max / beta_min).powf(1.0 / (l - 2) as f64);
    std::iter::once(0.0)
        .chain((0..l - 1).map(|i| beta_min * gamma.powi(i as i32)))
        .collect()
}

/// 隣接ペアの交換受理率 `rates`（長さ L - 1）がそろうように逆温度列を調整する．
///
/// β_0 = 0 と最高の逆温度 β_{L-1} は固定する．間隔 d_l = β_{l+1} - β_l を
///   log d_l ← log d_l + κ (A_l - Ā)，Ā は A_l の平均
/// で更新したあと，Σ d_l = β_{L-1} となるよう全体を定数倍する．
/// 受理率の高いペアは間隔が広がり，低いペア（交換のボトルネック）は狭まる．
/// 間隔は常に正なので列の単調性は保たれる．
///
/// 列を変えると詳細釣り合いの前提が崩れるので，burn-in 中にだけ呼ぶこと．
pub fn adapt_ladder(betas: &[f64], rates: &[f64], gain: f64) -> Vec<f64> {
    assert_eq!(betas[0], 0.0);
    assert_eq!(rates.len() + 1, betas.len());
    let beta_top = betas[betas.len() - 1];
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let gaps: Vec<f64> = betas
        .windows(2)
        .zip(rates)
        .map(|(w, &a)| (w[1] - w[0]) * (gain * (a - mean)).exp())
        .collect();
    let scale = beta_top / gaps.iter().sum::<f64>();
    let mut out: Vec<f64> = std::iter::once(0.0)
        .chain(gaps.iter().scan(0.0, |acc, &d| {
            *acc += d * scale;
            Some(*acc)
        }))
        .collect();
    // 丸め誤差で端がずれないよう，最高の逆温度は元の値に戻す
    *out.last_mut().unwrap() = beta_top;
    out
}

/// 隣接ペアの交換．parity で偶数ペア (0,1),(2,3),… と奇数ペア (1,2),(3,4),… を交互に試す
/// （deterministic even-odd 方式）．
fn exchange(
    slots: &mut [Slot],
    parity: usize,
    rng: &mut ChaCha8Rng,
    acc: &mut [u64],
    tried: &mut [u64],
) {
    let mut l = parity;
    while l + 1 < slots.len() {
        // split_at_mut で隣り合う 2 要素への可変参照を同時に得る（borrow checker 対策の定番）
        let (lo, hi) = slots.split_at_mut(l + 1);
        let (a, b) = (&mut lo[l], &mut hi[0]);
        // 受理確率 min(1, exp((β_{l+1} - β_l)(E(w_{l+1}) - E(w_l))))．事前分布は打ち消し合う．
        let log_r = (b.beta - a.beta) * (b.chain.energy() - a.chain.energy());
        tried[l] += 1;
        if log_r >= 0.0 || rng.gen::<f64>().ln() < log_r {
            std::mem::swap(&mut a.chain, &mut b.chain);
            acc[l] += 1;
        }
        l += 2;
    }
}

/// K 本のピークをもつモデルで交換モンテカルロを実行する
pub fn run(x: &[f64], y: &[f64], k: usize, prior: &Prior, cfg: &Config) -> RemcResult {
    let mut betas = ladder(cfg.n_replica, cfg.beta_min, cfg.beta_max);
    let l_max = betas.len();

    // スロット l には (seed, stream = K * 2^20 + l) の独立ストリームを割り当てる
    let mut slots: Vec<Slot> = betas
        .iter()
        .enumerate()
        .map(|(l, &beta)| {
            let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
            rng.set_stream(((k as u64) << 20) + l as u64);
            let peaks: Vec<Peak> = (0..k).map(|_| prior.sample_peak(&mut rng)).collect();
            Slot::new(beta, Chain::new(peaks, x, y), rng, prior.initial_steps())
        })
        .collect();

    let mut ex_rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    ex_rng.set_stream(u64::MAX - k as u64);
    let mut ex_acc = vec![0u64; l_max - 1];
    let mut ex_tried = vec![0u64; l_max - 1];

    let mut energies = vec![Vec::with_capacity(cfg.n_sample); l_max];
    let mut samples = vec![Vec::with_capacity(cfg.n_sample / cfg.thin + 1); l_max];

    for t in 0..cfg.burn_in + cfg.n_sample {
        // ---- 並列部分: 各スロットの Metropolis スイープ ----
        slots.par_iter_mut().for_each(|s| s.sweep(x, prior));

        // ---- 逐次部分: レプリカ交換 ----
        exchange(&mut slots, t % 2, &mut ex_rng, &mut ex_acc, &mut ex_tried);

        if (t + 1).is_multiple_of(cfg.recompute_interval) {
            slots.par_iter_mut().for_each(|s| s.chain.recompute(x, y));
        }

        if t < cfg.burn_in {
            if (t + 1).is_multiple_of(cfg.adapt_interval) {
                slots.iter_mut().for_each(|s| s.adapt(cfg.target_acc));
            }
            // 逆温度列の調整は burn-in の前半だけ．交換の統計はこの区間の窓ごとにリセットする．
            if cfg.ladder_interval > 0
                && t < cfg.burn_in / 2
                && (t + 1).is_multiple_of(cfg.ladder_interval)
            {
                let rates: Vec<f64> = ex_acc
                    .iter()
                    .zip(&ex_tried)
                    .map(|(&a, &m)| a as f64 / m.max(1) as f64)
                    .collect();
                betas = adapt_ladder(&betas, &rates, LADDER_GAIN);
                for (s, &b) in slots.iter_mut().zip(&betas) {
                    s.beta = b;
                }
                ex_acc.iter_mut().for_each(|v| *v = 0);
                ex_tried.iter_mut().for_each(|v| *v = 0);
            }
            if t + 1 == cfg.burn_in {
                // 本サンプリング開始時に統計をリセット
                ex_acc.iter_mut().for_each(|v| *v = 0);
                ex_tried.iter_mut().for_each(|v| *v = 0);
                slots.iter_mut().for_each(|s| {
                    s.acc = [0; 3];
                    s.tried = [0; 3];
                });
            }
        } else {
            let ts = t - cfg.burn_in;
            for (l, s) in slots.iter().enumerate() {
                energies[l].push(s.chain.energy());
                if ts.is_multiple_of(cfg.thin) {
                    // ラベルスイッチングに対処するため μ の昇順に並べてから保存
                    let mut p = s.chain.peaks.clone();
                    p.sort_by(|u, v| u.mu.total_cmp(&v.mu));
                    samples[l].push(p);
                }
            }
        }
    }

    RemcResult {
        exchange_rate: ex_acc
            .iter()
            .zip(&ex_tried)
            .map(|(&a, &t)| a as f64 / t.max(1) as f64)
            .collect(),
        acceptance: slots.iter().map(|s| s.acceptance()).collect(),
        betas,
        energies,
        samples,
    }
}

/// log mean exp を桁あふれなしに計算
fn log_mean_exp(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = v.clone().fold(f64::NEG_INFINITY, f64::max);
    let (s, cnt) = v.fold((0.0, 0usize), |(s, c), x| (s + (x - m).exp(), c + 1));
    m + (s / cnt as f64).ln()
}

/// log Z(β_l)，ただし Z(β) = ∫ exp(-β nE_n(w)) φ(w) dw，Z(0) = 1．
/// 隣接温度の比の積: log Z(β_{l+1}) - log Z(β_l) = log ⟨exp(-(β_{l+1} - β_l) nE_n)⟩_{β_l}
pub fn log_z(betas: &[f64], energies: &[Vec<f64>]) -> Vec<f64> {
    let mut lz = vec![0.0; betas.len()];
    for l in 0..betas.len() - 1 {
        let d = betas[l + 1] - betas[l];
        lz[l + 1] = lz[l] + log_mean_exp(energies[l].iter().map(move |&e| -d * e));
    }
    lz
}

/// 格子点以外の β での log Z(β)（0 ≤ β ≤ β_{L-1}）．
///
/// β_l ≤ β < β_{l+1} となる l をとり，スロット l のサンプルを再重み付けする:
///   log Z(β) = log Z(β_l) + log ⟨exp(-(β - β_l) nE_n)⟩_{β_l}
/// [`log_z`] の隣接比と同じ推定量なので，格子点では `lz` と一致し，β について連続になる．
pub fn log_z_at(betas: &[f64], energies: &[Vec<f64>], lz: &[f64], beta: f64) -> f64 {
    let top = betas.len() - 1;
    assert!(
        (0.0..=betas[top]).contains(&beta),
        "beta = {beta} is outside the ladder"
    );
    // partition_point: β_l ≤ β を満たす l の個数（列は単調増加）
    let l = betas.partition_point(|&b| b <= beta) - 1;
    if l == top {
        return lz[top];
    }
    let d = beta - betas[l];
    lz[l] + log_mean_exp(energies[l].iter().map(move |&e| -d * e))
}

/// F(β) を β について最小化し，(β̂, F(β̂)) を返す（雑音精度の経験ベイズ推定）．
///
/// 格子上の最小点 β_{l̂} の両隣 [β_{l̂-1}, β_{l̂+1}] を対数等間隔の細かい格子で探索し，
/// 格子点の間は [`log_z_at`] で補間する．β̂ の分解能が逆温度列の間隔に縛られなくなる．
pub fn minimize_free_energy(
    betas: &[f64],
    energies: &[Vec<f64>],
    lz: &[f64],
    n: usize,
) -> (f64, f64) {
    const N_FINE: usize = 400;
    let f = free_energy(betas, lz, n);
    let (l_hat, _) = f
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let top = betas.len() - 1;
    let lo = betas[l_hat.saturating_sub(1).max(1)];
    let hi = betas[(l_hat + 1).min(top)];
    let (llo, lhi) = (lo.ln(), hi.ln());
    let fine = (0..=N_FINE).map(|i| (llo + (lhi - llo) * i as f64 / N_FINE as f64).exp());
    std::iter::once(betas[l_hat])
        .chain(fine)
        .map(|b| b.clamp(lo, hi))
        .map(|b| {
            let fb = 0.5 * n as f64 * (2.0 * std::f64::consts::PI / b).ln()
                - log_z_at(betas, energies, lz, b);
            (b, fb)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// ベイズ自由エネルギー（雑音精度 β の関数として）
///   F(β) = -log ∫ Π_i N(y_i | f(x_i; w), 1/β) φ(w) dw = (n/2) log(2π/β) - log Z(β)
pub fn free_energy(betas: &[f64], lz: &[f64], n: usize) -> Vec<f64> {
    betas
        .iter()
        .zip(lz)
        .map(|(&b, &z)| {
            if b == 0.0 {
                f64::INFINITY
            } else {
                0.5 * n as f64 * (2.0 * std::f64::consts::PI / b).ln() - z
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::synthesize;

    fn toy_data() -> (Vec<f64>, Vec<f64>, Prior) {
        let x: Vec<f64> = (0..101).map(|i| i as f64 * 0.03).collect();
        let truth = [
            Peak {
                a: 1.0,
                mu: 1.2,
                b: 40.0,
            },
            Peak {
                a: 0.7,
                mu: 2.0,
                b: 30.0,
            },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let y = synthesize(&x, &truth, 0.1, &mut rng);
        let prior = Prior {
            a_rate: 1.0,
            mu_lo: 0.0,
            mu_hi: 3.0,
            b_shape: 2.0,
            b_rate: 2.0 / 30.0,
        };
        (x, y, prior)
    }

    fn small_config() -> Config {
        Config {
            n_replica: 8,
            beta_min: 1e-1,
            beta_max: 1e2,
            burn_in: 200,
            n_sample: 300,
            adapt_interval: 50,
            target_acc: 0.4,
            recompute_interval: 100,
            thin: 10,
            seed: 7,
            ladder_interval: 0,
        }
    }

    #[test]
    fn adapt_ladder_keeps_endpoints_and_narrows_bottleneck() {
        let b = ladder(6, 1e-1, 1e2);
        let rates = [0.7, 0.6, 0.1, 0.6, 0.7];
        let b2 = adapt_ladder(&b, &rates, LADDER_GAIN);
        assert_eq!(b2[0], 0.0);
        assert_eq!(b2[5], b[5]);
        assert!(b2.windows(2).all(|w| w[0] < w[1]));
        // 受理率の低いペア (2, 3) の間隔は，他のペアに比べて相対的に狭まる
        let ratio = |v: &[f64], l: usize| (v[l + 1] - v[l]) / (v[2] - v[1]);
        assert!(ratio(&b2, 2) < ratio(&b, 2));
        // 受理率がそろっていれば列は変わらない
        let same = adapt_ladder(&b, &[0.5; 5], LADDER_GAIN);
        assert!(same.iter().zip(&b).all(|(u, v)| (u - v).abs() <= 1e-12 * v));
    }

    /// 再重み付けによる log Z(β) が格子点で隣接比の積と一致し，格子点の間で連続であること
    #[test]
    fn log_z_at_agrees_with_ladder_at_grid_points() {
        let (x, y, prior) = toy_data();
        let res = run(&x, &y, 2, &prior, &small_config());
        let lz = log_z(&res.betas, &res.energies);
        for (&b, &z) in res.betas.iter().zip(&lz) {
            assert!(
                (log_z_at(&res.betas, &res.energies, &lz, b) - z).abs() <= 1e-9 * z.abs().max(1.0)
            );
        }
        // β_{l+1} の直前から近づけたときも lz[l+1] に連続につながる
        let l = 4;
        let below = res.betas[l + 1] * (1.0 - 1e-12);
        let z_below = log_z_at(&res.betas, &res.energies, &lz, below);
        assert!((z_below - lz[l + 1]).abs() < 1e-6 * lz[l + 1].abs().max(1.0));
        // 細かい格子で最小化した F は格子上の最小値以下
        let f = free_energy(&res.betas, &lz, x.len());
        let f_grid = f.iter().cloned().fold(f64::INFINITY, f64::min);
        let (_, f_fine) = minimize_free_energy(&res.betas, &res.energies, &lz, x.len());
        assert!(f_fine <= f_grid);
    }

    #[test]
    fn ladder_starts_at_zero_and_is_increasing() {
        let b = ladder(10, 1e-2, 1e3);
        assert_eq!(b.len(), 10);
        assert_eq!(b[0], 0.0);
        assert!((b[1] - 1e-2).abs() < 1e-12);
        assert!((b[9] - 1e3).abs() / 1e3 < 1e-12);
        assert!(b.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn log_mean_exp_is_stable_for_large_arguments() {
        let v = [1000.0, 1000.0, 1000.0];
        assert!((log_mean_exp(v.iter().copied()) - 1000.0).abs() < 1e-12);
        let v = [-1.0e4, 0.0];
        assert!((log_mean_exp(v.iter().copied()) - 0.5f64.ln()).abs() < 1e-12);
    }

    /// 差分更新した SSE が，一から計算し直した値と一致すること
    #[test]
    fn incremental_update_matches_recompute() {
        let (x, y, prior) = toy_data();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let peaks = (0..3).map(|_| prior.sample_peak(&mut rng)).collect();
        let mut slot = Slot::new(50.0, Chain::new(peaks, &x, &y), rng, prior.initial_steps());
        for _ in 0..500 {
            slot.sweep(&x, &prior);
        }
        let incremental = slot.chain.sse;
        slot.chain.recompute(&x, &y);
        assert!((incremental - slot.chain.sse).abs() <= 1e-9 * slot.chain.sse.max(1.0));
    }

    /// スレッド数を変えても結果がビット単位で一致すること（逆温度列の調整あり・なしの両方）
    #[test]
    fn result_is_independent_of_thread_count() {
        let (x, y, prior) = toy_data();
        for ladder_interval in [0, 50] {
            let cfg = Config {
                ladder_interval,
                ..small_config()
            };
            let run_with = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| run(&x, &y, 2, &prior, &cfg))
            };
            let r1 = run_with(1);
            let r2 = run_with(3);
            assert_eq!(r1.betas, r2.betas);
            assert_eq!(r1.energies, r2.energies);
            assert_eq!(r1.exchange_rate, r2.exchange_rate);
            if ladder_interval > 0 {
                // 調整が実際に働いて列が初期値から動いていること
                assert_ne!(r1.betas, ladder(cfg.n_replica, cfg.beta_min, cfg.beta_max));
            }
        }
    }
}
