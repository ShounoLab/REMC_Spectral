//! スペクトルモデル・事前分布・人工データ生成
//!
//! モデル: y_i = Σ_k a_k exp(-b_k (x_i - μ_k)^2 / 2) + ε_i,  ε_i ~ N(0, 1/β)
//! （ピーク形状はガウス型．b_k はピーク幅の精度パラメータ）

use rand::Rng;
use rand_distr::{Distribution, Exp, Gamma, Normal, Uniform};

/// ガウス型ピーク 1 本: a * exp(-b (x - mu)^2 / 2)
///
/// `Copy` を derive しているので，代入や関数への受け渡しはビットコピーになる
/// （所有権の移動を気にしなくてよい小さな値型）．
#[derive(Clone, Copy, Debug)]
pub struct Peak {
    pub a: f64,
    pub mu: f64,
    pub b: f64,
}

/// 単一成分 Metropolis 更新の単位となるパラメータの種類
#[derive(Clone, Copy, Debug)]
pub enum Param {
    A = 0,
    Mu = 1,
    B = 2,
}

pub const PARAMS: [Param; 3] = [Param::A, Param::Mu, Param::B];

impl Peak {
    #[inline(always)]
    pub fn eval(&self, x: f64) -> f64 {
        let d = x - self.mu;
        self.a * (-0.5 * self.b * d * d).exp()
    }

    #[inline]
    pub fn get(&self, p: Param) -> f64 {
        match p {
            Param::A => self.a,
            Param::Mu => self.mu,
            Param::B => self.b,
        }
    }

    #[inline]
    pub fn set(&mut self, p: Param, v: f64) {
        match p {
            Param::A => self.a = v,
            Param::Mu => self.mu = v,
            Param::B => self.b = v,
        }
    }
}

/// 事前分布 φ(w) = Π_k Exp(a_k; λ_a) · U(μ_k; lo, hi) · Gamma(b_k; α, rate)
///
/// 自由エネルギー計算では Z(β=0) = ∫φ(w)dw = 1 を基準にするので，
/// φ は「正規化された proper な分布」でなければならない．
/// ただし Metropolis 比では定数が打ち消し合うので，対数密度の正規化定数は省略してよい．
#[derive(Clone, Debug)]
pub struct Prior {
    pub a_rate: f64,
    pub mu_lo: f64,
    pub mu_hi: f64,
    pub b_shape: f64,
    pub b_rate: f64,
}

impl Prior {
    /// 1 パラメータ分の対数事前密度（定数項は省略）．台の外なら -∞．
    #[inline]
    pub fn log_density(&self, p: Param, v: f64) -> f64 {
        match p {
            Param::A => {
                if v <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    -self.a_rate * v
                }
            }
            Param::Mu => {
                if (self.mu_lo..=self.mu_hi).contains(&v) {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
            Param::B => {
                if v <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    (self.b_shape - 1.0) * v.ln() - self.b_rate * v
                }
            }
        }
    }

    /// 事前分布からピークを 1 本サンプル（初期値用）
    pub fn sample_peak<R: Rng + ?Sized>(&self, rng: &mut R) -> Peak {
        Peak {
            a: Exp::new(self.a_rate).unwrap().sample(rng),
            mu: Uniform::new(self.mu_lo, self.mu_hi).sample(rng),
            b: Gamma::new(self.b_shape, 1.0 / self.b_rate)
                .unwrap()
                .sample(rng),
        }
    }

    /// 提案分布のステップ幅の初期値（各パラメータの事前スケールの 1 割程度）
    pub fn initial_steps(&self) -> [f64; 3] {
        let b_mean = self.b_shape / self.b_rate;
        [
            0.1 / self.a_rate,
            0.05 * (self.mu_hi - self.mu_lo),
            0.1 * b_mean,
        ]
    }
}

/// モデル関数 f(x; w) = Σ_k g_k(x)
pub fn model(x: f64, peaks: &[Peak]) -> f64 {
    peaks.iter().map(|p| p.eval(x)).sum()
}

/// 人工データ: 真のピーク列にガウス雑音（標準偏差 sigma）を加える
pub fn synthesize<R: Rng + ?Sized>(x: &[f64], truth: &[Peak], sigma: f64, rng: &mut R) -> Vec<f64> {
    let noise = Normal::new(0.0, sigma).unwrap();
    x.iter()
        .map(|&xi| model(xi, truth) + noise.sample(rng))
        .collect()
}
