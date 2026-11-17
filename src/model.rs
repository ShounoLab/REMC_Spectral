//! スペクトルモデル・事前分布・人工データ生成
//!
//! モデル: y_i = Σ_k g(x_i; w_k) + ε_i,  ε_i ~ N(0, 1/β)
//! ピークの形 g は [`PeakShape`] トレイトで抽象化する．
//! いま実装しているのはガウス型 [`Gauss`]: g(x) = a exp(-b (x - μ)² / 2)（b はピーク幅の精度パラメータ）．

use rand::Rng;
use rand_distr::{Distribution, Exp, Gamma, Normal, Uniform};

/// ピークの形が満たすべき約束．
///
/// 交換モンテカルロと自由エネルギーの推定は「ピークの和」であることしか使わないので，
/// この約束を満たす型なら何でもピークとして使える．[`crate::remc::run`] はこのトレイトで
/// ジェネリックに書かれていて，形ごとに専用のコードがコンパイル時に生成される（静的ディスパッチ）．
///
/// パラメータは添字 j = 0, 1, …, [`PeakShape::N_PARAMS`] − 1 で指す．
/// 単一成分 Metropolis 更新はこの順に 1 個ずつ動かす．
pub trait PeakShape: Copy + Send + Sync + std::fmt::Debug {
    /// この形の事前分布の型．正規化された proper な分布であること．
    type Prior: Sync;
    /// パラメータの名前（表示用）．並び順が添字 j に対応する．
    const PARAM_NAMES: &'static [&'static str];
    /// パラメータの個数
    const N_PARAMS: usize = Self::PARAM_NAMES.len();

    /// 点 x でのピークの値．交換モンテカルロの最内ループで呼ばれる．
    fn eval(&self, x: f64) -> f64;
    /// パラメータ j の値
    fn get(&self, j: usize) -> f64;
    /// パラメータ j に値を書く
    fn set(&mut self, j: usize, v: f64);
    /// ピークの位置．ラベルスイッチング対策でサンプルを並べ替えるのに使う．
    fn center(&self) -> f64;

    /// パラメータ j の対数事前密度（定数項は省略してよい）．台の外なら -∞．
    fn log_prior(prior: &Self::Prior, j: usize, v: f64) -> f64;
    /// 事前分布からピークを 1 本サンプルする（初期値と β = 0 のスロットの引き直しに使う）
    fn sample_prior<R: Rng + ?Sized>(prior: &Self::Prior, rng: &mut R) -> Self;
    /// 提案分布のステップ幅の初期値（長さ [`PeakShape::N_PARAMS`]）
    fn initial_steps(prior: &Self::Prior) -> Vec<f64>;
}

/// (a, μ, b) 型のピークでのパラメータの添字
const A: usize = 0;
const MU: usize = 1;
const B: usize = 2;

/// ガウス型ピーク 1 本: a exp(-b (x - μ)² / 2)．パラメータの順は (a, μ, b)．
///
/// `Copy` を derive しているので，代入や関数への受け渡しはビットコピーになる
/// （所有権の移動を気にしなくてよい小さな値型）．
#[derive(Clone, Copy, Debug)]
pub struct Gauss {
    pub a: f64,
    pub mu: f64,
    pub b: f64,
}

impl PeakShape for Gauss {
    type Prior = Prior;
    const PARAM_NAMES: &'static [&'static str] = &["a", "mu", "b"];

    #[inline(always)]
    fn eval(&self, x: f64) -> f64 {
        let d = x - self.mu;
        self.a * (-0.5 * self.b * d * d).exp()
    }

    #[inline]
    fn get(&self, j: usize) -> f64 {
        match j {
            A => self.a,
            MU => self.mu,
            B => self.b,
            _ => panic!("Gauss has no parameter {j}"),
        }
    }

    #[inline]
    fn set(&mut self, j: usize, v: f64) {
        match j {
            A => self.a = v,
            MU => self.mu = v,
            B => self.b = v,
            _ => panic!("Gauss has no parameter {j}"),
        }
    }

    #[inline]
    fn center(&self) -> f64 {
        self.mu
    }

    #[inline]
    fn log_prior(prior: &Prior, j: usize, v: f64) -> f64 {
        prior.log_density(j, v)
    }

    fn sample_prior<R: Rng + ?Sized>(prior: &Prior, rng: &mut R) -> Self {
        let [a, mu, b] = prior.sample(rng);
        Gauss { a, mu, b }
    }

    fn initial_steps(prior: &Prior) -> Vec<f64> {
        prior.initial_steps().to_vec()
    }
}

/// (a, μ, b) 型のピークの事前分布 φ(w) = Π_k Exp(a_k; λ_a) · U(μ_k; lo, hi) · Gamma(b_k; α, rate)
///
/// a は高さ，μ は位置，b は幅の精度パラメータ．[`Gauss`] が使う．
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
    /// パラメータ j（0: a，1: μ，2: b）の対数事前密度（定数項は省略）．台の外なら -∞．
    #[inline]
    pub fn log_density(&self, j: usize, v: f64) -> f64 {
        match j {
            A => {
                if v <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    -self.a_rate * v
                }
            }
            MU => {
                if (self.mu_lo..=self.mu_hi).contains(&v) {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
            B => {
                if v <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    (self.b_shape - 1.0) * v.ln() - self.b_rate * v
                }
            }
            _ => panic!("Prior has no parameter {j}"),
        }
    }

    /// 事前分布から (a, μ, b) を 1 組サンプルする．乱数は a, μ, b の順に消費する．
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 3] {
        [
            Exp::new(self.a_rate).unwrap().sample(rng),
            Uniform::new(self.mu_lo, self.mu_hi).sample(rng),
            Gamma::new(self.b_shape, 1.0 / self.b_rate)
                .unwrap()
                .sample(rng),
        ]
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
pub fn model<S: PeakShape>(x: f64, peaks: &[S]) -> f64 {
    peaks.iter().map(|p| p.eval(x)).sum()
}

/// 人工データ: 真のピーク列にガウス雑音（標準偏差 sigma）を加える
pub fn synthesize<S: PeakShape, R: Rng + ?Sized>(
    x: &[f64],
    truth: &[S],
    sigma: f64,
    rng: &mut R,
) -> Vec<f64> {
    let noise = Normal::new(0.0, sigma).unwrap();
    x.iter()
        .map(|&xi| model(xi, truth) + noise.sample(rng))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// どの形にも課す基本の約束: 添字の数と名前の数が合い，get と set が往復し，
    /// 事前分布からのサンプルが台の内側にあり，初期ステップ幅の長さが合うこと
    fn check_shape_contract<S: PeakShape>(prior: &S::Prior) {
        assert_eq!(S::N_PARAMS, S::PARAM_NAMES.len());
        assert_eq!(S::initial_steps(prior).len(), S::N_PARAMS);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..100 {
            let mut p = S::sample_prior(prior, &mut rng);
            for j in 0..S::N_PARAMS {
                let v = p.get(j);
                assert!(S::log_prior(prior, j, v).is_finite());
                p.set(j, v + 0.5);
                assert_eq!(p.get(j), v + 0.5);
                p.set(j, v);
            }
        }
    }

    #[test]
    fn gauss_satisfies_shape_contract() {
        let prior = Prior {
            a_rate: 1.0,
            mu_lo: 0.0,
            mu_hi: 3.0,
            b_shape: 2.0,
            b_rate: 2.0 / 30.0,
        };
        check_shape_contract::<Gauss>(&prior);
        let g = Gauss {
            a: 2.0,
            mu: 1.0,
            b: 4.0,
        };
        assert_eq!(g.eval(1.0), 2.0);
        assert!((g.eval(1.5) - 2.0 * (-0.5f64).exp()).abs() < 1e-15);
        assert_eq!(g.center(), 1.0);
    }
}
