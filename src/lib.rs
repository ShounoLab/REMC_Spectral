//! ベイズ的スペクトル分解のための交換モンテカルロ法（レプリカ交換モンテカルロ法）
//!
//! - [`model`] … ガウス型ピークのモデル，事前分布，人工データ生成
//! - [`remc`] … 交換モンテカルロ本体と自由エネルギー推定
//!
//! 参考文献:
//! - K. Nagata, S. Sugita, M. Okada, "Bayesian spectral deconvolution with the
//!   exchange Monte Carlo method", Neural Networks 28, 82–89 (2012).
//! - S. Tokuda, K. Nagata, M. Okada, "Simultaneous estimation of noise variance and
//!   number of peaks in Bayesian spectral deconvolution", J. Phys. Soc. Jpn. 86, 024001 (2017).

pub mod model;
pub mod remc;
