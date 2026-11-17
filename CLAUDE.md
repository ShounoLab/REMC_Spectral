# remc-spectral

ベイズ的スペクトル分解（ガウス型ピークの和）を交換モンテカルロ法で解き，
ピーク数 K と雑音精度 β についてのベイズ自由エネルギー F(K, β) を求める Rust クレート．
背景と式は @README.md を参照．

設計判断や数値手法の検討はサブエージェントに委任しない．

## コマンド

- ビルド: `cargo build --release`
- テスト: `cargo test`（`[profile.test]` で最適化済み．結合テストを含め数秒）
- 整形・静的検査: `cargo fmt --all` / `cargo clippy --all-targets -- -D warnings`
- ドキュメント: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
- デモ実行: `cargo run --release -- [K_max] [n_sample] [seed] [data_seed] [n] [n_replica]`（スレッド数は `RAYON_NUM_THREADS`）

変更を終えたら fmt・clippy・test がすべて通ることを確認する（CI と同じ条件）．

## 構成

- `src/model.rs`: `Peak`，`Prior`（Exp × Uniform × Gamma），人工データ生成
- `src/remc.rs`: 交換モンテカルロ本体と自由エネルギー推定，単体テスト
- `src/main.rs`: デモ用バイナリ（ライブラリを `remc_spectral::` 経由で使う）
- `tests/free_energy.rs`: K = 1 の F(β) を格子数値積分と比較する結合テスト

`Slot`（β・ステップ幅・RNG．温度に固定）と `Chain`（ピーク列・残差・SSE．温度間を移動）を分離し，
交換は `std::mem::swap` で `Chain` だけを入れ替える．

## 数値計算上の不変条件（壊さないこと）

- 目標分布は p_β(w) ∝ exp(-β nE_n(w)) φ(w)，nE_n(w) = SSE/2．β は雑音精度を兼ねる．
- 逆温度列は β_0 = 0 を必ず含む．自由エネルギーは Z(0) = ∫φ = 1 を基準にするので，
  事前分布は正規化された proper な分布であること（対数密度の定数項は省略してよい）．
- `F(β) = (n/2) log(2π/β) - log Z(β)`．log Z は隣接温度比の積で，必ず log-sum-exp で計算する．
- β_0 = 0 のスロットは Metropolis ではなく，毎スイープ事前分布から独立に引き直す（`Slot::redraw_from_prior`）．
  乱数はそのスロットのストリームから取る．
- 交換の受理確率は `exp((β_{l+1} - β_l)(E_{l+1} - E_l))`．偶数ペアと奇数ペアを交互に試す．
- 残差 `resid`，寄与 `contrib`，`sse` は差分更新で常に整合させる．`recompute` で定期的に誤差をリセットする．
- ステップ幅の適応は burn-in 中のみ．本サンプリング中に変えると詳細釣り合いが崩れる．
- 逆温度列の適応（`adapt_ladder`）は burn-in の前半のみ．後半は固定した列で平衡化させる．β_0 = 0 と最大の β は動かさない．
- 格子点の間の log Z（`log_z_at`）は隣接比と同じ推定量を使い，格子点で `log_z` と一致させる．
- 乱数はスロットごとの `ChaCha8Rng` ストリームだけを使う．`thread_rng()` や共有 RNG を並列部分に持ち込まない
  （スレッド数に依存せずビット単位で再現することをテストで保証している）．

サンプラーや推定器に手を入れたら，`incremental_update_matches_recompute`，`beta_zero_slot_draws_independently_from_prior`，
`result_is_independent_of_thread_count`，`k1_free_energy_matches_quadrature`，
`k1_free_energy_matches_quadrature_with_adapted_ladder` が通ることを必ず確認する．
テストの許容誤差を緩めて通すのは禁止．通らない場合は原因を報告する．

## 実験記録とノート

- 作業を再開するとき（「続き」「前回の」などと言われたとき）は，まず `notes/README.md` の一覧から関連ノートと，
  `logs/README.md` の一覧を読み，経緯・未検証の仮説・未着手の候補を把握してから始める．
- 実験を実行したら `logs/YYYY-MM-DD_名前/` に目的・コード版・コマンド・要約表・所見（`README.md`）と生出力を残し，
  `logs/README.md` の一覧に 1 行足す．デモは CSV をカレントディレクトリに書くので，作業用ディレクトリで実行する．
- 会話で出たアイディアや仮説は `notes/` にテーマごとにまとめる．出どころ（[利用者]/[Claude]/[両者]）と
  状態（未着手/検討中/実装済み/却下）を付ける．書き方は `notes/README.md`．

## コーディング規約

- コメントと doc コメントは日本語（句読点は「，．」）．数式は Unicode で書いてよい．
- 添字ループよりイテレータ（`zip`，`chunks_exact_mut` など）を優先する．clippy の警告は抑制せず直す．
- 依存クレートを追加・更新する前に相談すること．
- 公開 API を変えたら `README.md` と doc コメントも更新する．

## 参考文献

- Nagata, Sugita & Okada, Neural Networks 28, 82–89 (2012)．交換モンテカルロによるベイズ的スペクトル分解．
- Tokuda, Nagata & Okada, J. Phys. Soc. Jpn. 86, 024001 (2017)．雑音分散とピーク数の同時推定．
