# remc-spectral

ベイズ的スペクトル分解（ガウス型ピークの和）を交換モンテカルロ法で行い，
ピーク数 K と雑音精度 β についてのベイズ自由エネルギー F(K, β) を計算する実装．

## 背景

観測モデル y_i = Σ_k a_k exp(-b_k (x_i - μ_k)² / 2) + ε_i，ε_i ~ N(0, 1/β) に対し，
逆温度を雑音精度 β とみなして交換モンテカルロを回すと，1 回の実行で

    F(K, β) = (n/2) log(2π/β) - log Z(β),   Z(β) = ∫ exp(-β nE_n(w)) φ(w) dw

が逆温度列上で得られる．log Z は隣接温度の比の積で推定する（Z(0) = 1）．

- K. Nagata, S. Sugita, M. Okada, Neural Networks 28, 82–89 (2012).
- S. Tokuda, K. Nagata, M. Okada, J. Phys. Soc. Jpn. 86, 024001 (2017).

## 使い方

    cargo run --release -- [K_max=5] [n_sample=20000] [seed=42] [data_seed=20120401] [n=301] [n_replica=40]
    # seed: 交換モンテカルロの乱数，data_seed: 人工データの雑音の乱数
    # n: データ点数（[0, 3] を等間隔），n_replica: レプリカ数
    RAYON_NUM_THREADS=8 cargo run --release      # スレッド数の指定

出力: `data.csv`（人工データ），`free_energy.csv`（K, β, F, ⟨nE⟩, 交換率）

## 逆温度列の適応と β̂ の推定

- β_0 = 0 のスロットの目標分布は事前分布 φ そのものなので，Metropolis ではなく毎スイープ φ から独立に引き直す．
  新しい配置が交換で上の温度へ運ばれ，log Z の最初の比 log ⟨exp(-β_1 nE_n)⟩_φ も独立なサンプルで推定できる．

- `Config::ladder_interval` > 0 のとき，burn-in の前半でその間隔ごとに隣接交換率を測り，
  交換率がそろうように逆温度列を動かす（`remc::adapt_ladder`）．β_0 = 0 と β の最大値は固定．
  burn-in の後半と本サンプリングでは列を固定する．0 なら等比の列のまま．
- `remc::log_z_at` は格子点の間の β で log Z(β) を再重み付けで求める
  （log Z(β) = log Z(β_l) + log ⟨exp(-(β - β_l) nE_n)⟩_{β_l}）．
  `remc::minimize_free_energy` はこれを使って F(β) を最小化し，β̂ を逆温度列の格子に縛られずに求める．

## 構成

```
.
├── Cargo.toml
├── src/
│   ├── lib.rs        ライブラリクレートのルート
│   ├── model.rs      ピーク関数，事前分布，人工データ生成
│   ├── remc.rs       交換モンテカルロ本体（差分残差更新，偶奇交換，自由エネルギー推定）＋単体テスト
│   └── main.rs       デモ用バイナリ（K = 1..K_max の実行とモデル選択）
├── tests/
│   └── free_energy.rs  結合テスト: K = 1 の自由エネルギーを格子数値積分と比較（逆温度列の適応あり・なし）
├── logs/              実験の記録（実験ごとのフォルダに要約と生出力）
├── notes/             会話で出たアイディア・仮説・検討中の設計のまとめ
├── .github/workflows/ci.yml  fmt / clippy / doc / test（Linux・macOS）
├── CLAUDE.md          Claude Code 向けのプロジェクト説明と不変条件
└── .claude/
    ├── settings.json    権限設定と，.rs 編集後に rustfmt を走らせるフック
    └── skills/check/    /check: CI と同じ検査を一括実行
```

## 開発

    cargo test                                   # 単体テスト＋結合テスト
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    cargo doc --no-deps --open                   # API ドキュメント

単体テストでは，差分更新した SSE が再計算値と一致すること，
β = 0 のスロットが事前分布から独立に引き直していること，
スレッド数を変えても結果がビット単位で一致すること（逆温度列の適応あり・なし），
再重み付けした log Z が格子点で隣接比の積と一致することを確認している．
