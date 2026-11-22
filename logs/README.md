# logs

実験の記録．1 回のまとまった実験ごとに `YYYY-MM-DD_短い名前/` を作る．
考察やアイディアの議論は `notes/` に書き，ここには事実（何を実行して何が出たか）を残す．

## フォルダの中身

```
logs/YYYY-MM-DD_名前/
├── README.md        目的，コード版（コミット），実行コマンド，結果の要約表，所見
├── <実行名>/         1 回の実行ごとに 1 つ（--out-dir の出力先）
│   ├── meta.yaml    出どころ・設定・結果の要約（git で追跡する）
│   ├── stdout.txt   デモの標準出力（git で追跡する）
│   └── *.csv        data.csv，free_energy.csv など（.gitignore で除外．必要なら再生成する）
└── *.py など        要約表を作るのに使った解析スクリプト（あれば）
```

CSV は git で追跡しない．残したい数値は README の要約表に書く．
2026-11-21 以前の記録は，実行ごとのフォルダを作らず標準出力を `*.txt` で並べた古い形のまま．

## 実験の手順

1. **コードをコミットしてから実行する**．`meta.yaml` の `dirty: false` は，CSV をそのコミットから再生成できることの保証になる．
   やむを得ず未コミットのコードで実行したら，README の「コード」欄に差分の内容と，のちにコミットしたハッシュを書く．
2. **フォルダを作り，ビルドする**．
   ```sh
   cargo build --release
   D=logs/$(date +%F)_名前
   ```
3. **実行する**．実行ごとにサブフォルダを分け，標準出力もそこに保存する．
   ```sh
   for ds in 1 2 3; do
     mkdir -p $D/ds$ds
     ./target/release/remc-spectral --out-dir $D/ds$ds 5 20000 42 $ds > $D/ds$ds/stdout.txt
   done
   ```
   出力先にすでに `meta.yaml` があると上書きせずに止まる．やり直すときはそのサブフォルダを消してから実行する．
4. **不調な実行がないか確かめる**．`meta.yaml` の `dirty` と `results.per_k` の `exchange_min`（目安 0.3 以上）を見る．
   交換が詰まった区間は `free_energy.csv` の `var_nE` でも検出できる（[var-check](2026-11-22_var-check/README.md) の `varcheck.py`）．
5. **README.md を書く**．目的，コード版，コマンド，要約表，所見．解析に使ったスクリプトは同じフォルダに置く．
   図を描いたら，描画用リポジトリのどの notebook（とそのコミット）で描いたかを書く．
6. **この表に 1 行足す**．
7. **コミットする**．コードの変更とは別のコミットにする（コードを先にコミットし，そのハッシュを README に書く）．

## CSV の再生成

CSV は追跡しないので，描画などで必要になったら `meta.yaml` の `provenance.commit` と `provenance.command` から作り直す．

```sh
c=<provenance.commit>
git worktree add ../remc-$c $c
(cd ../remc-$c && cargo build --release)
../remc-$c/target/release/remc-spectral --out-dir <新しい出力先> <command の引数（--out-dir 以外）>
git worktree remove ../remc-$c
```

- 結果はスレッド数によらずビット単位で再現する．ただし `exp` などの最後のビットは OS によって違いうるので，
  別の OS では値がわずかにずれることがある．正しい値は README の要約表と `meta.yaml` の結果を基準にする．
- `dirty: true` の実行は，README の「コード」欄にあるコミット（と差分の説明）から作り直す．

## 一覧

| 日付 | 実験 | 要点 |
|---|---|---|
| 2026-11-14 | [mc-seed-scan](2026-11-14_mc-seed-scan/README.md) | MCMC の seed だけを変えても F の揺らぎは約 0.05．K=4 が最小なのは系統的 |
| 2026-11-15 | [data-seed-scan](2026-11-15_data-seed-scan/README.md) | データを変えると 9 組中 6 組で K=3．K=5 の交換率が 0.10 まで低下 |
| 2026-11-15 | [ladder-adapt](2026-11-15_ladder-adapt/README.md) | 逆温度列の適応と β̂ の補間を入れて再実行．β̂ が 95〜106 の連続値に |
| 2026-11-16 | [n-scan](2026-11-16_n-scan/README.md) | n = 101〜10001．F(4)−F(3) は log n に傾き 0.22 で増加（正則なら 1.5）．n ≥ 1001 でほぼ K = 3 |
| 2026-11-17 | [mixing-diag](2026-11-17_mixing-diag/README.md) | エネルギーの τ_int は β ≤ 400 で 3〜9 スイープ．K = 5 は β ≳ 450 で数百〜1400（F(β̂) には影響しない） |
| 2026-11-17 | [beta0-prior](2026-11-17_beta0-prior/README.md) | β_0 を事前分布から独立に引き直す．β_0 の τ_int 6〜11 → 1，F の平均は不変，K = 1 の seed 間の揺らぎが約半分に |
| 2026-11-22 | [model-selection-csv](2026-11-22_model-selection-csv/README.md) | PR #1 の補間曲線とモデル選択の CSV は整合．格子だけの最小は F_min から最大 0.49 ずれる（K の差と同程度） |
| 2026-11-22 | [var-check](2026-11-22_var-check/README.md) | `var_nE` 列を追加．Δ⟨nE⟩ / (−∫Var dβ) は交換率 ≥ 0.3 の区間で 0.81〜1.11，交換が詰まった区間だけ 3.1〜3.7 |
