---
description: CI と同じ条件（fmt・clippy・doc・test）でクレート全体を検査し，失敗があれば原因をまとめる
---

次の順に実行し，すべての結果を確認する．

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets -- -D warnings`
3. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
4. `cargo test`

失敗したものがあれば，該当箇所と原因を報告する．
clippy の警告は `#[allow(...)]` で抑制せずに直す．
テストが失敗した場合，許容誤差を緩めて通すことはせず，数値的な原因（差分更新の不整合，乱数ストリームの共有など）を調べて報告する．
