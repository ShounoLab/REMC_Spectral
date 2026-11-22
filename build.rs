//! ビルド時に git のコミットハッシュと未コミットの変更の有無を埋め込む．
//!
//! 実行結果の `meta.yaml` に書き，どのコードで出した結果かを後から追えるようにする．
//! git が使えない環境（ソースの tarball など）では "unknown" にする．

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // 追跡しているファイルの変更だけを見る（未追跡のファイルは結果に影響しない）
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(s) => (!s.is_empty()).to_string(),
        None => "unknown".into(),
    };
    println!("cargo:rustc-env=REMC_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=REMC_GIT_DIRTY={dirty}");

    // コミットや checkout で HEAD・index が変わったとき，ソースを編集したとき（dirty が変わる）に再実行する．
    // ディレクトリを指定すると，その中のどのファイルが変わっても再実行される．
    for path in [
        ".git/HEAD",
        ".git/index",
        ".git/refs",
        "src",
        "tests",
        "Cargo.toml",
        "build.rs",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
}
