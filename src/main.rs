//! ベイズ的スペクトル分解 + 交換モンテカルロ法（Nagata, Sugita & Okada 2012 の枠組み）
//! 雑音精度を逆温度とみなして F(K, β) を同時に求める（Tokuda, Nagata & Okada 2017 の考え方）
//!
//! 使い方: cargo run --release -- [--out-dir DIR] [K_max] [n_sample] [seed] [data_seed] [n] [n_replica]
//! seed は交換モンテカルロの乱数，data_seed は人工データの雑音の乱数に使う．
//! n はデータ点数（観測範囲 [0, 3] を等間隔に取る），n_replica はレプリカ数．
//! スレッド数は環境変数 RAYON_NUM_THREADS で指定（未指定なら論理コア数）．
//!
//! 出力は DIR（既定はカレントディレクトリ）に書く:
//! data.csv（人工データ），free_energy.csv（格子点ごとの F），
//! free_energy_fine.csv（格子点の間を再重み付けで補間した F(β)），model_selection.csv（K ごとの β̂ と F_min），
//! meta.yaml（出どころ（コミット，引数，日時），データ，事前分布，設定，K ごとの交換率と実行時間）．
//! 既存の結果を上書きしないよう，DIR に meta.yaml があれば何もせずに終了する．

use remc_spectral::{model, remc, yaml};

use model::{Gauss, PeakShape, Prior};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use yaml::{ToYaml, YamlWriter};

/// free_energy_fine.csv で F(β) を求める β の点数（[β_1, β_max] を対数等間隔に取る）．
const N_FINE_CURVE: usize = 200;

/// K ごとの実行の要約（CSV にない値だけ．`meta.yaml` に書く）
struct KSummary {
    k: usize,
    min_ex: f64,
    max_ex: f64,
    elapsed_s: f64,
}

fn main() -> std::io::Result<()> {
    let command: Vec<String> = std::env::args().collect();
    let started = SystemTime::now();
    let t_start = Instant::now();

    // `--out-dir DIR`（または `--out-dir=DIR`）を取り除き，残りを位置引数として読む
    let mut out_dir = PathBuf::from(".");
    let mut args: Vec<String> = Vec::new();
    let mut it = command.iter().cloned();
    while let Some(a) = it.next() {
        if a == "--out-dir" {
            out_dir = it
                .next()
                .map(PathBuf::from)
                .expect("--out-dir needs a directory");
        } else if let Some(d) = a.strip_prefix("--out-dir=") {
            out_dir = PathBuf::from(d);
        } else {
            args.push(a);
        }
    }
    std::fs::create_dir_all(&out_dir)?;
    let meta_path = out_dir.join("meta.yaml");
    if meta_path.exists() {
        eprintln!(
            "{} already exists; refusing to overwrite a previous run (choose another --out-dir)",
            meta_path.display()
        );
        std::process::exit(1);
    }
    let k_max: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let n_sample: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);
    let data_seed: u64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_120_401);
    let n: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(301);
    let n_replica: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(40);

    // ---- 人工データ: 3 本の重なったピーク，雑音 σ = 0.1（真の β = 100） ----
    let x: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 / (n - 1) as f64).collect();
    let truth = [
        Gauss {
            a: 1.0,
            mu: 1.0,
            b: 1.0 / 0.15f64.powi(2),
        },
        Gauss {
            a: 0.8,
            mu: 1.4,
            b: 1.0 / 0.15f64.powi(2),
        },
        Gauss {
            a: 1.2,
            mu: 2.2,
            b: 1.0 / 0.20f64.powi(2),
        },
    ];
    let sigma = 0.1;
    let mut data_rng = ChaCha8Rng::seed_from_u64(data_seed);
    let y = model::synthesize(&x, &truth, sigma, &mut data_rng);
    {
        let mut w = BufWriter::new(File::create(out_dir.join("data.csv"))?);
        writeln!(w, "x,y")?;
        for (xi, yi) in x.iter().zip(&y) {
            writeln!(w, "{xi},{yi}")?;
        }
    }

    // ---- 事前分布（デモ用の値．実データでは物理的な知見から設計する） ----
    let prior = Prior {
        a_rate: 1.0, // a ~ Exp(1)
        mu_lo: 0.0,  // μ ~ U(0, 3)（観測範囲内に限定）
        mu_hi: 3.0,
        b_shape: 2.0, // b ~ Gamma(2, rate)，平均 30（幅 ≈ 0.18）
        b_rate: 2.0 / 30.0,
    };

    let cfg = remc::Config {
        n_replica,
        beta_min: 1e-2,
        beta_max: 1e3,
        burn_in: n_sample / 2,
        n_sample,
        adapt_interval: 100,
        target_acc: 0.4,
        recompute_interval: 1000,
        thin: 10,
        seed,
        ladder_interval: 250,
    };

    println!(
        "threads = {}, N = {}, L = {}, sweeps = {} + {}",
        rayon::current_num_threads(),
        n,
        cfg.n_replica,
        cfg.burn_in,
        cfg.n_sample
    );
    println!(
        "true sigma = {sigma}  (true beta = {:.1}), seed = {seed}, data_seed = {data_seed}\n",
        1.0 / (sigma * sigma)
    );

    let mut csv = BufWriter::new(File::create(out_dir.join("free_energy.csv"))?);
    writeln!(csv, "K,beta,F,mean_nE,exchange_rate,var_nE")?;
    let mut fine_csv = BufWriter::new(File::create(out_dir.join("free_energy_fine.csv"))?);
    writeln!(fine_csv, "K,beta,F")?;

    let mut best: Option<(usize, f64, f64)> = None; // (K, F_min, β̂)
    let mut results = Vec::new();
    let mut summaries = Vec::new();

    for k in 1..=k_max {
        let t0 = Instant::now();
        let res = remc::run::<Gauss>(&x, &y, k, &prior, &cfg);
        let lz = remc::log_z(&res.betas, &res.energies);
        let f = remc::free_energy(&res.betas, &lz, n);

        // β について最小化（= 雑音分散の経験ベイズ推定）．格子点の間は再重み付けで補間する．
        // 事後平均の表示には，F が最小の格子点 l̂ のスロットのサンプルを使う．
        let (l_hat, _) = f
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let (beta_hat, f_min) = remc::minimize_free_energy(&res.betas, &res.energies, &lz, n);
        let (min_ex, max_ex) = res
            .exchange_rate
            .iter()
            .fold((1.0f64, 0.0f64), |(lo, hi), &r| (lo.min(r), hi.max(r)));

        let elapsed_s = t0.elapsed().as_secs_f64();
        println!(
            "K={k}: F_min = {f_min:9.3} at beta = {beta_hat:8.2} (sigma_hat = {:.4}), \
             exchange rate = [{min_ex:.3}, {max_ex:.3}], {elapsed_s:.1} s",
            1.0 / beta_hat.sqrt(),
        );
        summaries.push(KSummary {
            k,
            min_ex,
            max_ex,
            elapsed_s,
        });

        // exchange_rate[l] はスロット l と l+1 の間の交換率（最後のスロットは NaN）
        let rows = res
            .betas
            .iter()
            .zip(&f)
            .zip(&res.energies)
            .enumerate()
            .skip(1);
        // var_nE は本サンプリング全スイープの nE の分散（1/T で割る）．平衡なら d⟨nE⟩/dβ = −Var(nE) なので，
        // 隣接格子点の ⟨nE⟩ の差分と突き合わせると交換が詰まった区間の非平衡を検出できる．β² Var は実効的な λ．
        for (l, ((beta, fl), e)) in rows {
            let t = e.len() as f64;
            let mean_e = e.iter().sum::<f64>() / t;
            let var_e = e.iter().map(|v| (v - mean_e).powi(2)).sum::<f64>() / t;
            let ex = res.exchange_rate.get(l).copied().unwrap_or(f64::NAN);
            writeln!(csv, "{k},{beta},{fl},{mean_e},{ex},{var_e}")?;
        }

        // 格子点の間も含めた F(β) の曲線．β_0 = 0 では F が発散するので β_1 から取る．
        let (b_lo, b_hi) = (res.betas[1], *res.betas.last().unwrap());
        let (llo, lhi) = (b_lo.ln(), b_hi.ln());
        for i in 0..=N_FINE_CURVE {
            let beta = (llo + (lhi - llo) * i as f64 / N_FINE_CURVE as f64)
                .exp()
                .clamp(b_lo, b_hi);
            let fb = 0.5 * n as f64 * (2.0 * std::f64::consts::PI / beta).ln()
                - remc::log_z_at(&res.betas, &res.energies, &lz, beta);
            writeln!(fine_csv, "{k},{beta},{fb}")?;
        }

        let improved = match best {
            Some((_, f_best, _)) => f_min < f_best,
            None => true,
        };
        if improved {
            best = Some((k, f_min, beta_hat));
        }
        results.push((res, l_hat, f_min, beta_hat));
    }

    // ---- 最小値からの差 ΔF(K)．1 nat 程度の差はモデル選択の根拠としては弱い ----
    let f_best = best.unwrap().1;
    let mut sel_csv = BufWriter::new(File::create(out_dir.join("model_selection.csv"))?);
    writeln!(sel_csv, "K,beta_hat,sigma_hat,F_min,delta_F")?;
    println!("\nDelta F (K) = F_min(K) - min_K F_min(K):");
    for (k, (_, _, fm, bh)) in results.iter().enumerate() {
        println!("  K={}: {:8.3}", k + 1, fm - f_best);
        writeln!(
            sel_csv,
            "{},{bh},{},{fm},{}",
            k + 1,
            1.0 / bh.sqrt(),
            fm - f_best
        )?;
    }

    // ---- 選ばれたモデルの事後平均（β̂ のスロットのサンプルから） ----
    let (k_hat, _, beta_hat) = best.unwrap();
    let (res, l_hat, _, _) = &results[k_hat - 1];
    println!("\nargmin K = {k_hat}, beta_hat = {beta_hat:.2}");
    println!(
        "posterior mean ± sd at grid beta = {:.2} nearest to the F minimum (peaks sorted by center):",
        res.betas[*l_hat]
    );
    let s = &res.samples[*l_hat];
    for k in 0..k_hat {
        // パラメータ j の事後平均と標準偏差
        let stats: Vec<String> = Gauss::PARAM_NAMES
            .iter()
            .enumerate()
            .map(|(j, name)| {
                let v: Vec<f64> = s.iter().map(|p| p[k].get(j)).collect();
                let m = v.iter().sum::<f64>() / v.len() as f64;
                let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
                format!("{name} = {m:.3}±{sd:.3}")
            })
            .collect();
        println!("  peak {k}: {}", stats.join(", "));
    }
    println!(
        "acceptance ({}) at that grid beta: {:.2?}",
        Gauss::PARAM_NAMES.join(", "),
        res.acceptance[*l_hat]
    );
    for w in [&mut csv, &mut fine_csv, &mut sel_csv] {
        w.flush()?;
    }

    let run = RunInfo {
        command: &command,
        started,
        total_s: t_start.elapsed().as_secs_f64(),
        n,
        sigma,
        data_seed,
        truth: &truth,
        k_max,
        prior: &prior,
        cfg: &cfg,
        summaries: &summaries,
        k_hat,
        beta_hat,
    };
    write_meta(File::create(&meta_path)?, &run)?;
    println!(
        "\nwrote data.csv, free_energy.csv, free_energy_fine.csv, model_selection.csv, meta.yaml in {}",
        out_dir.display()
    );
    Ok(())
}

/// `meta.yaml` に書く内容
struct RunInfo<'a> {
    command: &'a [String],
    started: SystemTime,
    total_s: f64,
    n: usize,
    sigma: f64,
    data_seed: u64,
    truth: &'a [Gauss],
    k_max: usize,
    prior: &'a Prior,
    cfg: &'a remc::Config,
    summaries: &'a [KSummary],
    k_hat: usize,
    beta_hat: f64,
}

/// 実行のメタ情報を YAML で書く．
///
/// `Prior` と `Config` はフィールドを漏れなく分解しているので，フィールドを足すと
/// ここがコンパイルエラーになり，書き忘れを防げる．
fn write_meta<W: Write>(w: W, r: &RunInfo) -> std::io::Result<()> {
    let mut y = YamlWriter::new(BufWriter::new(w));
    y.comment("remc-spectral の実行のメタ情報．CSV の列の意味は README.md の「出力」を参照．")?;
    y.field("format_version", &1u32)?;

    y.map("provenance", |y| {
        y.field("commit", env!("REMC_GIT_COMMIT"))?;
        // ビルド時に追跡ファイルに未コミットの変更があったか（git が使えなければ "unknown"）
        match env!("REMC_GIT_DIRTY") {
            "true" => y.field("dirty", &true),
            "false" => y.field("dirty", &false),
            other => y.field("dirty", other),
        }?;
        y.field("version", env!("CARGO_PKG_VERSION"))?;
        y.field("started_utc", &utc_iso8601(r.started))?;
        y.field("elapsed_s", &r.total_s)?;
        y.field("command", r.command)?;
        y.field("threads", &rayon::current_num_threads())?;
        y.field("os", std::env::consts::OS)?;
        y.field("arch", std::env::consts::ARCH)
    })?;

    y.map("data", |y| {
        y.field("kind", "synthetic")?;
        y.field("file", "data.csv")?;
        y.field("n", &r.n)?;
        y.field("x_range", &[0.0, 3.0][..])?;
        y.field("sigma", &r.sigma)?;
        y.field("data_seed", &r.data_seed)?;
        y.field("truth_shape", shape_name::<Gauss>())?;
        y.rows("truth", r.truth.iter().map(peak_row))
    })?;

    y.map("model", |y| {
        y.field("shape", shape_name::<Gauss>())?;
        y.field("k_max", &r.k_max)
    })?;

    let Prior {
        a_rate,
        mu_lo,
        mu_hi,
        b_shape,
        b_rate,
    } = r.prior;
    y.map("prior", |y| {
        y.field("a_rate", a_rate)?;
        y.field("mu_lo", mu_lo)?;
        y.field("mu_hi", mu_hi)?;
        y.field("b_shape", b_shape)?;
        y.field("b_rate", b_rate)
    })?;

    let remc::Config {
        n_replica,
        beta_min,
        beta_max,
        burn_in,
        n_sample,
        adapt_interval,
        target_acc,
        recompute_interval,
        thin,
        seed,
        ladder_interval,
    } = r.cfg;
    y.map("config", |y| {
        y.field("n_replica", n_replica)?;
        y.field("beta_min", beta_min)?;
        y.field("beta_max", beta_max)?;
        y.field("burn_in", burn_in)?;
        y.field("n_sample", n_sample)?;
        y.field("adapt_interval", adapt_interval)?;
        y.field("target_acc", target_acc)?;
        y.field("recompute_interval", recompute_interval)?;
        y.field("thin", thin)?;
        y.field("seed", seed)?;
        y.field("ladder_interval", ladder_interval)?;
        y.field("n_fine_curve", &N_FINE_CURVE)
    })?;

    y.map("results", |y| {
        y.map("files", |y| {
            y.field("grid", "free_energy.csv")?;
            y.field("fine", "free_energy_fine.csv")?;
            y.field("model_selection", "model_selection.csv")
        })?;
        y.field("k_hat", &r.k_hat)?;
        y.field("beta_hat", &r.beta_hat)?;
        // F_min・β̂ は model_selection.csv にあるので，ここには CSV にない値だけを書く
        y.rows(
            "per_k",
            r.summaries.iter().map(|s| {
                vec![
                    ("K", s.k.to_yaml()),
                    ("exchange_min", s.min_ex.to_yaml()),
                    ("exchange_max", s.max_ex.to_yaml()),
                    ("elapsed_s", s.elapsed_s.to_yaml()),
                ]
            }),
        )
    })?;

    y.into_inner().flush()
}

/// ピーク 1 本を `{a: …, mu: …, b: …}` の行にする
fn peak_row<S: PeakShape>(p: &S) -> Vec<(&'static str, String)> {
    S::PARAM_NAMES
        .iter()
        .enumerate()
        .map(|(j, &name)| (name, p.get(j).to_yaml()))
        .collect()
}

/// 形の型名（モジュールのパスを除いたもの）
fn shape_name<S: PeakShape>() -> &'static str {
    let full = std::any::type_name::<S>();
    full.rsplit("::").next().unwrap_or(full)
}

/// UTC の ISO 8601 表記（秒まで）．依存クレートを増やさないよう，日付は
/// 1970-01-01 からの日数を暦に直す標準的な算法（H. Hinnant の civil_from_days）で求める．
fn utc_iso8601(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, sod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        sod / 3600,
        sod % 3600 / 60,
        sod % 60
    )
}
