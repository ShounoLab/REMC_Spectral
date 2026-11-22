//! 実行のメタ情報（`meta.yaml`）を書くための最小限の YAML 書き出し
//!
//! 書き出すだけで読み込みはしないので，依存クレートを使わずに手で書く．
//! 分析側の Python（PyYAML の `yaml.safe_load`）で型が崩れずに読めることを優先し，
//! YAML 1.1 の型推測の癖を次のように避ける．
//!
//! - f64 は必ず小数点を含む 10 進表記にする（`1000` は int，`1e-10` は文字列と解釈されるため）．
//!   Rust の `{}` 表示は指数表記を使わず，値を復元できる最短の桁を出すので，読み戻しても値は変わらない．
//! - NaN と ±∞ は `.nan`，`.inf`，`-.inf` と書く．
//! - 文字列はすべて二重引用符で囲む（`yes` が bool，数字だけのハッシュが int になるのを防ぐ）．
//! - キーは英数字と `_` だけにする．

use std::io::{self, Write};

/// YAML のスカラーとして書ける値
pub trait ToYaml {
    /// YAML のスカラー表記
    fn to_yaml(&self) -> String;
}

impl ToYaml for f64 {
    fn to_yaml(&self) -> String {
        float(*self)
    }
}

impl ToYaml for bool {
    fn to_yaml(&self) -> String {
        self.to_string()
    }
}

impl ToYaml for str {
    fn to_yaml(&self) -> String {
        string(self)
    }
}

impl ToYaml for String {
    fn to_yaml(&self) -> String {
        string(self)
    }
}

macro_rules! impl_to_yaml_int {
    ($($t:ty),*) => {$(
        impl ToYaml for $t {
            fn to_yaml(&self) -> String {
                self.to_string()
            }
        }
    )*};
}
impl_to_yaml_int!(usize, u32, u64, i32, i64);

impl<T: ToYaml + ?Sized> ToYaml for &T {
    fn to_yaml(&self) -> String {
        (**self).to_yaml()
    }
}

impl<T: ToYaml> ToYaml for [T] {
    /// 1 行の flow 形式の列 `[a, b, …]`
    fn to_yaml(&self) -> String {
        let items: Vec<String> = self.iter().map(ToYaml::to_yaml).collect();
        format!("[{}]", items.join(", "))
    }
}

impl<T: ToYaml> ToYaml for Vec<T> {
    fn to_yaml(&self) -> String {
        self.as_slice().to_yaml()
    }
}

/// f64 の YAML 表記．有限なら小数点を必ず含む 10 進表記，そうでなければ `.nan`/`.inf`/`-.inf`．
pub fn float(v: f64) -> String {
    if v.is_nan() {
        ".nan".into()
    } else if v.is_infinite() {
        if v > 0.0 { ".inf" } else { "-.inf" }.into()
    } else {
        let s = v.to_string();
        if s.contains('.') {
            s
        } else {
            s + ".0"
        }
    }
}

/// 二重引用符で囲んだ YAML 文字列．`\`，`"`，制御文字はエスケープする．
pub fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn check_key(key: &str) {
    assert!(
        !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "YAML key must be [A-Za-z0-9_]+: {key:?}"
    );
}

/// ブロック形式の YAML を字下げを管理しながら書く
pub struct YamlWriter<W: Write> {
    w: W,
    depth: usize,
}

impl<W: Write> YamlWriter<W> {
    pub fn new(w: W) -> Self {
        YamlWriter { w, depth: 0 }
    }

    fn indent(&self) -> String {
        "  ".repeat(self.depth)
    }

    /// 行頭のコメント `# …`
    pub fn comment(&mut self, text: &str) -> io::Result<()> {
        text.lines()
            .try_for_each(|line| writeln!(self.w, "{}# {line}", self.indent()))
    }

    /// `key: value`
    pub fn field<T: ToYaml + ?Sized>(&mut self, key: &str, value: &T) -> io::Result<()> {
        check_key(key);
        writeln!(self.w, "{}{key}: {}", self.indent(), value.to_yaml())
    }

    /// 入れ子のマップ `key:` を開き，中身を `f` で書く
    pub fn map(
        &mut self,
        key: &str,
        f: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        check_key(key);
        writeln!(self.w, "{}{key}:", self.indent())?;
        self.depth += 1;
        let r = f(self);
        self.depth -= 1;
        r
    }

    /// 小さなマップの列 `key:` と，各行 `- {k: v, …}`（flow 形式）．
    /// 値は [`ToYaml::to_yaml`] で表記済みの文字列で渡す．
    pub fn rows<'a, I>(&mut self, key: &str, rows: I) -> io::Result<()>
    where
        I: IntoIterator<Item = Vec<(&'a str, String)>>,
    {
        check_key(key);
        writeln!(self.w, "{}{key}:", self.indent())?;
        for row in rows {
            let items: Vec<String> = row
                .iter()
                .map(|(k, v)| {
                    check_key(k);
                    format!("{k}: {v}")
                })
                .collect();
            writeln!(self.w, "{}  - {{{}}}", self.indent(), items.join(", "))?;
        }
        Ok(())
    }

    /// 書き出し先を返す
    pub fn into_inner(self) -> W {
        self.w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_always_have_a_decimal_point() {
        assert_eq!(float(1000.0), "1000.0");
        assert_eq!(float(0.01), "0.01");
        assert_eq!(float(1e-10), "0.0000000001");
        assert_eq!(float(-2.5), "-2.5");
        assert_eq!(float(f64::NAN), ".nan");
        assert_eq!(float(f64::INFINITY), ".inf");
        assert_eq!(float(f64::NEG_INFINITY), "-.inf");
        // 最短表記なので読み戻すと同じ値になる
        for v in [0.1 + 0.2, 1.0 / 3.0, 1.0 / 0.15f64.powi(2), 1e300, 5e-324] {
            assert_eq!(float(v).parse::<f64>().unwrap(), v);
        }
    }

    #[test]
    fn strings_are_quoted_and_escaped() {
        assert_eq!(string("yes"), "\"yes\"");
        assert_eq!(string("1234567"), "\"1234567\"");
        assert_eq!(string("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
    }

    #[test]
    fn writer_nests_maps_and_rows() {
        let mut y = YamlWriter::new(Vec::new());
        y.field("name", "demo").unwrap();
        y.map("prior", |y| {
            y.field("a_rate", &1.0)?;
            y.map("inner", |y| y.field("n", &3usize))
        })
        .unwrap();
        y.rows(
            "results",
            [vec![("K", 1usize.to_yaml()), ("F", (-1.5f64).to_yaml())]],
        )
        .unwrap();
        y.field("list", &[1.0, 2.0][..]).unwrap();
        let out = String::from_utf8(y.into_inner()).unwrap();
        assert_eq!(
            out,
            "name: \"demo\"\nprior:\n  a_rate: 1.0\n  inner:\n    n: 3\nresults:\n  - {K: 1, F: -1.5}\nlist: [1.0, 2.0]\n"
        );
    }
}
