// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `--config <file>` support for the `tensor-wasm` CLI (cli feature F6).
//!
//! A small TOML file can supply *defaults* for the global connection
//! settings so operators do not have to repeat `--tenant` / `--ca-cert` /
//! `--token-file` on every invocation. The precedence model is strict and
//! one-directional: an explicit CLI flag (or the `TENSOR_WASM_TOKEN` env var,
//! for the token) always wins; the config file only fills in a value that was
//! not otherwise provided. This keeps the file from silently overriding what
//! an operator typed on the command line.
//!
//! ## Schema
//!
//! The recognised top-level keys are:
//!
//! ```toml
//! server = "https://tensor-wasm.example.com"   # default --server / --addr
//! tenant = 7                                    # default --tenant
//! token_file = "/etc/tensor-wasm/token"        # default --token-file
//! ca_cert = "/etc/tensor-wasm/ca.pem"          # default --ca-cert
//! insecure = false                             # default --insecure
//! timeout_secs = 30                            # default --timeout
//! ```
//!
//! ## Parser
//!
//! We deliberately do NOT pull in the `toml` crate for this tiny, flat
//! schema — that would add a new dependency (and lockfile churn) for what is
//! a handful of `key = value` lines. The parser below understands the strict
//! subset this config needs: blank lines, `#` comments, and
//! `key = value` where the value is a double-quoted string, a bare integer,
//! or a bare boolean. Anything else is a hard error so a typo surfaces
//! immediately rather than being silently ignored. A `[section]` header is
//! rejected because the schema is flat; if a future setting needs nesting we
//! can revisit (and at that point the `toml` crate would be justified).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Parsed `--config` file. Every field is optional; an absent field means
/// "no default supplied, fall back to the built-in / flag value".
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CliConfig {
    /// Default server base URL (used for `--server` / `--addr`). NOTE: this is
    /// surfaced for completeness and documented in the schema; threading it
    /// into the per-command `server` args is a follow-up because those are
    /// positional/required per subcommand today.
    pub server: Option<String>,
    /// Default tenant id.
    pub tenant: Option<u64>,
    /// Default path to the bearer-token file.
    pub token_file: Option<PathBuf>,
    /// Default path to an additional PEM CA root.
    pub ca_cert: Option<PathBuf>,
    /// Default for the `--insecure` TLS-verification-disable switch.
    pub insecure: Option<bool>,
    /// Default per-request HTTP timeout, in seconds.
    pub timeout_secs: Option<u64>,
}

impl CliConfig {
    /// Read and parse a config file from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading --config file {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing --config file {}", path.display()))
    }

    /// Parse the flat TOML subset described in the module docs.
    fn parse(text: &str) -> Result<Self> {
        let mut cfg = CliConfig::default();
        for (lineno, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') {
                anyhow::bail!(
                    "line {}: section headers (`{}`) are not supported; the \
                     config schema is flat (key = value)",
                    lineno + 1,
                    line
                );
            }
            let (key, value) = line.split_once('=').ok_or_else(|| {
                anyhow::anyhow!("line {}: expected `key = value`, got `{line}`", lineno + 1)
            })?;
            let key = key.trim();
            let value = value.trim();
            match key {
                "server" => cfg.server = Some(parse_string(value, key, lineno)?),
                "token_file" => {
                    cfg.token_file = Some(PathBuf::from(parse_string(value, key, lineno)?))
                }
                "ca_cert" => cfg.ca_cert = Some(PathBuf::from(parse_string(value, key, lineno)?)),
                "tenant" => cfg.tenant = Some(parse_u64(value, key, lineno)?),
                "timeout_secs" => cfg.timeout_secs = Some(parse_u64(value, key, lineno)?),
                "insecure" => cfg.insecure = Some(parse_bool(value, key, lineno)?),
                other => anyhow::bail!(
                    "line {}: unknown config key `{other}`; recognised keys are \
                     server, tenant, token_file, ca_cert, insecure, timeout_secs",
                    lineno + 1
                ),
            }
        }
        Ok(cfg)
    }
}

/// Strip an unquoted trailing `#` comment from a line. A `#` inside a quoted
/// string is preserved so a value like `"a#b"` round-trips.
fn strip_comment(line: &str) -> &str {
    let mut in_quote = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_quote = !in_quote,
            '#' if !in_quote => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Parse a double-quoted string value. We accept only the simple quoted form
/// (no escape processing beyond a literal backslash pass-through) because the
/// values here are file paths / URLs that never legitimately need escapes.
fn parse_string(value: &str, key: &str, lineno: usize) -> Result<String> {
    let inner = value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "line {}: `{key}` must be a double-quoted string, got `{value}`",
                lineno + 1
            )
        })?;
    if inner.is_empty() {
        anyhow::bail!("line {}: `{key}` must not be empty", lineno + 1);
    }
    Ok(inner.to_string())
}

/// Parse a bare unsigned-integer value.
fn parse_u64(value: &str, key: &str, lineno: usize) -> Result<u64> {
    value.parse::<u64>().map_err(|_| {
        anyhow::anyhow!(
            "line {}: `{key}` must be a non-negative integer, got `{value}`",
            lineno + 1
        )
    })
}

/// Parse a bare boolean (`true` / `false`).
fn parse_bool(value: &str, key: &str, lineno: usize) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => anyhow::bail!(
            "line {}: `{key}` must be `true` or `false`, got `{value}`",
            lineno + 1
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_config() {
        let cfg = CliConfig::parse(
            r#"
            # connection defaults
            server = "https://tensor-wasm.example.com"
            tenant = 7
            token_file = "/etc/tensor-wasm/token"
            ca_cert = "/etc/tensor-wasm/ca.pem"
            insecure = false
            timeout_secs = 30
        "#,
        )
        .expect("valid config parses");
        assert_eq!(
            cfg.server.as_deref(),
            Some("https://tensor-wasm.example.com")
        );
        assert_eq!(cfg.tenant, Some(7));
        assert_eq!(
            cfg.token_file,
            Some(PathBuf::from("/etc/tensor-wasm/token"))
        );
        assert_eq!(cfg.ca_cert, Some(PathBuf::from("/etc/tensor-wasm/ca.pem")));
        assert_eq!(cfg.insecure, Some(false));
        assert_eq!(cfg.timeout_secs, Some(30));
    }

    #[test]
    fn empty_config_is_all_none() {
        let cfg = CliConfig::parse("# nothing here\n\n").expect("empty parses");
        assert_eq!(cfg, CliConfig::default());
    }

    #[test]
    fn unknown_key_is_rejected() {
        let err = CliConfig::parse("bogus = 1\n").unwrap_err();
        assert!(format!("{err}").contains("unknown config key"));
    }

    #[test]
    fn section_header_is_rejected() {
        let err = CliConfig::parse("[connection]\nserver = \"x\"\n").unwrap_err();
        assert!(format!("{err}").contains("section headers"));
    }

    #[test]
    fn bad_type_is_rejected() {
        assert!(CliConfig::parse("tenant = \"not-a-number\"\n").is_err());
        assert!(CliConfig::parse("insecure = yes\n").is_err());
        assert!(CliConfig::parse("server = bareword\n").is_err());
    }

    #[test]
    fn inline_comment_is_stripped_outside_quotes() {
        let cfg = CliConfig::parse("tenant = 5   # the prod tenant\n").expect("parses");
        assert_eq!(cfg.tenant, Some(5));
    }

    #[test]
    fn hash_inside_quoted_value_is_preserved() {
        let cfg = CliConfig::parse("server = \"https://h/a#b\"\n").expect("parses");
        assert_eq!(cfg.server.as_deref(), Some("https://h/a#b"));
    }

    #[test]
    fn missing_equals_is_rejected() {
        assert!(CliConfig::parse("tenant 5\n").is_err());
    }
}
