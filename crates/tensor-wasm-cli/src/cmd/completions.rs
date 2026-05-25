// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm completions <shell>` — emit shell completion scripts.
//!
//! Uses `clap_complete::generate` against the CLI's `Command` to produce a
//! completion script for the requested shell. By default the script is
//! written to stdout; `--out-dir <dir>` writes it to a conventional filename
//! inside `<dir>` instead so callers (or CI) can regenerate the committed
//! scaffolding under `crates/tensor-wasm-cli/completions/` in one pass.
//!
//! ```bash
//! # bash, stdout
//! tensor-wasm completions bash > /etc/bash_completion.d/tensor-wasm
//!
//! # zsh, stdout
//! tensor-wasm completions zsh > ~/.zsh/completions/_tensor-wasm
//!
//! # fish, stdout
//! tensor-wasm completions fish > ~/.config/fish/completions/tensor-wasm.fish
//!
//! # Regenerate every script under the committed `completions/` directory
//! tensor-wasm completions bash --out-dir crates/tensor-wasm-cli/completions
//! tensor-wasm completions zsh  --out-dir crates/tensor-wasm-cli/completions
//! tensor-wasm completions fish --out-dir crates/tensor-wasm-cli/completions
//! ```

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::CommandFactory;
use clap_complete::{generate, generate_to, Shell};

use crate::Cli;

/// Entry point for `tensor-wasm completions`.
///
/// When `out_dir` is `None`, the generated script is written to stdout.
/// Otherwise the script is written to `<out_dir>/<conventional-name>` where
/// the conventional name matches what `clap_complete::generate_to` picks
/// (`tensor-wasm.bash`, `_tensor-wasm` for zsh, `tensor-wasm.fish`,
/// `tensor-wasm.elv`, `_tensor-wasm.ps1`).
pub fn run(shell: Shell, out_dir: Option<PathBuf>) -> Result<()> {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();

    match out_dir {
        None => {
            generate(shell, &mut cmd, bin_name, &mut io::stdout());
        }
        Some(dir) => {
            ensure_dir(&dir)?;
            let written = generate_to(shell, &mut cmd, bin_name, &dir)
                .with_context(|| format!("writing completion script to {}", dir.display()))?;
            println!("wrote {}", written.display());
        }
    }
    Ok(())
}

/// Create `dir` (and any missing parents) so the very first regen against a
/// fresh checkout does not fail on a missing `completions/` directory.
fn ensure_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        let meta = fs::metadata(dir)
            .with_context(|| format!("stat'ing --out-dir {}", dir.display()))?;
        if !meta.is_dir() {
            anyhow::bail!("--out-dir {} exists but is not a directory", dir.display());
        }
        return Ok(());
    }
    fs::create_dir_all(dir)
        .with_context(|| format!("creating --out-dir {}", dir.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn ensure_dir_creates_missing_parent() {
        let tmp = tempdir().unwrap();
        let nested = tmp.path().join("a").join("b").join("c");
        ensure_dir(&nested).unwrap();
        assert!(nested.is_dir());
    }

    #[test]
    fn ensure_dir_rejects_non_directory() {
        let tmp = tempdir().unwrap();
        let file = tmp.path().join("not-a-dir");
        fs::write(&file, b"x").unwrap();
        assert!(ensure_dir(&file).is_err());
    }
}
