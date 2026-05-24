// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm completions <shell>` — emit shell completion scripts.
//!
//! Uses `clap_complete::generate` against the CLI's `Command` to produce a
//! completion script for the requested shell on stdout. Wire-up:
//!
//! ```bash
//! # bash
//! tensor-wasm completions bash > /etc/bash_completion.d/tensor-wasm
//!
//! # zsh (assuming $fpath contains ~/.zsh/completions)
//! tensor-wasm completions zsh > ~/.zsh/completions/_tensor-wasm
//!
//! # fish
//! tensor-wasm completions fish > ~/.config/fish/completions/tensor-wasm.fish
//!
//! # PowerShell
//! tensor-wasm completions powershell | Out-String | Invoke-Expression
//! ```

use std::io;

use anyhow::Result;
use clap::CommandFactory;
use clap_complete::{generate, Shell};

use crate::Cli;

/// Entry point for `tensor-wasm completions`.
pub fn run(shell: Shell) -> Result<()> {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    generate(shell, &mut cmd, bin_name, &mut io::stdout());
    Ok(())
}
