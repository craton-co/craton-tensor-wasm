//! `bali completions <shell>` — emit shell completion scripts.
//!
//! Uses `clap_complete::generate` against the CLI's `Command` to produce a
//! completion script for the requested shell on stdout. Wire-up:
//!
//! ```bash
//! # bash
//! bali completions bash > /etc/bash_completion.d/bali
//!
//! # zsh (assuming $fpath contains ~/.zsh/completions)
//! bali completions zsh > ~/.zsh/completions/_bali
//!
//! # fish
//! bali completions fish > ~/.config/fish/completions/bali.fish
//!
//! # PowerShell
//! bali completions powershell | Out-String | Invoke-Expression
//! ```

use std::io;

use anyhow::Result;
use clap::CommandFactory;
use clap_complete::{generate, Shell};

use crate::Cli;

/// Entry point for `bali completions`.
pub fn run(shell: Shell) -> Result<()> {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    generate(shell, &mut cmd, bin_name, &mut io::stdout());
    Ok(())
}
