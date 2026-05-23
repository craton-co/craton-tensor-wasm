//! `bali deploy` — upload a Wasm module to a Bali server.
//!
//! S18 ships a stub that validates the file exists and the `--server` URL
//! parses, then prints the planned action and exits 0. The real HTTP
//! transport (multipart upload of the `.wasm` artefact, returning a
//! function id) lands in S20 once `reqwest` is added as a workspace
//! dependency.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

/// Arguments to `bali deploy`.
#[derive(Debug, Args)]
pub struct DeployArgs {
    /// Path to the `.wasm` file to deploy.
    pub file: PathBuf,
    /// Base URL of the target Bali server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
}

/// Entry point for `bali deploy`.
pub fn run(args: DeployArgs) -> Result<()> {
    let metadata = std::fs::metadata(&args.file)
        .with_context(|| format!("locating wasm file {}", args.file.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("{} is not a regular file", args.file.display());
    }
    super::validate_server_url(&args.server)?;

    println!(
        "deploy: would upload {} ({} bytes) to {}",
        args.file.display(),
        metadata.len(),
        args.server
    );
    println!("HTTP transport wired in S20 with reqwest");
    Ok(())
}
