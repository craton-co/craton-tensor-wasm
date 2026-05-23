//! `bali invoke` — call a previously deployed function by id.
//!
//! S18 stub: parses arguments, validates the server URL, then prints the
//! planned request and exits 0. The wire-level invocation (POST
//! `/v1/invoke/{id}` with JSON args, streaming the response back) lands in
//! S20 alongside `reqwest`.

use anyhow::Result;
use clap::Args;

/// Arguments to `bali invoke`.
#[derive(Debug, Args)]
pub struct InvokeArgs {
    /// Identifier of the deployed function to invoke.
    pub id: String,
    /// Base URL of the target Bali server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
    /// Arguments forwarded to the function, encoded as a JSON array.
    #[arg(long)]
    pub args: Option<String>,
}

/// Entry point for `bali invoke`.
pub fn run(args: InvokeArgs) -> Result<()> {
    super::validate_server_url(&args.server)?;
    if let Some(json) = &args.args {
        let parsed: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("--args is not valid JSON: {e}"))?;
        if !parsed.is_array() {
            anyhow::bail!("--args must be a JSON array, got {}", parsed);
        }
    }

    println!(
        "invoke: would POST {}/v1/invoke/{} with args={}",
        args.server.trim_end_matches('/'),
        args.id,
        args.args.as_deref().unwrap_or("[]")
    );
    println!("HTTP transport wired in S20 with reqwest");
    Ok(())
}
