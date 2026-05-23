//! `bali snapshot` — save or restore a running instance to a `.bali` file.
//!
//! Stub implementations until S20 wires the CLI to `bali-snapshot::Writer` /
//! `Reader`. Both sub-actions parse their arguments, validate the obvious
//! invariants (input file exists, output directory writable), then print
//! `todo` and exit 0 so smoke tests and downstream tooling can build against
//! a stable surface.

use std::path::PathBuf;

use anyhow::Result;
use clap::Subcommand;

/// `bali snapshot` sub-actions.
#[derive(Debug, Subcommand)]
pub enum SnapshotAction {
    /// Capture the state of a running instance into a `.bali` file.
    Save {
        /// Identifier of the running instance to snapshot.
        instance_id: String,
        /// Output path for the resulting `.bali` archive.
        out: PathBuf,
    },
    /// Restore an instance from a `.bali` archive.
    Restore {
        /// Path to the `.bali` archive to load.
        input: PathBuf,
    },
}

/// Entry point for `bali snapshot`.
pub fn run(action: SnapshotAction) -> Result<()> {
    match action {
        SnapshotAction::Save { instance_id, out } => {
            println!(
                "snapshot save: would capture instance {} -> {} (todo)",
                instance_id,
                out.display()
            );
        }
        SnapshotAction::Restore { input } => {
            println!("snapshot restore: would load {} (todo)", input.display());
        }
    }
    Ok(())
}
