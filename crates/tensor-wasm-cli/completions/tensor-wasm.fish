# Fish completion for `tensor-wasm`. Regenerate with:
#   tensor-wasm completions fish --out-dir crates/tensor-wasm-cli/completions

# Top-level: global flags and subcommand names.
complete -c tensor-wasm -n "__fish_use_subcommand" -l tenant -d 'Tenant id to advertise on outbound API requests' -r
complete -c tensor-wasm -n "__fish_use_subcommand" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_use_subcommand" -s V -l version -d 'Print version'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "run" -d 'Run a Wasm module locally against the in-process TensorWasm engine'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "deploy" -d 'Upload a Wasm module to a TensorWasm server'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "invoke" -d 'Invoke a previously deployed function by id'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "bench" -d 'Benchmark local invocation latency'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "snapshot" -d 'Save or restore an instance snapshot'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "metrics" -d 'Fetch and pretty-print Prometheus metrics from a TensorWasm server'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "observe" -d 'Live operator dashboard over /healthz + /metrics'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "completions" -d 'Emit shell completion scripts for the named shell'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "man" -d 'Generate roff(7) man pages from the clap command tree'
complete -c tensor-wasm -n "__fish_use_subcommand" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'

# tensor-wasm run
complete -c tensor-wasm -n "__fish_seen_subcommand_from run" -l export -d 'Name of the exported function to call' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from run" -l args -d 'Arguments to pass to the export, encoded as a JSON array' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from run" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from run" -s h -l help -d 'Print help'

# tensor-wasm deploy
complete -c tensor-wasm -n "__fish_seen_subcommand_from deploy" -l server -d 'Base URL of the target TensorWasm server' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from deploy" -l name -d 'Tenant-supplied display name' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from deploy" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from deploy" -s h -l help -d 'Print help'

# tensor-wasm invoke
complete -c tensor-wasm -n "__fish_seen_subcommand_from invoke" -l server -d 'Base URL of the target TensorWasm server' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from invoke" -l args -d 'Arguments forwarded to the function, encoded as a JSON array' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from invoke" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from invoke" -s h -l help -d 'Print help'

# tensor-wasm bench
complete -c tensor-wasm -n "__fish_seen_subcommand_from bench" -l export -d 'Name of the export to call on each iteration' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from bench" -l n -d 'Number of iterations to run' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from bench" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from bench" -s h -l help -d 'Print help'

# tensor-wasm snapshot
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and not __fish_seen_subcommand_from save restore help" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and not __fish_seen_subcommand_from save restore help" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and not __fish_seen_subcommand_from save restore help" -f -a "save" -d 'Capture the state of a running instance into a .tensor-wasm file via the API'
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and not __fish_seen_subcommand_from save restore help" -f -a "restore" -d 'Restore an instance from a .tensor-wasm archive via the API'
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and not __fish_seen_subcommand_from save restore help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'

# tensor-wasm snapshot save
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from save" -l instance -d 'Identifier of the running instance to snapshot' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from save" -l output -d 'Output path for the resulting .tensor-wasm archive' -r -F
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from save" -l server -d 'Base URL of the target TensorWasm server' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from save" -s h -l help -d 'Print help'

# tensor-wasm snapshot restore
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from restore" -l input -d 'Path to the .tensor-wasm archive to upload' -r -F
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from restore" -l as-instance -d 'Identifier to assign to the restored instance' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from restore" -l server -d 'Base URL of the target TensorWasm server' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from restore" -l max-decompressed -d 'Maximum decompressed snapshot size, in bytes' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from snapshot; and __fish_seen_subcommand_from restore" -s h -l help -d 'Print help'

# tensor-wasm metrics
complete -c tensor-wasm -n "__fish_seen_subcommand_from metrics" -l server -d 'Base URL of the target TensorWasm server' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from metrics" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from metrics" -s h -l help -d 'Print help'

# tensor-wasm observe
complete -c tensor-wasm -n "__fish_seen_subcommand_from observe" -l addr -d 'Base URL of the target TensorWasm server' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from observe" -l interval -d 'Refresh interval, in seconds' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from observe" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from observe" -s h -l help -d 'Print help'

# tensor-wasm completions
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions; and not __fish_seen_subcommand_from bash zsh fish powershell elvish" -f -a "bash" -d 'Bourne Again SHell (bash)'
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions; and not __fish_seen_subcommand_from bash zsh fish powershell elvish" -f -a "zsh" -d 'Z SHell (zsh)'
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions; and not __fish_seen_subcommand_from bash zsh fish powershell elvish" -f -a "fish" -d 'Friendly Interactive SHell (fish)'
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions; and not __fish_seen_subcommand_from bash zsh fish powershell elvish" -f -a "powershell" -d 'PowerShell'
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions; and not __fish_seen_subcommand_from bash zsh fish powershell elvish" -f -a "elvish" -d 'Elvish'
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions" -l out-dir -d 'Optional output directory' -r -F
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from completions" -s h -l help -d 'Print help'

# tensor-wasm man
complete -c tensor-wasm -n "__fish_seen_subcommand_from man" -l out-dir -d 'Output directory for the per-subcommand .1 files' -r -F
complete -c tensor-wasm -n "__fish_seen_subcommand_from man" -l tenant -d 'Tenant id' -r
complete -c tensor-wasm -n "__fish_seen_subcommand_from man" -s h -l help -d 'Print help'
