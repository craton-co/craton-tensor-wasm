# NOTE: regenerate via `tensor-wasm completions fish` after building.
# Print an optspec for argparse to handle cmd's options that are independent of any subcommand.
function __fish_tensor_wasm_global_optspecs
	string join \n tenant= h/help V/version
end

function __fish_tensor_wasm_needs_command
	# Figure out if the current invocation already has a command.
	set -l cmd (commandline -opc)
	set -e cmd[1]
	argparse -s (__fish_tensor_wasm_global_optspecs) -- $cmd 2>/dev/null
	or return
	if set -q argv[1]
		# Also print the command, so this can be used to figure out what it is.
		echo $argv[1]
		return 1
	end
	return 0
end

function __fish_tensor_wasm_using_subcommand
	set -l cmd (__fish_tensor_wasm_needs_command)
	test -z "$cmd"
	and return 1
	contains -- $cmd[1] $argv
end

complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -s V -l version -d 'Print version'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "run" -d 'Run a Wasm module locally against the in-process TensorWasm engine'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "deploy" -d 'Upload a Wasm module to a TensorWasm server'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "invoke" -d 'Invoke a previously deployed function by id'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "bench" -d 'Benchmark local invocation latency (P50/P95/P99/max)'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "snapshot" -d 'Save or restore an instance snapshot'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "metrics" -d 'Fetch and pretty-print Prometheus metrics from a TensorWasm server'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "observe" -d 'Live operator dashboard over `/healthz` + `/metrics` (refreshes in place)'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "serve" -d 'Run the TensorWasm HTTP API gateway in-process (binds and serves)'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "completions" -d 'Emit shell completion scripts for the named shell'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "man" -d 'Generate roff(7) man pages from the clap command tree'
complete -c tensor-wasm -n "__fish_tensor_wasm_needs_command" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand run" -l export -d 'Name of the exported function to call. Defaults to `main`' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand run" -l args -d 'Arguments to pass to the export, encoded as a JSON array' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand run" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand run" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand deploy" -l server -d 'Base URL of the target TensorWasm server (e.g. `http://localhost:8080`)' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand deploy" -l name -d 'Tenant-supplied display name. Defaults to the file stem when omitted' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand deploy" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand deploy" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand invoke" -l server -d 'Base URL of the target TensorWasm server (e.g. `http://localhost:8080`)' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand invoke" -l args -d 'Arguments forwarded to the function, encoded as a JSON array' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand invoke" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand invoke" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand bench" -l export -d 'Name of the export to call on each iteration' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand bench" -l n -d 'Number of iterations to run. Must be >= 1' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand bench" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand bench" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and not __fish_seen_subcommand_from save restore help" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and not __fish_seen_subcommand_from save restore help" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and not __fish_seen_subcommand_from save restore help" -f -a "save" -d 'Capture the state of a running instance into a `.tensor-wasm` file via the API'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and not __fish_seen_subcommand_from save restore help" -f -a "restore" -d 'Restore an instance from a `.tensor-wasm` archive via the API'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and not __fish_seen_subcommand_from save restore help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from save" -l instance -d 'Identifier of the running instance to snapshot' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from save" -l output -d 'Output path for the resulting `.tensor-wasm` archive' -r -F
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from save" -l server -d 'Base URL of the target TensorWasm server (e.g. `http://localhost:8080`)' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from save" -l max-restore-bytes -d 'Maximum number of bytes the CLI will accept from the server and write to `--output`. Defaults to 256 MiB; values above the default are clamped down so a malicious server cannot fill the operator\'s disk by streaming an unbounded response body' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from save" -l hmac-key-file -d 'Path to a 32-byte HMAC-SHA256 key. The file is interpreted as 64 hex characters if it\'s that length (whitespace trimmed), otherwise as 32 raw bytes. Mismatched length → error' -r -F
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from save" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from save" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -l input -d 'Path to the `.tensor-wasm` archive to upload' -r -F
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -l as-instance -d 'Identifier to assign to the restored instance' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -l server -d 'Base URL of the target TensorWasm server (e.g. `http://localhost:8080`)' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -l max-archive-bytes -d 'Maximum *on-disk archive* size the CLI will upload, in bytes (default 256 MiB). This bounds the compressed payload only — the decompressed footprint is enforced server-side and may be much larger. The deprecated alias `--max-decompressed` is accepted for one release; prefer `--max-archive-bytes` in new scripts' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -l hmac-key-file -d 'Path to a 32-byte HMAC-SHA256 key. The file is interpreted as 64 hex characters if it\'s that length (whitespace trimmed), otherwise as 32 raw bytes. Mismatched length → error' -r -F
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -l require-signature -d 'Refuse to restore an unsigned (v2) snapshot'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from restore" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from help" -f -a "save" -d 'Capture the state of a running instance into a `.tensor-wasm` file via the API'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from help" -f -a "restore" -d 'Restore an instance from a `.tensor-wasm` archive via the API'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand snapshot; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand metrics" -l server -d 'Base URL of the target TensorWasm server (e.g. `http://localhost:8080`)' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand metrics" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand metrics" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand observe" -l addr -d 'Base URL of the target TensorWasm server. Defaults to `http://localhost:8080`' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand observe" -l interval -d 'Refresh interval, in seconds. Must be at least 1' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand observe" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand observe" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand serve" -l addr -d 'Address to bind the HTTP server to (e.g. `127.0.0.1:8080`, `0.0.0.0:8080`)' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand serve" -l token -d 'Bearer token accepted by the gateway. Repeat to allowlist multiple tokens' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand serve" -l tenant-header-policy -d 'Policy for the X-TensorWasm-Tenant header' -r -f -a "optional\t'' required\t''"
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand serve" -l cors-origin -d 'Origin to allow via CORS. Repeat for multiple origins' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand serve" -l max-body-bytes -d 'Maximum inbound request body size, in bytes' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand serve" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand serve" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand completions" -l out-dir -d 'Optional output directory. When provided, the script is written to `<dir>/<conventional-name>` (e.g. `tensor-wasm.bash`, `_tensor-wasm` for zsh, `tensor-wasm.fish`)' -r -F
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand completions" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand completions" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand man" -l out-dir -d 'Output directory. When omitted, only the root `tensor-wasm.1` page is emitted to stdout. When provided, every subcommand gets its own page written as `<binary>-<subcommand>.1` (matching the established convention used by `git-status.1`, `cargo-build.1`, etc.)' -r -F
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand man" -l tenant -d 'Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`. Zero (the default) suppresses the header for backwards compatibility' -r
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand man" -s h -l help -d 'Print help'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "run" -d 'Run a Wasm module locally against the in-process TensorWasm engine'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "deploy" -d 'Upload a Wasm module to a TensorWasm server'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "invoke" -d 'Invoke a previously deployed function by id'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "bench" -d 'Benchmark local invocation latency (P50/P95/P99/max)'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "snapshot" -d 'Save or restore an instance snapshot'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "metrics" -d 'Fetch and pretty-print Prometheus metrics from a TensorWasm server'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "observe" -d 'Live operator dashboard over `/healthz` + `/metrics` (refreshes in place)'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "serve" -d 'Run the TensorWasm HTTP API gateway in-process (binds and serves)'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "completions" -d 'Emit shell completion scripts for the named shell'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "man" -d 'Generate roff(7) man pages from the clap command tree'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and not __fish_seen_subcommand_from run deploy invoke bench snapshot metrics observe serve completions man help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and __fish_seen_subcommand_from snapshot" -f -a "save" -d 'Capture the state of a running instance into a `.tensor-wasm` file via the API'
complete -c tensor-wasm -n "__fish_tensor_wasm_using_subcommand help; and __fish_seen_subcommand_from snapshot" -f -a "restore" -d 'Restore an instance from a `.tensor-wasm` archive via the API'
