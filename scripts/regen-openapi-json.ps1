# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Regenerate crates/tensor-wasm-api/openapi.json from the authoritative
# openapi/tensor-wasm-api.yaml. Windows / PowerShell sibling of
# scripts/regen-openapi-json.sh -- behaviour and output are identical.
#
# The YAML is the single source of truth for the tensor-wasm-api HTTP
# contract; the JSON copy is kept in the crate root for tooling that
# does not parse YAML (eg. some OpenAPI typed-client generators and
# operator-facing UIs that consume JSON only). The two MUST agree on
# `paths` keys -- the `openapi_json_yaml_sync` regression test under
# crates/tensor-wasm-api/tests/ enforces this at CI time.
#
# Usage:
#   pwsh scripts/regen-openapi-json.ps1
#
# Requires Python 3.6+ with the `pyyaml` package on PATH (because
# Windows PowerShell does not ship a YAML parser). If pyyaml is
# missing, run:
#   python -m pip install --user pyyaml

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$YamlPath = Join-Path $RepoRoot "openapi/tensor-wasm-api.yaml"
$JsonPath = Join-Path $RepoRoot "crates/tensor-wasm-api/openapi.json"

if (-not (Test-Path $YamlPath)) {
    Write-Error "source YAML not found at $YamlPath"
    exit 1
}

$PythonScript = @"
import json, sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, 'r', encoding='utf-8') as f:
    doc = yaml.safe_load(f)

banner = (
    'Generated from openapi/tensor-wasm-api.yaml. Do NOT edit by hand -- '
    'edit the YAML and re-export this JSON (see scripts/regen-openapi-json.{sh,ps1}). '
    'The openapi_json_yaml_sync regression test enforces path-key parity at CI time.'
)
out = {'x-comment': banner}
out.update(doc)

with open(dst, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.write('\n')
print(f'wrote {dst}')
"@

# Pipe the inline script through python via stdin to avoid quoting trouble.
$PythonScript | python - $YamlPath $JsonPath
