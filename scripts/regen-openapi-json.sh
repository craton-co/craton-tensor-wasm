#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Regenerate crates/tensor-wasm-api/openapi.json from the authoritative
# openapi/tensor-wasm-api.yaml.
#
# The YAML is the single source of truth for the tensor-wasm-api HTTP
# contract; the JSON copy is kept in the crate root for tooling that
# does not parse YAML (eg. some OpenAPI typed-client generators and
# operator-facing UIs that consume JSON only). The two MUST agree on
# `paths` keys -- the `openapi_json_yaml_sync` regression test under
# crates/tensor-wasm-api/tests/ enforces this at CI time.
#
# Usage:
#   bash scripts/regen-openapi-json.sh        # writes crates/tensor-wasm-api/openapi.json
#
# Requires Python 3.6+ with the `pyyaml` package available. If your
# environment does not have pyyaml, run:
#   python3 -m pip install --user pyyaml
# (or, if `yq` is on PATH, swap the python invocation for
#  `yq -o=json '.' openapi/tensor-wasm-api.yaml`).
#
# The leading `x-comment` field in the JSON output is added by the
# inline Python and warns hand-editors off. It uses the OpenAPI `x-`
# extension prefix so the 3.1 schema's `unevaluatedProperties: false`
# root check (enforced by swagger-cli) accepts it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YAML_PATH="${REPO_ROOT}/openapi/tensor-wasm-api.yaml"
JSON_PATH="${REPO_ROOT}/crates/tensor-wasm-api/openapi.json"

if [[ ! -f "${YAML_PATH}" ]]; then
  echo "error: source YAML not found at ${YAML_PATH}" >&2
  exit 1
fi

python3 - "${YAML_PATH}" "${JSON_PATH}" <<'PY'
import json, sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    doc = yaml.safe_load(f)

# Inject the do-not-edit banner as the first key so it shows up at the
# top of the file. Python 3.7+ preserves insertion order on plain dicts.
banner = (
    "Generated from openapi/tensor-wasm-api.yaml. Do NOT edit by hand -- "
    "edit the YAML and re-export this JSON (see scripts/regen-openapi-json.{sh,ps1}). "
    "The openapi_json_yaml_sync regression test enforces path-key parity at CI time."
)
out = {"x-comment": banner}
out.update(doc)

with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"wrote {dst}")
PY
