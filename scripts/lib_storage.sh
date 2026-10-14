#!/usr/bin/env bash
# Resolves where benchmark results get written. Precedence:
#   1. $COMMBENCH_RESULTS_DIR env var
#   2. first non-comment line of <repo>/.storage-config (gitignored, machine-local override)
#   3. <repo>/results (the default -- keeps a fresh clone fully self-contained)
resolve_results_root() {
    if [[ -n "${COMMBENCH_RESULTS_DIR:-}" ]]; then
        echo "$COMMBENCH_RESULTS_DIR"
        return
    fi
    local config_file="$PROJECT_ROOT/.storage-config"
    if [[ -f "$config_file" ]]; then
        local line
        while IFS= read -r line || [[ -n "$line" ]]; do
            line="${line%%#*}"
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            if [[ -n "$line" ]]; then
                echo "$line"
                return
            fi
        done <"$config_file"
    fi
    echo "$PROJECT_ROOT/results"
}
