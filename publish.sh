#!/bin/zsh
# Publish all workspace crates to crates.io.
#
# Default mode uses cargo's native workspace publish (cargo >= 1.90), which
# computes the dependency-correct order itself and verifies each crate against
# the in-flight set. If a run fails partway (network, rate limit), use
# --resume: it walks the tiered list below sequentially and skips crates whose
# version is already on crates.io.
#
# Internal dev-deps are path-only (no version), so cargo strips them from
# published manifests — only normal/build deps constrain publish order.
#
# Usage:
#   ./publish.sh           # native workspace publish (recommended)
#   ./publish.sh --dry-run # native publish, no upload
#   ./publish.sh --resume  # sequential tiered publish, skip already-published

set -euo pipefail

CRATES=(
  # Tier 1: no internal library deps
  adk-core
  adk-anthropic
  adk-deploy
  adk-enterprise
  adk-rust-macros
  adk-telemetry
  awp-types

  # Tier 2: depends on Tier 1
  adk-action
  adk-artifact
  adk-awp
  adk-browser
  adk-gemini
  adk-guardrail
  adk-memory
  adk-mistralrs
  adk-plugin
  adk-sandbox
  adk-session

  # Tier 3: depends on Tier 1-2
  adk-code
  adk-graph
  adk-model
  adk-rag
  adk-realtime
  adk-retry-reflect
  adk-skill

  # Tier 4: depends on Tier 1-3
  adk-runner
  adk-tool
  adk-agent
  adk-audio

  # Tier 5: depends on Tier 1-4
  adk-acp
  adk-eval
  adk-managed
  adk-server

  # Tier 6: depends on Tier 1-5
  adk-auth
  adk-bench
  adk-cli

  # Tier 7: depends on Tier 1-6
  adk-payments
  cargo-adk

  # Tier 8: umbrella (depends on everything)
  adk-rust
)

MODE="native"
DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    --resume)  MODE="resume" ;;
    --dry-run) DRY_RUN="--dry-run" ;;
    *) echo "unknown flag: $arg"; exit 2 ;;
  esac
done

if [[ "$MODE" == "native" ]]; then
  echo "=== Publishing ADK-Rust (cargo publish --workspace) ==="
  echo "Crates: ${#CRATES[@]}  $DRY_RUN"
  echo "If this fails partway, finish with: ./publish.sh --resume"
  echo ""
  cargo publish --workspace ${DRY_RUN:+$DRY_RUN}
  echo "✅ Done"
  exit 0
fi

# ── Resume mode: sequential tiered publish, skipping published crates ──────

echo "=== Publishing ADK-Rust (sequential resume) ==="
echo "Total crates: ${#CRATES[@]}"
echo ""

PUBLISHED=0
SKIPPED=0
FAILED=0
FAILED_CRATES=()

for crate in "${CRATES[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📦 [$((PUBLISHED + SKIPPED + FAILED + 1))/${#CRATES[@]}] Publishing: $crate"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  STATUS=0
  OUTPUT=$(cargo publish -p "$crate" 2>&1) || STATUS=$?

  echo "$OUTPUT"
  echo ""

  if echo "$OUTPUT" | grep -q "already exists\|already uploaded"; then
    echo "⏭  Already published"
    SKIPPED=$((SKIPPED + 1))
    sleep 1
  elif [ $STATUS -eq 0 ]; then
    echo "✅ Published"
    PUBLISHED=$((PUBLISHED + 1))
    echo "⏳ Waiting for crates.io indexing..."
    sleep 15
  else
    echo "❌ FAILED (exit $STATUS)"
    FAILED=$((FAILED + 1))
    FAILED_CRATES+=("$crate")
  fi

  echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "=== SUMMARY ==="
echo "✅ Published: $PUBLISHED"
echo "⏭  Skipped:   $SKIPPED"
echo "❌ Failed:    $FAILED"

if [ ${#FAILED_CRATES[@]} -gt 0 ]; then
  echo ""
  echo "Failed crates:"
  for c in "${FAILED_CRATES[@]}"; do
    echo "- $c"
  done
  exit 1
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
