#!/usr/bin/env bash
# Run all models in parallel waves with live progress monitoring.
# Usage: bash scripts/run_parallel.sh
#
# Compatible with macOS bash 3.2 (no associative arrays).
# Safe to Ctrl+C — checkpointing means no work is lost.
# Re-run to resume from where each model left off.

set -euo pipefail
cd "$(dirname "$0")/.."

PROJ_DIR="$(pwd)"
CLI="${PROJ_DIR}/.venv/bin/cot-faithfulness"
BUDGET=75
CONCURRENCY=5
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"

timestamp() { date "+%H:%M:%S"; }

header() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}${CYAN}  $1${RESET}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

count_lines() {
    if [ -f "$1" ]; then
        wc -l < "$1" | tr -d ' '
    else
        echo "0"
    fi
}

model_progress() {
    local model="$1"
    local total=0
    local base=$(count_lines "results/base/${model}.jsonl")
    if [ "$base" -gt 498 ]; then base=498; fi
    total=$((total + base))
    for hint in sycophancy consistency visual_pattern metadata grader unethical; do
        local c=$(count_lines "results/hinted/${model}/${hint}.jsonl")
        if [ "$c" -gt 498 ]; then c=498; fi
        total=$((total + c))
    done
    echo "$total"
}

progress_bar() {
    local current=$1 total=$2 width=20
    if [ "$total" -eq 0 ]; then
        printf "[                    ]   0%%"
        return
    fi
    local pct=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    printf "["
    local i=0
    while [ $i -lt $filled ]; do printf "█"; i=$((i+1)); done
    i=0
    while [ $i -lt $empty ]; do printf "░"; i=$((i+1)); done
    printf "] %3d%%" "$pct"
}

# ── Wave runner ──────────────────────────────────────────────────────────────

run_wave() {
    local wave_name="$1"
    shift
    local models=("$@")
    local wave_start=$(date +%s)
    local model_target=3486  # 498 * 7

    header "$wave_name — ${#models[@]} models"
    echo -e "  ${DIM}Budget: \$${BUDGET}/model | Concurrency: ${CONCURRENCY} | $(timestamp)${RESET}"
    echo ""

    # Check which models need running
    local models_to_run=()
    local pids=()
    for model in "${models[@]}"; do
        local prog=$(model_progress "$model")
        if [ "$prog" -ge "$model_target" ]; then
            echo -e "  ${GREEN}✓${RESET} ${BOLD}${model}${RESET} — already complete (${prog}/${model_target})"
        else
            local remaining=$((model_target - prog))
            echo -e "  ${CYAN}▶${RESET} ${BOLD}${model}${RESET} — ${remaining} remaining (${prog} cached)"
            models_to_run+=("$model")
        fi
    done

    if [ ${#models_to_run[@]} -eq 0 ]; then
        echo -e "\n  ${GREEN}All models in this wave already complete.${RESET}"
        return
    fi

    echo ""

    # Launch background processes
    for model in "${models_to_run[@]}"; do
        "$CLI" run-fast \
            --model "$model" \
            --budget "$BUDGET" \
            --concurrency "$CONCURRENCY" \
            > "$LOG_DIR/${model}.log" 2>&1 &
        pids+=($!)
    done

    echo -e "  ${DIM}Monitoring progress every 15s... (Ctrl+C to stop, re-run to resume)${RESET}"
    echo ""

    # Monitor loop
    local all_done=false
    while ! $all_done; do
        sleep 15

        all_done=true
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                all_done=false
                break
            fi
        done

        local i=0
        while [ $i -lt ${#models_to_run[@]} ]; do
            local model="${models_to_run[$i]}"
            local pid="${pids[$i]}"
            local current=$(model_progress "$model")

            local icon
            if kill -0 "$pid" 2>/dev/null; then
                icon="${YELLOW}⟳${RESET}"
            else
                wait "$pid" 2>/dev/null && icon="${GREEN}✓${RESET}" || icon="${RED}✗${RESET}"
            fi

            printf "  %b %-28s %s %4d/%d\n" \
                "$icon" "$model" \
                "$(progress_bar "$current" "$model_target")" \
                "$current" "$model_target"

            i=$((i+1))
        done

        if ! $all_done; then
            # Move cursor up to overwrite on next refresh
            local j=0
            while [ $j -lt ${#models_to_run[@]} ]; do
                printf "\033[1A\033[2K"
                j=$((j+1))
            done
        fi
    done

    # Final timing
    local wave_end=$(date +%s)
    local elapsed=$(( wave_end - wave_start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    echo ""
    echo -e "  ${DIM}Wave completed in ${mins}m ${secs}s${RESET}"

    # Report failures
    local failed=0
    local i=0
    while [ $i -lt ${#models_to_run[@]} ]; do
        local model="${models_to_run[$i]}"
        local pid="${pids[$i]}"
        if ! wait "$pid" 2>/dev/null; then
            local last_line
            last_line=$(tail -1 "$LOG_DIR/${model}.log" 2>/dev/null || echo "unknown error")
            echo -e "  ${RED}✗ ${model}:${RESET} $last_line"
            echo -e "    ${DIM}Log: $LOG_DIR/${model}.log${RESET}"
            failed=$((failed + 1))
        fi
        i=$((i+1))
    done

    if [ "$failed" -eq 0 ]; then
        echo -e "  ${GREEN}All ${#models_to_run[@]} models succeeded.${RESET}"
    else
        echo -e "  ${YELLOW}${failed}/${#models_to_run[@]} had errors (re-run to retry).${RESET}"
    fi
}

# ── Model waves ──────────────────────────────────────────────────────────────

WAVE1=(
    "ernie-4.5-21b-thinking"
    "gpt-oss-120b"
    "nemotron-nano-9b"
    "olmo-3-7b-think"
    "olmo-3.1-32b-think"
    "seed-1.6-flash"
    "step-3.5-flash"
)

WAVE2=(
    "qwq-32b"
    "minimax-m2.5"
    "qwen3.5-27b"
    "deepseek-v3.2-speciale"
)

WAVE3=(
    "deepseek-r1"
    # "glm-5"       # skipped — ~$52 estimated cost ($2.56/M output)
    # "kimi-k2.5"   # skipped — ~$45 estimated cost ($2.20/M output)
)

# ── Main ─────────────────────────────────────────────────────────────────────

OVERALL_START=$(date +%s)

header "CoT Faithfulness — Parallel Runner"
echo -e "  ${DIM}Started: $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo -e "  ${DIM}Budget: \$${BUDGET}/model | Concurrency: ${CONCURRENCY}/model${RESET}"
echo -e "  ${DIM}Models: ${#WAVE1[@]} + ${#WAVE2[@]} + ${#WAVE3[@]} = $(( ${#WAVE1[@]} + ${#WAVE2[@]} + ${#WAVE3[@]} )) total${RESET}"

run_wave "Wave 1 — Cheap/fast" "${WAVE1[@]}"
run_wave "Wave 2 — Mid-tier" "${WAVE2[@]}"
run_wave "Wave 3 — Expensive" "${WAVE3[@]}"

OVERALL_END=$(date +%s)
TOTAL_ELAPSED=$(( OVERALL_END - OVERALL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

header "Run Complete — ${TOTAL_MINS}m ${TOTAL_SECS}s total"
echo ""
"$CLI" status
echo ""
