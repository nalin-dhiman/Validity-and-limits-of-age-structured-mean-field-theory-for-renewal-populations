#!/bin/bash
set -euo pipefail

LOG_FILE="phaseIX_suite.log"
echo ">>> Starting Phase IX Full Suite at $(date) <<<" | tee -a $LOG_FILE

run_step() {
    local STEP_NAME="$1"
    shift
    
    # We group the header and the command execution, divert stderr to stdout, 
    # and pipe everything to tee.
    # 'set -o pipefail' (set at top) ensures that if "$@" fails, the pipeline fails.
    {
        echo ""
        echo "---------------------------------------------------"
        echo "$STEP_NAME"
        echo "---------------------------------------------------"
        "$@"
    } 2>&1 | tee -a "$LOG_FILE"
}

# 1. Validation (Sanity Check)
# Clean previous validation artifacts
rm -rf results/validate_pde.npz results/validate_scaling

run_step "STEP 1: VALIDATION (Conservation & Scaling Sanity)" bash scripts/validate_phaseIX.sh

# 2. Bias Benchmark
run_step "STEP 2: BIAS BENCHMARK (T=60s, 5 Seeds)" python3 src/check_bias_fix.py

# 2.5 Scaling Analysis
run_step "STEP 2.5: FULL SCALING ANALYSIS (N=16k, T=60s, Offset Fit)" python3 src/run_scaling.py --out_dir results/scaling_full

# 3. Variance Check
run_step "STEP 3: VARIANCE CHECK (N=50k, OU Baseline)" python3 src/check_variance.py

# 4. Phase Scan
echo "---------------------------------------------------" | tee -a $LOG_FILE
echo "PRE-FLIGHT: Checking Phase Scan Syntax..." | tee -a $LOG_FILE
python3 -m py_compile src/run_phase_scan.py
if [ $? -ne 0 ]; then
    echo "!!! CRITICAL: Syntax Error in run_phase_scan.py" | tee -a $LOG_FILE
    exit 1
fi

run_step "STEP 4: PHASE SCAN (Full Grid, 3 Seeds)" python3 src/run_phase_scan.py --out_dir results/phase_scan --clean

# 5. Success Validation
echo "---------------------------------------------------" | tee -a $LOG_FILE
echo "VALIDATING PHASE SCAN COMPLETION..." | tee -a $LOG_FILE

STATUS_CSV="results/phase_scan/status.csv"
if [ ! -f "$STATUS_CSV" ]; then
    echo "!!! FAILURE: status.csv not found in results/phase_scan/" | tee -a $LOG_FILE
    exit 1
fi

# Count Errors
ERR_COUNT=$(grep -cE "PIPELINE_ERROR|IO_ERROR" "$STATUS_CSV" || true)
TOTAL_COUNT=$(wc -l < "$STATUS_CSV")
TOTAL_COUNT=$((TOTAL_COUNT - 1)) # Header

if [ "$ERR_COUNT" -gt 0 ]; then
    PERCENT=$(echo "scale=2; $ERR_COUNT * 100 / $TOTAL_COUNT" | bc)
    echo "!!! WARNING: $ERR_COUNT errors ($PERCENT%) detected in Phase Scan." | tee -a $LOG_FILE
    
    # Threshold < 2%
    if (( $(echo "$PERCENT > 2.0" | bc -l) )); then
         echo "!!! FAILURE: Error rate too high (>2%). Aborting success status." | tee -a $LOG_FILE
         exit 1
    else
         echo "Acceptable error rate (<2%). Proceeding." | tee -a $LOG_FILE
    fi
else
    echo "Phase Scan Clean: 0 Pipeline Errors." | tee -a $LOG_FILE
fi

echo "" | tee -a $LOG_FILE
echo ">>> ALL STEPS COMPLETED SUCCESSFULY at $(date) <<<" | tee -a $LOG_FILE
