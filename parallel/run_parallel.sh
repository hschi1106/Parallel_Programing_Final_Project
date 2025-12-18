#!/bin/bash

# 定義 benchmark 名稱 (決定執行順序)
ORDERED_BENCHMARKS=(
    "keijzer7"
    "nguyen6"
    "nguyen9"
    "feynman-1-26-2"
    "feynman-1-47-23"
)

# 定義每個 benchmark 對應的參數
declare -A PARAMS
PARAMS["keijzer7"]=1
PARAMS["nguyen6"]=1
PARAMS["nguyen9"]=2
PARAMS["feynman-1-26-2"]=2
PARAMS["feynman-1-47-23"]=3

# 設定路徑
BENCH_DIR="../benchmarks"
EXE="./gpg_parallel"

# 建立 logs 資料夾
mkdir -p logs

echo "=== Starting Parallel Experiments with Specific Params ==="

for bench in "${ORDERED_BENCHMARKS[@]}"; do
    # 取得該 benchmark 對應的參數
    P_VAL=${PARAMS[$bench]}
    
    # 組合檔案路徑
    TRAIN_FILE="${BENCH_DIR}/${bench}_train.txt"
    TEST_FILE="${BENCH_DIR}/${bench}_test.txt"
    
    # 定義 Log 檔案名稱
    LOG_FILE="logs/parallel_${bench}.log"
    
    echo "Running ${bench} with param ${P_VAL}..."
    
    # 執行指令: ./gpg_parallel [train] [test] [param]
    $EXE $TRAIN_FILE $TEST_FILE $P_VAL > "$LOG_FILE" 2>&1
done

echo "=== All Parallel Experiments Completed ==="