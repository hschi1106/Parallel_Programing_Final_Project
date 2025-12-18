#!/bin/bash

# 定義 benchmark 名稱 (決定執行順序)
ORDERED_BENCHMARKS=(
    "keijzer7"
    "nguyen6"
    "nguyen9"
    "feynman-1-26-2"
    "feynman-1-47-23"
)

# 定義每個 benchmark 對應的參數 (Key-Value Mapping)
declare -A PARAMS
PARAMS["keijzer7"]=1
PARAMS["nguyen6"]=1
PARAMS["nguyen9"]=2
PARAMS["feynman-1-26-2"]=2
PARAMS["feynman-1-47-23"]=3

# 設定路徑
BENCH_DIR="../benchmarks"
EXE="./gpg_serial"

# 建立 logs 資料夾
mkdir -p logs

echo "=== Starting Serial Experiments with Specific Params ==="

for bench in "${ORDERED_BENCHMARKS[@]}"; do
    # 取得該 benchmark 對應的參數
    P_VAL=${PARAMS[$bench]}
    
    # 組合檔案路徑
    TRAIN_FILE="${BENCH_DIR}/${bench}_train.txt"
    TEST_FILE="${BENCH_DIR}/${bench}_test.txt"
    
    # 定義 Log 檔案名稱
    LOG_FILE="logs/serial_${bench}.log"
    
    echo "Running ${bench} with param ${P_VAL}..."
    
    # 執行指令: ./gpg_serial [train] [test] [param]
    $EXE $TRAIN_FILE $TEST_FILE $P_VAL > "$LOG_FILE" 2>&1
done

echo "=== All Serial Experiments Completed ==="