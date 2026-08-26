nohup cargo test --release test_recall_precision_real_data -- --nocapture > test_output.log 2>&1 &

# Note the process ID for later if needed
echo $! > test_pid.txt