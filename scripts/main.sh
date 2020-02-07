PYTHON='python'
echo "Without GCC:"
data_name='mag-acm-full'
test_ratio=0.1
num_exp=2
$PYTHON ../src/data_preparer.py --data-name $data_name --silent --test-ratio $test_ratio --num-exp $num_exp
$PYTHON ../src/link_predictor.py --data-name $data_name --silent --num-exp $num_exp

echo "With GCC:"
$PYTHON ../src/data_preparer.py --data-name $data_name --gcc --silent --test-ratio $test_ratio --num-exp $num_exp
$PYTHON ../src/link_predictor.py --data-name $data_name --silent --num-exp $num_exp
