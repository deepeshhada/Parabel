dataset="EURLex-4K"
weighted="1"
tail_classifier="1"
per_label_predict="0"

results_dir="./Results/$dataset"
model_dir="$results_dir/model"
data_dir="../Datasets/$dataset"

mkdir -p $model_dir

./xreg_train.cpp $model_dir $data_dir/trn_X_Xf.txt $data_dir/trn_X_Y.txt -s 0 -kleaf 1 -k 1 -w $weighted -t 3 -T 4 -m 100 -r $tail_classifier -tcl 0.05 -ecl 0.1 -n 20 -c 10

./xreg_predict.cpp $model_dir $results_dir/score_mat.txt $data_dir/tst_X_Xf.txt -s 0 -T 4 -p $per_label_predict -r $tail_classifier -a 0.5

./xreg_metric.cpp $results_dir/score_mat.txt $data_dir/tst_X_Y.txt 5 $weighted