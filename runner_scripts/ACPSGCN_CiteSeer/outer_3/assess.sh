python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_3/model_assess/iteration_0 --dataset_name CiteSeer --w_lr 0.02 --adj_lr 0.001 --prune_ratio 50 --preserve_rate 95 --centrality CC_EC --outer_k 3 --inner_k 1 --ac_select zscore --test_run
python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_3/model_assess/iteration_1 --dataset_name CiteSeer --w_lr 0.02 --adj_lr 0.001 --prune_ratio 50 --preserve_rate 95 --centrality CC_EC --outer_k 3 --inner_k 1 --ac_select zscore --test_run
python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_3/model_assess/iteration_2 --dataset_name CiteSeer --w_lr 0.02 --adj_lr 0.001 --prune_ratio 50 --preserve_rate 95 --centrality CC_EC --outer_k 3 --inner_k 1 --ac_select zscore --test_run
