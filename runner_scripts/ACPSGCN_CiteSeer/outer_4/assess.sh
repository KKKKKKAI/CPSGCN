python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_4/model_assess/iteration_0 --dataset_name CiteSeer --w_lr 0.03 --adj_lr 0.001 --prune_ratio 10 --preserve_rate 97 --centrality CC_EC --outer_k 4 --inner_k 1 --ac_select minmax --test_run
python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_4/model_assess/iteration_1 --dataset_name CiteSeer --w_lr 0.03 --adj_lr 0.001 --prune_ratio 10 --preserve_rate 97 --centrality CC_EC --outer_k 4 --inner_k 1 --ac_select minmax --test_run
python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_4/model_assess/iteration_2 --dataset_name CiteSeer --w_lr 0.03 --adj_lr 0.001 --prune_ratio 10 --preserve_rate 97 --centrality CC_EC --outer_k 4 --inner_k 1 --ac_select minmax --test_run