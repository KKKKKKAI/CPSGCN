python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_1/model_assess/iteration_0 --dataset_name CiteSeer --w_lr 0.01 --adj_lr 0.0001 --prune_ratio 50 --preserve_rate 99 --centrality DC_EC --outer_k 1 --inner_k 1 --ac_select minmax --test_run
python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_1/model_assess/iteration_1 --dataset_name CiteSeer --w_lr 0.01 --adj_lr 0.0001 --prune_ratio 50 --preserve_rate 99 --centrality DC_EC --outer_k 1 --inner_k 1 --ac_select minmax --test_run
python ACPSGCN.py --use_gpu --total_epochs 400 --dest results/ACPSGCN_CiteSeer/outer_fold_1/model_assess/iteration_2 --dataset_name CiteSeer --w_lr 0.01 --adj_lr 0.0001 --prune_ratio 50 --preserve_rate 99 --centrality DC_EC --outer_k 1 --inner_k 1 --ac_select minmax --test_run
