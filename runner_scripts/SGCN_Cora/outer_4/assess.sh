python benchmark_models/sgcn/sgcn.py --use_gpu --epochs 400 --dest results/SGCN_Cora/outer_fold_4/model_assess/iteration_0 --dataset Cora --ratio_weight 90 --ratio_graph 70 --w_lr 0.03 --adj_lr 0.001 --outer_k 4 --inner_k 1 --test_run
python benchmark_models/sgcn/sgcn.py --use_gpu --epochs 400 --dest results/SGCN_Cora/outer_fold_4/model_assess/iteration_1 --dataset Cora --ratio_weight 90 --ratio_graph 70 --w_lr 0.03 --adj_lr 0.001 --outer_k 4 --inner_k 1 --test_run
python benchmark_models/sgcn/sgcn.py --use_gpu --epochs 400 --dest results/SGCN_Cora/outer_fold_4/model_assess/iteration_2 --dataset Cora --ratio_weight 90 --ratio_graph 70 --w_lr 0.03 --adj_lr 0.001 --outer_k 4 --inner_k 1 --test_run
