python benchmark_models/sgcn/sgcn.py --use_gpu --epochs 400 --dest results/SGCN_CiteSeer/outer_fold_1/model_assess/iteration_0 --dataset CiteSeer --ratio_weight 70 --ratio_graph 90 --w_lr 0.02 --adj_lr 0.001 --outer_k 1 --inner_k 1 --test_run
python benchmark_models/sgcn/sgcn.py --use_gpu --epochs 400 --dest results/SGCN_CiteSeer/outer_fold_1/model_assess/iteration_1 --dataset CiteSeer --ratio_weight 70 --ratio_graph 90 --w_lr 0.02 --adj_lr 0.001 --outer_k 1 --inner_k 1 --test_run
python benchmark_models/sgcn/sgcn.py --use_gpu --epochs 400 --dest results/SGCN_CiteSeer/outer_fold_1/model_assess/iteration_2 --dataset CiteSeer --ratio_weight 70 --ratio_graph 90 --w_lr 0.02 --adj_lr 0.001 --outer_k 1 --inner_k 1 --test_run