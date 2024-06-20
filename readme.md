python3 pytorch_train.py --epochs 10 --dataset OrganC

python3 pytorch_prune_weight_cotrain.py --ratio_graph 60 --ratio_weight 60 --dataset OrganC

python3 pytorch_retrain_with_graph.py --load_path prune_weight_cotrain/model.pth.tar --dataset OrganC
