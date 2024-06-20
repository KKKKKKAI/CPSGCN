python CPSGCN.py --use_gpu --dataset_name Cora --centrality RAND --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/RAND/ 
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/CC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality DC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/EC/ --ac_select zscore

python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC_CC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_DC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC_DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC_DC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/CC_DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC_EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/CC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality DC_EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/DC_EC/ --ac_select zscore

python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC_DC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC_CC_DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC_EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC_CC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_DC_EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC_DC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC_DC_EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/CC_DC_EC/ --ac_select zscore

python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC_DC_EC --preserve_rate 95 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P95/BC_CC_DC_EC/ --ac_select zscore

python CPSGCN.py --use_gpu --dataset_name Cora --centrality RAND --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/RAND/ 
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/CC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality DC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/EC/ --ac_select zscore

python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC_CC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_DC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC_DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC_DC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/CC_DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC_EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/CC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality DC_EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/DC_EC/ --ac_select zscore

python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC_DC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC_CC_DC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC_EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC_CC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_DC_EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC_DC_EC/ --ac_select zscore
python ACPSGCN.py --use_gpu --dataset_name Cora --centrality CC_DC_EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/CC_DC_EC/ --ac_select zscore

python ACPSGCN.py --use_gpu --dataset_name Cora --centrality BC_CC_DC_EC --preserve_rate 97 --preserve_duration 3  --dest results/ZACPSGCN_Cora_Table/P97/BC_CC_DC_EC/ --ac_select zscore