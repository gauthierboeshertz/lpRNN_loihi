CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main.py  --nlayers 2 --nunits 512  --lr 0.001   --nepochs 50 --lpff_size 128  --batch_size 128  --ret_ratio 0.8 --frac_bits 3  --wth 2 --seed 0 --augment

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main.py  --task_n_words 12 --v2  --nlayers 2 --nunits 512  --lr 0.001   --nepochs 50 --lpff_size 128  --batch_size 128  --ret_ratio 0.8 --frac_bits 3  --wth 2 --seed 0 --augment

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main.py  --task_n_words 35 --v2  --nlayers 2 --nunits 512  --lr 0.001   --nepochs 50 --lpff_size 128   --batch_size 128  --ret_ratio 0.8 --frac_bits 3  --wth 2 --seed 0 --augment






