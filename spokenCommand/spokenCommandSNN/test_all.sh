python3 -u gen_loihi_inputs.py --nthreads 20   --frac_bits 3 --wth 0   --nunits 512  --augment

python3 -u gen_loihi_inputs.py --nthreads 20   --frac_bits 3 --wth 1   --nunits 512  --augment

python3 -u gen_loihi_inputs.py --nthreads 20   --frac_bits 3 --wth 0   --nunits 512  --augment --task_n_words 12 

python3 -u gen_loihi_inputs.py --nthreads 20   --frac_bits 3 --wth 1   --nunits 512  --augment --task_n_words 12 

python3 -u gen_loihi_inputs.py --nthreads 20   --frac_bits 3 --wth 0   --nunits 512  --augment --task_n_words 35 

python3 -u gen_loihi_inputs.py --nthreads 20   --frac_bits 3 --wth 1   --nunits 512  --augment --task_n_words 35 
