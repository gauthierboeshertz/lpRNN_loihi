Folder for HD dataset

Go to  https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/ to download the dataset and put it in the shdANN/data folder.

The trained weights are already provided, they were obtained by runnning the following command:
```python3 main.py --frac_bits 3 --nunits 512 --use_mels --wth 1 ```

Then generate the data and the weights for the SNN:

``` python3 gen_weight_data_for_snn.py --hop_length 256 --nunits 512  --use_mels --wth 0 ```

To run the SNN on the HD data go in shdSNN and run the following command:

```python3 -u gen_loihi_inputs.py  --nthreads 5  --wth 0 --frac_bits 3 --nunits 512 ```
