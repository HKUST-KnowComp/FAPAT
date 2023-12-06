### Data Download/Collect

#### Public Data

* ```Tmall``` data can be downloaded from [data_format1.zip](https://tianchi.aliyun.com/dataset/42) and [tmall.zip](https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AAAMMlmNKL-wAAYK8QWyL9MEa/Datasets?dl=0&subfolder_nav_tracking=1). Download both, unzip them, put ```user_log_format1.csv``` and ```dataset15.csv``` into the ```Tmall``` folder.
* ```diginetica``` data can be downloaed from [dataset-train-diginetica.zip](https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw). Download it, unzip it, put ```train-item-views.csv``` and ```product-categories.csv``` into the ```diginetica``` folder.

#### Amazon Data

We collect and clean the session data from Amazon. The data is under the progress of legal approval.

### Data Split

We provide scripts to split public data and Amazon data: `process_tmall.py`, `process_diginetica.py`, `process_amazon.py`. To reproduce the experiments, please run `process_all.sh`.
