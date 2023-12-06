### Session-based Recommendation

We implement our model ```FAPAT``` in `fapat` for session-based recommendations with frequent attribute patterns.
We also implement many baselines:
* ```FPMC``` in `fpmc.py`
* ```GRU4Rec``` in `gru4rec.py`
* ```NARM``` in `narm.py`
* ```STAMPT``` in `stamp.py`
* ```CSRM``` in `csrm.py`
* ```MT2Rec``` in `mt2rec.py`
* ```GRAPHFORMER``` in `graphformer.py`
* ```GNN``` in `gnn.py`
* ```SRGNN``` in `srgnn.py`
* ```GCSAN``` in `gcsan.py`
* ```GCEGNN``` in `gce_gnn.py`
* ```DHCN``` (aka. ```S2-DHCN```) in `dhcn.py`
* ```NCL``` in `ncl.py`
* ```LESSR``` in `lessr.py`
* ```MSGIFSR``` in `msgifsr.py`

To train models on public data, please run the following commands:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gru4rec
CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gru4rec

CUDA_VISIBLE_DEVICES=2 python train.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model mt2rec --mtl --attributes brand category
CUDA_VISIBLE_DEVICES=3 python train.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model mt2rec --mtl --attributes category

CUDA_VISIBLE_DEVICES=4 python train.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gce_gnn
CUDA_VISIBLE_DEVICES=5 python train.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gce_gnn

CUDA_VISIBLE_DEVICES=6 python train.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model fapat --attributes brand category
CUDA_VISIBLE_DEVICES=7 python train.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model fapat --attributes category
```

And the evaluation can be conducted by:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gru4rec
CUDA_VISIBLE_DEVICES=1 python eval.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gru4rec

CUDA_VISIBLE_DEVICES=2 python eval.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model mt2rec --mtl --attributes brand category
CUDA_VISIBLE_DEVICES=3 python eval.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model mt2rec --mtl --attributes category

CUDA_VISIBLE_DEVICES=4 python eval.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gce_gnn
CUDA_VISIBLE_DEVICES=5 python eval.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model gce_gnn

CUDA_VISIBLE_DEVICES=6 python eval.py --batch_size 200 --dataset Tmall --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model fapat --attributes brand category
CUDA_VISIBLE_DEVICES=7 python eval.py --batch_size 200 --dataset diginetica --n_iter 1 --mem_size 12 --batch_size 200 --seed 2022 --prepad --model fapat --attributes category
```

For experiments on internal data, you can run the code:
```bash
CUDA_VISIBLE_DEVICES=5 python train_amazon.py --dataset Amazon_nips2023 --category beauty --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes browse brand color_text size_name
CUDA_VISIBLE_DEVICES=6 python train_amazon.py --dataset Amazon_nips2023 --category books --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes browse brand genre author
CUDA_VISIBLE_DEVICES=7 python train_amazon.py --dataset Amazon_nips2023 --category electronics --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes product_type browse brand color_text
```

But we strongly recommend you to run through distributed data parallel (DDP):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 41111 \
train_amazon_dist.py --dataset Amazon_nips2023 --category electronics --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes product_type browse brand color_text
```

And the evaluation can be evaluated by `eval_amazon.py` or `eval_amazon_dist.py`.
```bash
CUDA_VISIBLE_DEVICES=7 python eval_amazon.py --dataset Amazon_nips2023 --category electronics --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes product_type browse brand color_text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 41111 \
eval_amazon_dist.py --dataset Amazon_nips2023 --category electronics --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes product_type browse brand color_text
```

### Attribute Estimations and Period-Item Recommendations

We also extend the performance evaluation to attribute estimations and period-item recommendations, where the former evaluate the hits and MRR of each attribute, and the latter reviews the aligment of long-period predictions via recall and NDCG.
This can be done by `eval_attr.py` and `eval_period.py`.
```bash
CUDA_VISIBLE_DEVICES=7 python eval_attr.py --dataset Amazon_nips2023 --category electronics --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes product_type browse brand color_text
CUDA_VISIBLE_DEVICES=7 python eval_period.py --dataset Amazon_nips2023 --category electronics --n_iter 1 --mem_size 12 --batch_size 500 --seed 2022 --prepad --model fapat --attributes product_type browse brand color_text
```