# How to train or inference in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Yk7dF2YpVsYV6XRJWsM1HKWoWXXKRT1T?usp=sharing)

#### You can check Jupyter Notebook Version at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kdBMySPzX4wxI_I9xJzguFKxx99HBz-T?usp=sharing) 
 

## install
#### [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
```python
$ pip install -qqq torchmetrics
```

#### [colorama](https://github.com/tartley/colorama)
```python
$ pip install -qqq colorama
```

#### HuggingFace Transformer
```python
$ pip install -qqq --no-cache-dir transformers sentencepiece
```


## Train 
```python

$ python train.py --base_path './data/' \
                  --model_save '/content/drive/MyDrive/ ... /Dacon Shopping/' \
                  --sub_path '/content/drive/MyDrive/ ... /Dacon Shopping/' \
                  --model "beomi/KcELECTRA-base" \
                  --grad_clipping True\
                  --n_folds 5 \
                  --n_epochs 5 \
                  --device 'cuda' \
                  --train_bs 64

```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: Huggingface Pratrained Model (Default: `"beomi/KcELECTRA-base"`)
- `n_folds`  : Fold 수
- `n_epochs` : Epoch
- `seed` : Random Seed (Default: 2022)
- `train_bs` : Batch Size (Default: 16)
- `max_length` : Max Length (Default: 128) for HuggingFace Tokenizer
- `grad_clipping`: [Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size 결정
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- `learning_rate`, `weight_decay`, `min_lr`, `T_max` 등은 생략 
- [`train.py`](https://github.com/renslightsaber/Dacon_Sentence_Type_Classification/blob/main/train.py) 참고!   


## 주의
 - CLI 환경에서 train 시킬 때, `tqdm`의 Progress Bar가 엄청 많이 생성된다. 아직 원인과 해결을 못 찾은 상태이다.
 - Colab과 Jupyter Notebook에서는 정상적으로 Progress Bar가 나타난다.



## Inference 
```python

$ python inference.py --base_path './data/' \
                      --model_save '/content/drive/MyDrive/ .. /Dacon Shopping/' \
                      --sub_path '/content/drive/MyDrive/ ... /Dacon Shopping/' \
                      --model "beomi/KcELECTRA-base" \
                      --n_folds 5 \
                      --n_epochs 5 \
                      --device 'cuda' \
                      --train_bs 64 

```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: train했을 때의 Huggingface Pratrained Model (Default: `"beomi/KcELECTRA-base"`)
- `n_folds`  : `train.py`에서 진행항 KFold 수
- `n_epochs` : train 했을 때의 Epoch 수 (submission 파일명에 사용)  
- `seed` : Random Seed (Default: 2022)
- `train_bs` : Batch Size (Default: 16) 
- `max_length` : Max Length (Default: 128) for HuggingFace Tokenizer
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.






