# pos-tagger-using-tensorflow

feed forward 모델을 이용하여 pos tagger를 합니다.  

## data
학습에 쓰이는 데이터의 형식은 `형태소/품사`쌍을 이루는 문장입니다.
```
0.1%/N 최상위/N 근로소득자보다/N_j 9.1%포인트/N 더/B 높은/V_e 증가율이다/N_c_e ./q 
0.8평/N 소형/N 매장도/N_j 1년/N 임대료가/N_j 8400만/N_j 원에/N_j 달했다/V_f_e ./q 
0보다/N_j 낮으면/V_e 그/D 반대/N 의미를/N_j 갖는다/V_e ./q 
1%/N 미만의/N_j 소수주주/N 지분/N 약/D 50%도/N_j 고려됐다/N_t_f_e ./q 
```

## train
데이터를 `src/data`에 위치를 한 후, [train.py](https://github.com/YoungHunCho/pos-tagger-using-tensorflow/blob/master/src/train.py)의 `main`함수에서 입려 파일을 명시르 합니다.  
입력 파일 설정 후. 
> python3 train.py  
train.py를 실행합니다.  
학습이 완료되면 학습 모델은 `checkpoints`폴더에 저장이 되며, 그 모델을 이용하여 실행을 하게 됩니다.  

## run
학습을 완료 한 후에는. 
> python3 run.py  
run.py를 실행하여 해당 결과를 확인 할 수 있습니다.

## result
![](https://github.com/YoungHunCho/pos-tagger-using-tensorflow/blob/master/doc/result.jpg)
