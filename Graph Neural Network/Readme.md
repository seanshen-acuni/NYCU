---
title: IntroToAI_HW5

---

## Introduction to Artificial Intelligence Homework 5
```
Editor: 沈昱翔
Student ID: 110612008
Email: seanshen92328.c@nycu.edu.tw
```
### 開始
##### 套件
> torch>=2.5.1
> numpy>=1.26.3
> pandas>=2.2.3
> tqdm>=4.67.0
```
pip install torch
pip install numpy
pip install pandas
pip install ast
pip install tqdm
```
##### 建議版本
> python 3.12.0
> 輸入以下程式來確認目前環境版本
 ```
 python --version
```
##### 檔案路徑和說明
> 110612008.csv: 使用模型預測 private_test.csv和public_test.csv 的結果
> 110612008.pth: 訓練完成的模型（用pytorch(torch.save())存取）
> 110612008.pdf: 本次project報告
>Readme.md: 自述文件
> requirments.txt: 環境套件和版本
> 輸入以下內容來解壓縮檔案：
```
cd {location}
tar -xf 110612008.zip
```
### 運行專案
1.確認以下檔案皆在與程式相同的資料夾:
> train.csv
rebbit_graph.csv
private_test.csv
public_test.csv

你可以輸入以下指令:
```
cd 110612008
```
for windows:
```
dir /b | findstr /E /C:"train.csv" /C:"rebbit_graph.csv" /C:"private_test.csv" /C:"public_test.csv"
```
for Unix/Linux/macOS:
```
ls | grep -E '^(train.csv|rebbit_graph.csv|private_test.csv|public_test.csv)$'
```

2.執行 110612008.py。
```
python 110612008.py
```
### 程式功能
1. 讀取四個 CSV 檔案:
 - rebbit_graph.csv：儲存圖結構 (節點與節點之間的邊)。
 - train.csv：訓練集資料，內含每個訓練節點的 node_id、features 以及 label。
 - public_test.csv 與 private_test.csv：測試資料，包含節點的 node_id 與 features。

2. 資料預處理、針對訓練集做正規化處理
3. GNN模型訓練(每個epoch紀錄一次loss和代回訓練集的計算accuracy)
5. 儲存訓練好的模型檔案，同時利用訓練完成的模型將public_test.csv、private_test.csv的每個node_id標上標籤，產生一個包含 (node_id, label) 的 DataFrame，最後存入110612008.csv 這個檔案。