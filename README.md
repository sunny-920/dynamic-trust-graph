# 動態信任圖模型：假訊息傳播分析

以有向圖模擬社群網路中的假訊息傳播，結合「關係、互動、信任」三種權重計算傳播分數，並提供 PySide6 圖形介面進行互動式分析。

## 功能

- **圖形編輯**：新增／刪除節點與有向邊，可載入預設範例或回聲室範例
- **互動矩陣 I 與信任矩陣 T**：可隨機產生、手動編輯或清空／重置
- **Score 矩陣**：依權重 α（關係）、β（互動）、γ（信任）計算節點間的傳播分數
- **動態信任更新**：通報假訊息時，該節點對外信任值 −0.20；通報優良互動時 +0.05
- **擴散模擬**：從指定來源節點以 BFS 模擬傳播；動態模式下，每次傳播後傳送者的信任值會衰減
- **高風險路徑搜尋**：以 Dijkstra 演算法找出最容易傳播假訊息的路徑
- **圖形分析**：PageRank 影響力、入度／出度、強連通分量（SCC）回聲室偵測、高風險節點排行
- **推播順序**：列出每個節點的來源，依 Score 由高到低排序

## 模型說明

傳播分數：

```
S(i, j) = α·R(i, j) + β·I(i, j) + γ·T(i, j)
```

- `R`：關係矩陣。單向資訊流 R = 0.7，雙向互相關注／互傳 R = 1.0
- `I`：互動矩陣，值介於 0～1
- `T`：信任矩陣，值介於 0～1
- 權重預設 α = 0.3、β = 0.4、γ = 0.3，計算時會正規化使總和為 1
- **信任閘門**：節點的平均對外信任值低於門檻（預設 0.2）時，該節點發出的所有 Score 歸零
- 只有 Score ≥ 傳播門檻（預設 0.4）的邊才會傳播訊息

高風險路徑使用成本 `cost = 1 / (S + ε)`，Score 越高的邊成本越低。

回聲室判定：節點數 ≥ 3 的強連通分量視為潛在回聲室；若內部平均互動 ≥ 0.7 且平均信任 ≥ 0.6，則標記為高風險回聲室。

## 安裝與執行

需要 Python 3.10 以上。

```bash
git clone https://github.com/sunny-920/dynamic-trust-graph.git
cd dynamic-trust-graph

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

## 專案結構

```
dynamic_trust_graph/
├── core/
│   ├── graph_model.py     # 圖模型：節點、邊、矩陣與情境產生
│   ├── matrix_model.py    # 關係矩陣與 Score 計算
│   ├── trust_update.py    # 信任值衰減與獎勵
│   ├── diffusion.py       # BFS 擴散模擬
│   ├── path_analysis.py   # Dijkstra 高風險路徑
│   └── centrality.py      # PageRank、度數、SCC 回聲室
├── gui/
│   ├── main_window.py     # 主視窗與各分頁
│   ├── control_panel.py   # 左側控制面板
│   ├── graph_canvas.py    # 圖形繪製（matplotlib）
│   └── matrix_table.py    # 矩陣表格元件
├── utils/
│   └── sample_data.py     # 預設範例資料
└── main.py
main.py                    # 程式進入點
```

## 文件

- [報告書.pdf](報告書.pdf)
- [使用手冊.pdf](使用手冊.pdf)

## 使用套件

PySide6、matplotlib、NetworkX、NumPy
