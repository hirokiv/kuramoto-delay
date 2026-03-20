# Delayed Kuramoto Model Simulation

時間遅延付き蔵本モデルのシミュレーションと安定性解析。

## モデル

全結合・同一振動数の蔵本モデルに時間遅延 $\tau$ を導入:

$$\dot{\theta}_j(t) = \omega_0 + \frac{\epsilon}{N} \sum_{k=1}^{N} \sin\bigl(\theta_k(t-\tau) - \theta_j(t)\bigr)$$

- $N$: 振動子数 (=100)
- $\omega_0 = \pi/2$: 固有振動数
- $\epsilon$: 結合強度
- $\tau$: 時間遅延

秩序パラメータ $R(t) = \left|\frac{1}{N}\sum_j e^{i\theta_j(t)}\right|$ で同期度を評価。

## 出力

### Phase Diagram (`output/phase_diagram.png`)

$(\tau, \epsilon)$ 平面上の安定性分類。各点で2種類の初期条件（ランダム / 同期）からシミュレーションし、最終的な $R$ の値を $R_\mathrm{threshold}=0.5$ で判定:

| 分類 | 条件 |
|------|------|
| Synchronized (青) | 両方の初期条件で $R > 0.5$ |
| Incoherent (赤) | 両方の初期条件で $R < 0.5$ |
| Bistable (紫) | 同期スタート → $R > 0.5$、ランダムスタート → $R < 0.5$ |

理論的境界線 $\tau = 1/\epsilon$ を重ねて表示。

### Time Series (`output/timeseries.png`)

代表的な3点における $R(t)$ の時間発展:

| モード | パラメータ | 振る舞い |
|--------|-----------|---------|
| Synchronized | $\tau=0.5,\ \epsilon=0.3$ | 両初期条件とも $R \to 1$ |
| Incoherent | $\tau=1.8,\ \epsilon=0.3$ | 両初期条件とも $R \to 0$ |
| Bistable | $\tau=2.5,\ \epsilon=0.3$ | 初期条件依存（同期→1、ランダム→0） |

## 実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python main.py
```

## ファイル構成

```
main.py                  # エントリポイント
src/
  model.py               # Euler法による遅延蔵本モデルのシミュレーション
  phase_diagram.py       # (tau, eps) グリッドの並列スイープと分類
  plot.py                # 相図・時系列プロット
output/
  phase_diagram.png
  timeseries.png
```

## TODO

- [ ] ML学習可能性の検討: シミュレーション結果 $(\tau, \epsilon) \to \text{phase}$ の写像をNNで学習できるか評価する
