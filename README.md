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

## ML訓練用データセット生成

相転移ダイナミクスをMLで学習するための実験データセットを生成する機能。2種類のプロトコルに対応。

### Quench（パラメータ急変）

$\tau$ を急変させて相境界 $\tau = 1/\epsilon$ を横断し、過渡応答を記録する。

- **プロトコル**: $(\tau_0, \epsilon)$ で $t=50\,\mathrm{s}$ 平衡化 → $\tau_1$ に急変 → $t=80\,\mathrm{s}$ 観測
- **遷移タイプ**: Sync→Incoh (40%) / Incoh→Sync (20%) / Sync→Bistable (20%) / Incoh→Bistable (20%)
- 各パラメータ組で `random` / `sync` 2通りの初期条件
- 出力: `output/dataset_quench.npz`（`t`, `R`, `tau_0`, `eps`, `tau_1`, `init_state`, `transition`, `R_before`, `R_after`）

### Periodic Forcing（周期外力）

$\epsilon(t) = \epsilon_0 + A\sin(\Omega t)$ の周期外力に対する応答を記録する。

- **プロトコル**: $\tau$ 固定、$t_\mathrm{max}=200\,\mathrm{s}$
- **ベース点**: Sync / Incoherent / Bistable 各領域からサンプリング
- $A$（振幅）と $\Omega$（周波数）を対数スケールでスイープ
- 出力: `output/dataset_forcing.npz`（`t`, `R`, `tau`, `eps_0`, `A`, `Omega`, `init_state`, `base_phase`, `R_mean`, `R_std`）

### 実行方法

```bash
# Quenchデータセット
python generate_dataset.py quench --n-pairs 10     # 小規模テスト（20サンプル）
python generate_dataset.py quench                   # フル生成（400サンプル）
python generate_dataset.py quench --plot             # 生成後に検証プロット

# Forcingデータセット
python generate_dataset.py forcing --n-base 3       # 小規模テスト
python generate_dataset.py forcing                   # フル生成（480サンプル）

# 両方 + 検証プロット
python generate_dataset.py both
```

検証プロットは `output/validation/` に出力される:
- `quench_trajectories.png` — 遷移タイプ別の $R(t)$ 軌跡
- `quench_deltaR.png` — $\Delta R$ の分布
- `forcing_trajectories.png` — ベース相別の $R(t)$ 軌跡（$\epsilon(t)$ オーバーレイ付き）
- `forcing_response_map.png` — $(A, \Omega)$ 平面上の $R_\mathrm{std}$ ヒートマップ

## 実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python main.py
```

## ファイル構成

```
main.py                  # エントリポイント（相図生成）
generate_dataset.py      # エントリポイント（ML用データセット生成）
src/
  model.py               # Euler法による遅延蔵本モデルのシミュレーション
  model_variable.py      # 時変パラメータ対応シミュレータ + スケジュールファクトリ
  phase_diagram.py       # (tau, eps) グリッドの並列スイープと分類
  dataset.py             # データセット生成ロジック（quench + forcing）
  plot.py                # 相図・時系列プロット
  plot_dataset.py        # データセット検証プロット
output/
  phase_diagram.png
  timeseries.png
  dataset_quench.npz     # Quenchデータセット
  dataset_forcing.npz    # Forcingデータセット
  validation/            # 検証プロット
```

## TODO

- [ ] ML学習可能性の検討: シミュレーション結果 $(\tau, \epsilon) \to \text{phase}$ の写像をNNで学習できるか評価する
- [ ] 生成データセットを用いた相転移ダイナミクスの予測モデル構築
