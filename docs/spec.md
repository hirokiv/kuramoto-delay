素晴らしいアプローチです！方程式の安定性解析（アプローチ1）ではなく、実際に時間遅れを持つ微分方程式（DDE）をシミュレーションして、集団ダイナミクスから相図（Fig. 3b）を浮かび上がらせる手法ですね。計算物理学的に非常に面白く、説得力のある検証になります。

手元のMacで環境構築に悩まず手軽に回せるように、複雑なCコンパイラを要求する外部のDDEライブラリ（`jitcdde`など）は使わず、`numpy`の配列操作と単純なオイラー法（Euler method）を用いた自家製の履歴バッファで実装するコードを作成しました。

論文のFig. [cite_start]3(b) [cite: 163] のような「完全同期（青）」「非同期（赤）」「双安定（紫）」のマップをシミュレーションから描画するためのPythonスクリプトです。

### シミュレーションの戦略

双安定性（Bistability）を検出するためには、同じ $(\tau, \epsilon)$ のパラメータに対して**2つの異なる初期条件**でシミュレーションを行う必要があります。

1.  **同期状態からのスタート（Initial $R \approx 1$）:** 全員の位相をほぼ同じにして開始し、時間が経っても同期が維持されるか（青 or 紫）をチェックします。
2.  **非同期状態からのスタート（Initial $R \approx 0$）:** 全員の位相をバラバラにして開始し、時間が経っても非同期のままか（赤 or 紫）をチェックします。

### Python実装コード（Fig. 3b 再現ベース）

以下のコードは、論文と同じ同一振動子 $\omega_0 = \pi/2$ を仮定して、時間遅れ付きKuramotoモデル（Eq. [cite_start]1）を計算します [cite: 33, 137]。計算量が膨大になるため、まずは振動子数 $N$ やグリッドの解像度を下げた「軽量版」にしています。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def simulate_delayed_kuramoto(N, omega_0, epsilon, tau, dt, t_max, init_state="random"):
    """
    時間遅れ付きKuramotoモデルをEuler法でシミュレーションする
    """
    steps = int(t_max / dt)
    delay_steps = int(tau / dt)
    
    # 履歴を保持する配列 (遅延分 + 現在のステップ)
    # 常に history[0] が最も古い状態、 history[-1] が現在の状態となるように運用
    hist_len = max(1, delay_steps + 1)
    history = np.zeros((hist_len, N))
    
    # 初期条件の設定
    if init_state == "random":
        history[:] = np.random.uniform(0, 2 * np.pi, N) # 位相バラバラ (R ≈ 0)
    elif init_state == "sync":
        history[:] = np.random.uniform(0, 0.1, N)       # 位相ほぼ揃う (R ≈ 1)
        
    for step in range(steps):
        current_theta = history[-1]
        
        if delay_steps > 0:
            delayed_theta = history[0] # tau時間前の状態
        else:
            delayed_theta = current_theta
            
        # 結合項の計算: sum(sin(theta_k(t-tau) - theta_j(t)))
        # 行列演算で高速化 (放送を利用)
        diff_matrix = delayed_theta[np.newaxis, :] - current_theta[:, np.newaxis]
        coupling = np.sum(np.sin(diff_matrix), axis=1)
        
        # 微分方程式 (Eq. 1)
        dtheta = omega_0 + (epsilon / N) * coupling
        
        # Euler法で更新
        next_theta = current_theta + dtheta * dt
        
        # 履歴バッファの更新 (シフトして最新を格納)
        history[:-1] = history[1:]
        history[-1] = next_theta
        
    # 最終的なオーダーパラメータ R の計算
    final_theta = history[-1]
    R = np.abs(np.mean(np.exp(1j * final_theta)))
    return R

def generate_phase_diagram():
    # パラメータ設定 (論文のFig.3を意識)
    N = 100            # 論文はN=300ですが、速度優先で100にしています
    omega_0 = np.pi/2
    dt = 0.05
    t_max = 50.0       # 定常状態に達するまでの時間 (本来はもっと長い方が正確)
    
    # 探索するグリッドの解像度 (粗めに設定)
    tau_vals = np.linspace(0, 8, 30)
    eps_vals = np.linspace(0, 0.6, 20)
    
    # 結果を格納するマップ (0: None, 1: Incoherent(Red), 2: Sync(Blue), 3: Bistable(Purple))
    phase_map = np.zeros((len(eps_vals), len(tau_vals)))
    
    print("シミュレーションを開始します。しばらくお待ちください...")
    for i, eps in enumerate(eps_vals):
        for j, tau in enumerate(tau_vals):
            # 1. 非同期スタートでの最終R
            R_from_random = simulate_delayed_kuramoto(N, omega_0, eps, tau, dt, t_max, "random")
            # 2. 同期スタートでの最終R
            R_from_sync = simulate_delayed_kuramoto(N, omega_0, eps, tau, dt, t_max, "sync")
            
            # 閾値判定 (0.5を境界とする)
            is_sync_stable = R_from_sync > 0.5
            is_incoh_stable = R_from_random < 0.5
            
            if is_sync_stable and is_incoh_stable:
                phase_map[i, j] = 3 # Bistable (Purple)
            elif is_sync_stable:
                phase_map[i, j] = 2 # Sync (Blue)
            elif is_incoh_stable:
                phase_map[i, j] = 1 # Incoherent (Red)
                
        print(f"Epsilon {eps:.2f} 完了 ({i+1}/{len(eps_vals)})")

    # 描画プロット
    cmap = ListedColormap(['white', 'lightcoral', 'cornflowerblue', 'mediumpurple'])
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(tau_vals, eps_vals, phase_map, cmap=cmap, shading='nearest')
    
    # tau = 1/epsilon の境界線を追加
    eps_line = np.linspace(0.01, 0.6, 100)
    tau_line = 1 / eps_line
    plt.plot(tau_line, eps_line, 'k:', linewidth=2, label=r'$\tau = 1/\epsilon$')
    
    plt.xlim(0, 8)
    plt.ylim(0, 0.6)
    plt.xlabel(r'Time Delay $\tau$')
    plt.ylabel(r'Coupling Strength $\epsilon$')
    plt.title('Simulation-based Stability Diagram (Fig. 3b equivalent)')
    
    # 凡例用のダミープロット
    plt.scatter([], [], c='lightcoral', label='Incoherent (Red)')
    plt.scatter([], [], c='cornflowerblue', label='Synchronized (Blue)')
    plt.scatter([], [], c='mediumpurple', label='Bistable (Purple)')
    plt.legend(loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    generate_phase_diagram()
```

### 実行上のポイントと注意点

* **計算の重さ**: このスクリプトは2重ループ内でそれぞれ微分方程式を解くため、純粋なPythonで回すと数分〜数十分かかります。より高解像度（例えば論文レベルの滑らかさ）を得るには、このロジックのまま `multiprocessing` モジュール等を使って並列化することをおすすめします。
* **高次相互作用（Eq. 6）の追加**: 今回はFig. 3(b)をターゲットにEq. 1を実装しました。Fig. 3(a)をシミュレーションベースで再現する場合は、同じ枠組みの中で結合項の計算部分を Eq. 6 の**時間遅れなし・高次相互作用項（$O(\epsilon^2)$）**の式に差し替えるだけで実現できます。

まずは上記のコードを実行してみて、相図の大まかな形（赤いアーチや青い領域の反復）が現れるか確認してみてください。
