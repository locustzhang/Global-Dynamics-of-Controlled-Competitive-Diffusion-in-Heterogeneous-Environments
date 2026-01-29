#three typical intervention strategies
#Four types of spatial heterogeneity
#parameter sensitivity analysis
#complete eradication
#Explicit comparisons between the no-intervention case
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ===================== 核心参数配置 =====================
lambda1, lambda2 = 0.5, 0.05
alpha1, alpha2 = 2.0, 3.0
gamma2 = 0.4
I_inf = 0.3
v0_mean = 0.5
convergence_threshold = 1e-3
dt = 0.01
t_max = 8000
sample_num = 3
nx, ny = 40, 40
dx, dy = 1.0 / nx, 1.0 / ny
EARLY_STOP = True

# 自定义配色
colors = {
    'constant': '#2E86AB',
    'periodic': '#A23B72',
    'pulse': '#F18F01',
    'no_interv': '#C73E1D',
    'homo': '#3498DB',
    'sharp_gradient': '#E74C3C',
    'multiple_patches': '#2ECC71',
    'random_field': '#F39C12'
}

# ===================== 空间异质性P(x)构建 =====================
def create_P_env(p_type):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    if p_type == 'homo':
        P = 0.8 * np.ones((nx, ny))
    elif p_type == 'sharp_gradient':
        P = 0.2 + 0.8 * (X > 0.5).astype(float)
    elif p_type == 'multiple_patches':
        P = 0.5 * np.ones((nx, ny))
        centers = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
        for (cx, cy) in centers:
            mask = (X - cx) ** 2 + (Y - cy) ** 2 < 0.1 ** 2
            P[mask] = 1.0
    elif p_type == 'random_field':
        np.random.seed(42)
        P = np.random.uniform(0.4, 1.0, (nx, ny))
        from scipy.ndimage import gaussian_filter
        P = gaussian_filter(P, sigma=1)
    elif p_type == 'hetero_patch':
        P = 0.6 * np.ones((nx, ny))
        mask = (X - 0.5) ** 2 + (Y - 0.5) ** 2 < 0.2 ** 2
        P[mask] = 1.0
    elif p_type == 'hetero_gradient':
        P = 0.6 + 0.4 * X
    return P, X, Y

# ===================== 干预策略函数 =====================
def get_intervention(t, interv_type):
    if interv_type == 'no_interv':
        return 0.0
    elif interv_type == 'constant':
        return I_inf
    elif interv_type == 'periodic':
        return I_inf if (t % 20) < 1 else 0.0
    elif interv_type == 'pulse':
        return 2 * I_inf if (t % 50 == 0) and (t < 500) else 0.0

# ===================== 有限差分求解器 =====================
def solve_rde(interv_type, p_type, gamma1_val, D1=0.1, D2=0.1, alpha2=3.0, lambda2=0.05):
    P, X, Y = create_P_env(p_type)
    u = np.random.uniform(0.1, 0.3, (nx, ny))
    v = np.random.uniform(v0_mean - 0.1, v0_mean + 0.1, (nx, ny))
    v_initial = np.mean(v)
    u_initial = np.mean(u)
    v_time_series = [v_initial]
    u_time_series = [u_initial]
    time_points = [0]

    def build_laplacian(D):
        n = nx * ny
        diag = -2 * D * (1 / (dx ** 2) + 1 / (dy ** 2)) * np.ones(n)
        diag_x = D / (dx ** 2) * np.ones(n - 1)
        diag_y = D / (dy ** 2) * np.ones(n - nx)
        for i in range(1, nx):
            diag_x[i * nx - 1] = 0.0
        diag_y[-nx:] = 0.0
        L = sp.diags([diag, diag_x, diag_x, diag_y, diag_y],
                     [0, 1, -1, nx, -nx], format='csr')
        return L

    L1 = build_laplacian(D1)
    L2 = build_laplacian(D2)
    n = nx * ny
    u_prev, v_prev = u.copy(), v.copy()
    convergence_flag = False

    for t_step in range(t_max):
        I_t = get_intervention(t_step, interv_type)
        u_flat = u_prev.flatten()
        v_flat = v_prev.flatten()
        P_flat = P.flatten()

        reaction_u = alpha1 * P_flat * u_flat * (1 - u_flat - v_flat) - lambda1 * u_flat + gamma1_val * u_flat * I_t
        reaction_v = alpha2 * P_flat * v_flat - lambda2 * v_flat - gamma2 * v_flat * I_t - alpha2 * P_flat * v_flat * (u_flat + v_flat)
        u_new = spsolve(sp.eye(n) - dt * L1, u_flat + dt * reaction_u).reshape((nx, ny))
        v_new = spsolve(sp.eye(n) - dt * L2, v_flat + dt * reaction_v).reshape((nx, ny))
        u_new, v_new = np.maximum(u_new, 0), np.maximum(v_new, 0)

        if EARLY_STOP and not convergence_flag:
            if t_step % 100 == 0:
                v_time_series.append(np.mean(v_new))
                u_time_series.append(np.mean(u_new))
                time_points.append(t_step * dt)
                if np.max(np.abs(v_new - v_prev)) < convergence_threshold:
                    convergence_flag = True
                    break
        u_prev, v_prev = u_new.copy(), v_new.copy()

    v_final = np.mean(v_prev)
    u_final = np.mean(u_prev)
    suppression_eff = 1 - (v_final / v_initial)
    v_spatial_std = np.std(v_prev)
    v_spatial_cv = v_spatial_std / v_final
    corr_Pv = np.corrcoef(P.flatten(), v_prev.flatten())[0, 1] if np.std(P) > 0 else 0
    convergence_time = time_points[-1] if convergence_flag else t_max * dt
    v_decay_rate = (v_initial - v_final) / convergence_time if convergence_time > 0 else 0
    P_mean = np.mean(P)  # 明确计算P_mean

    return {
        'interv_type': interv_type, 'p_type': p_type, 'gamma1': gamma1_val,
        'threshold_satisfied': gamma1_val <= lambda1, 'suppression_eff': suppression_eff,
        'v_initial': v_initial, 'v_final': v_final, 'u_final': u_final,
        'v_spatial_std': v_spatial_std, 'v_spatial_cv': v_spatial_cv, 'corr_Pv': corr_Pv,
        'P_mean': P_mean, 'v_time_series': v_time_series, 'time_points': time_points,
        'convergence_time': convergence_time, 'v_decay_rate': v_decay_rate,
        'v_final_spatial': v_prev, 'P': P, 'X': X, 'Y': Y
    }

# ===================== 统计显著性分析 =====================
def statistical_analysis(df):
    results = []
    groups = df.groupby('interv_type')['suppression_eff_mean'].apply(list).to_dict()
    for g1 in groups.keys():
        for g2 in groups.keys():
            if g1 < g2:
                stat, p_val = stats.ttest_ind(groups[g1], groups[g2])
                results.append({
                    'comparison': f'{g1} vs {g2}', 'statistic': stat, 'p_value': p_val,
                    'significant': p_val < 0.05, 'sig_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                })
    return pd.DataFrame(results)

# ===================== 批量运行+1行2列图表生成 =====================
def run_reviewer_required_analyses():
    # 1. 阈值条件验证
    gamma1_list = [0.4, 0.5, 0.6, 0.7]
    interv_types = ['constant', 'periodic', 'pulse', 'no_interv']
    threshold_results = []
    threshold_full_results = []
    for gamma1 in gamma1_list:
        for interv in interv_types:
            sample_effs = []
            sample_full = []
            for _ in range(sample_num):
                res = solve_rde(interv, 'hetero_gradient', gamma1)
                sample_effs.append(res['suppression_eff'])
                sample_full.append(res)
            threshold_results.append({
                'gamma1': gamma1, 'interv_type': interv,
                'suppression_eff_mean': np.mean(sample_effs),
                'suppression_eff_std': np.std(sample_effs),
                'suppression_eff_se': stats.sem(sample_effs),
                'threshold_satisfied': gamma1 <= lambda1
            })
            threshold_full_results.append(sample_full[0])
    df_threshold = pd.DataFrame(threshold_results)
    df_threshold_full = pd.DataFrame(threshold_full_results)

    # 2. 空间异质性分析（修复：添加P_mean字段）
    p_types = ['homo', 'sharp_gradient', 'multiple_patches', 'random_field']
    spatial_results = []
    spatial_full_results = []
    for p in p_types:
        sample_res = []
        for _ in range(sample_num):
            res = solve_rde('constant', p, 0.4)
            sample_res.append(res)
        spatial_full_results.append(sample_res[0])
        spatial_results.append({
            'p_type': p,
            'suppression_eff_mean': np.mean([r['suppression_eff'] for r in sample_res]),
            'suppression_eff_std': np.std([r['suppression_eff'] for r in sample_res]),
            'v_final_mean': np.mean([r['v_final'] for r in sample_res]),
            'corr_Pv_mean': np.mean([r['corr_Pv'] for r in sample_res]),
            'convergence_time_mean': np.mean([r['convergence_time'] for r in sample_res]),
            'P_mean': np.mean([r['P_mean'] for r in sample_res])  # 关键修复：添加P_mean字段
        })
    df_spatial = pd.DataFrame(spatial_results)
    df_spatial_full = pd.DataFrame(spatial_full_results)

    # 3. 参数敏感性分析
    param_grid = {'D1': [0.05, 0.1, 0.3], 'D2': [0.05, 0.1, 0.3], 'alpha2': [2.0, 3.0, 4.0], 'lambda2': [0.03, 0.05, 0.07]}
    sensitivity_results = []
    for param, values in param_grid.items():
        for val in values:
            kwargs = {param: val}
            sample_effs = []
            for _ in range(sample_num):
                res = solve_rde('constant', 'hetero_gradient', 0.4,** kwargs)
                sample_effs.append(res['suppression_eff'])
            sensitivity_results.append({
                'param': param, 'value': val,
                'suppression_eff_mean': np.mean(sample_effs),
                'suppression_eff_std': np.std(sample_effs)
            })
    df_sensitivity = pd.DataFrame(sensitivity_results)

    # 4. 根除验证
    eradication_results = []
    gamma1_range = np.linspace(0.3, 0.6, 7)
    gamma2_range = np.linspace(0.2, 0.6, 5)
    for g1 in gamma1_range:
        for g2 in gamma2_range:
            v_final_list = []
            for _ in range(2):
                res = solve_rde('constant', 'hetero_gradient', g1)
                v_final_list.append(res['v_final'])
            eradication_results.append({
                'gamma1': g1, 'gamma2': g2,
                'v_final_mean': np.mean(v_final_list),
                'complete_eradication': np.mean(v_final_list) < 1e-4
            })
    df_eradication = pd.DataFrame(eradication_results)

    # 5. 统计分析
    df_stats = statistical_analysis(df_threshold)

    # ===================== 图表1：阈值效应 + 动态演化（1行2列） =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Threshold Effect and Dynamic Evolution of v Density', fontsize=16, fontweight='bold')

    # 左子图：阈值效应
    for interv in interv_types:
        data = df_threshold[df_threshold['interv_type'] == interv]
        ax1.errorbar(data['gamma1'], data['suppression_eff_mean'], yerr=data['suppression_eff_se'],
                     label=interv, color=colors[interv], marker='o', linewidth=2, capsize=5)
    ax1.axvline(x=lambda1, color='red', linestyle='--', linewidth=2, label='Threshold γ1=λ1=0.5')
    ax1.set_xlabel('γ1 (Intervention Coefficient)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Suppression Efficiency', fontsize=12, fontweight='bold')
    ax1.set_title('Threshold Effect Across Intervention Strategies', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=-0.1)

    # 右子图：动态演化
    data_04 = df_threshold_full[(df_threshold_full['gamma1'] == 0.4) & (df_threshold_full['interv_type'] == 'constant')].iloc[0]
    data_06 = df_threshold_full[(df_threshold_full['gamma1'] == 0.6) & (df_threshold_full['interv_type'] == 'constant')].iloc[0]
    ax2.plot(data_04['time_points'], data_04['v_time_series'], color=colors['constant'], label='γ1=0.4 (≤λ1)', linewidth=2.5)
    ax2.plot(data_06['time_points'], data_06['v_time_series'], color=colors['no_interv'], label='γ1=0.6 (>λ1)', linewidth=2.5, linestyle='--')
    ax2.axhline(y=v0_mean, color='black', linestyle=':', alpha=0.7, label='Initial v Density')
    ax2.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean v Density', fontsize=12, fontweight='bold')
    ax2.set_title('Dynamic Evolution (Constant Intervention)', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig1_threshold_dynamics.png', bbox_inches='tight', dpi=300)
    plt.close()

    # ===================== 图表2：空间异质性 + P-v相关性（1行2列） =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Impact of Spatial Heterogeneity (P(x))', fontsize=16, fontweight='bold')

    # 左子图：抑制效率对比
    x = np.arange(len(df_spatial))
    bars = ax1.bar(x, df_spatial['suppression_eff_mean'], yerr=df_spatial['suppression_eff_std'],
                   color=[colors[p] for p in df_spatial['p_type']], capsize=8, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_spatial['p_type'], rotation=15, ha='right')
    ax1.set_xlabel('Spatial Heterogeneity Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Suppression Efficiency (mean ± std)', fontsize=12, fontweight='bold')
    ax1.set_title('Suppression Efficiency Across P(x) Types', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(df_spatial['suppression_eff_mean']):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # 右子图：P-v相关性散点图（现在P_mean字段存在，不会报错）
    ax2.scatter(df_spatial['P_mean'], df_spatial['corr_Pv_mean'], s=150,
                c=[colors[p] for p in df_spatial['p_type']], alpha=0.8, edgecolors='black')
    ax2.set_xlabel('Mean P(x) Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P-v Correlation Coefficient', fontsize=12, fontweight='bold')
    ax2.set_title('Correlation Between P(x) and v Distribution', fontsize=14)
    ax2.grid(alpha=0.3)
    for i, txt in enumerate(df_spatial['p_type']):
        ax2.annotate(txt, (df_spatial['P_mean'].iloc[i], df_spatial['corr_Pv_mean'].iloc[i]), fontsize=10, ha='right')
    plt.tight_layout()
    plt.savefig('fig2_spatial_heterogeneity.png', bbox_inches='tight', dpi=300)
    plt.close()

    # ===================== 图表3：参数敏感性 + 根除相图（1行2列） =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Parameter Sensitivity and Eradication Phase Diagram', fontsize=16, fontweight='bold')

    # 左子图：参数敏感性
    param_order = ['D1', 'D2', 'alpha2', 'lambda2']
    x_pos = np.arange(3)
    width = 0.2
    for i, param in enumerate(param_order):
        data = df_sensitivity[df_sensitivity['param'] == param]
        ax1.bar(x_pos + (i-1.5)*width, data['suppression_eff_mean'], width,
                yerr=data['suppression_eff_std'], label=param, capsize=3, alpha=0.8)
    ax1.set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Suppression Efficiency', fontsize=12, fontweight='bold')
    ax1.set_title('Parameter Sensitivity Analysis', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Low', 'Medium', 'High'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 右子图：根除相图
    pivot = df_eradication.pivot(index='gamma2', columns='gamma1', values='v_final_mean')
    cmap = LinearSegmentedColormap.from_list('eradication', ['darkblue', 'lightblue', 'yellow', 'orange', 'red'])
    im = ax2.imshow(pivot, cmap=cmap, aspect='auto', vmin=0, vmax=0.5)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if val < 1e-4:
                ax2.text(j, i, '✓', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            elif val < 0.1:
                ax2.text(j, i, '≈', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax2.axvline(x=np.where(pivot.columns == 0.5)[0][0], color='white', linestyle='--', linewidth=2)
    ax2.set_xlabel('γ1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('γ2', fontsize=12, fontweight='bold')
    ax2.set_title('Eradication Phase Diagram (v Final Density)', fontsize=14)
    ax2.set_xticks(np.arange(len(pivot.columns)))
    ax2.set_xticklabels([f'{g:.2f}' for g in pivot.columns])
    ax2.set_yticks(np.arange(len(pivot.index)))
    ax2.set_yticklabels([f'{g:.2f}' for g in pivot.index])
    plt.colorbar(im, ax=ax2, label='v Final Mean Density')
    plt.tight_layout()
    plt.savefig('fig3_parameter_eradication.png', bbox_inches='tight', dpi=300)
    plt.close()

    # ===================== 图表4：干预策略对比 + 统计显著性（1行2列） =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Intervention Strategy Comparison and Statistical Significance', fontsize=16, fontweight='bold')

    # 左子图：策略效率对比
    data_strategy = df_threshold[df_threshold['gamma1'] == 0.4]
    x = np.arange(len(data_strategy))
    bars = ax1.bar(x, data_strategy['suppression_eff_mean'], yerr=data_strategy['suppression_eff_std'],
                   color=[colors[interv] for interv in data_strategy['interv_type']], capsize=8, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(data_strategy['interv_type'])
    ax1.set_xlabel('Intervention Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Suppression Efficiency (γ1=0.4 ≤λ1)', fontsize=12, fontweight='bold')
    ax1.set_title('Strategy Comparison at Sub-threshold γ1', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(data_strategy['suppression_eff_mean']):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # 右子图：统计显著性热图（修复：处理空值）
    sig_data = df_stats.pivot_table(index=df_stats['comparison'].str.split(' vs ').str[0],
                                    columns=df_stats['comparison'].str.split(' vs ').str[1],
                                    values='p_value')
    # 填充空值为1（无显著性）
    sig_data = sig_data.fillna(1)
    # 转换为显著性矩阵（p<0.05为1，否则为0）
    sig_matrix = (sig_data < 0.05).astype(float)
    im = ax2.imshow(sig_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xlabel('Strategy 2', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Strategy 1', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Significance (p < 0.05 = Green)', fontsize=14)
    ax2.set_xticks(np.arange(len(sig_data.columns)))
    ax2.set_xticklabels(sig_data.columns)
    ax2.set_yticks(np.arange(len(sig_data.index)))
    ax2.set_yticklabels(sig_data.index)
    plt.colorbar(im, ax=ax2, label='Significance (1=Significant, 0=Not)')
    plt.tight_layout()
    plt.savefig('fig4_strategy_statistics.png', bbox_inches='tight', dpi=300)
    plt.close()

    # ===================== 输出表格和结论 =====================
    df_threshold.pivot(index='gamma1', columns='interv_type', values=['suppression_eff_mean', 'suppression_eff_std']).round(4).to_csv('table_threshold.csv')
    df_spatial[['p_type', 'suppression_eff_mean', 'corr_Pv_mean', 'P_mean', 'convergence_time_mean']].round(4).to_csv('table_spatial.csv', index=False)
    df_sensitivity.round(4).to_csv('table_sensitivity.csv', index=False)

    print("\n✅ All 1x2 layout figures generated!")
    print("✅ Generated figures: fig1_threshold_dynamics.png, fig2_spatial_heterogeneity.png, fig3_parameter_eradication.png, fig4_strategy_statistics.png")
    print("✅ Generated CSV tables for paper.")

    return df_threshold, df_spatial, df_sensitivity, df_eradication, df_stats

# ===================== 运行主函数 =====================
if __name__ == '__main__':
    import time
    start_time = time.time()
    df_threshold, df_spatial, df_sensitivity, df_eradication, df_stats = run_reviewer_required_analyses()
    print(f"\nTotal runtime: {(time.time() - start_time)/60:.1f} minutes")