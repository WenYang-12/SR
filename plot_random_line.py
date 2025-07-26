import matplotlib.pyplot as plt
import numpy as np

def plot_bar_grouped_per_dataset(micro_data, macro_data, save_prefix=None):
    """
    micro_data, macro_data: {dataset: {model: (mean, std)}}
    每个数据集一张图，横坐标为Micro和Macro两类，每类下有四个bar（Delaunay, random_same, random_double, random_half）。
    """
    color_palette = ['#ff165d', '#3ec1d3', '#fbb13c', '#9b5de5']
    strategies = ['Delaunay', 'random_same', 'random_double', 'random_half']
    strategy_labels = ['Delaunay', 'Random Same', 'Random Double', 'Random Half']
    datasets = list(micro_data.keys())
    n_group = 2  # Micro, Macro
    n_bar = len(strategies)
    bar_width = 0.18
    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        # Micro/Macro means/stds
        micro = micro_data.get(dataset, {})
        macro = macro_data.get(dataset, {})
        micro_means = [micro.get(s, (np.nan, 0))[0] for s in strategies]
        micro_stds = [micro.get(s, (0, 0))[1] for s in strategies]
        macro_means = [macro.get(s, (np.nan, 0))[0] for s in strategies]
        macro_stds = [macro.get(s, (0, 0))[1] for s in strategies]
        # x轴分组
        x = np.arange(n_group)
        for i, (means, stds) in enumerate(zip([micro_means, macro_means], [micro_stds, macro_stds])):
            for j, (mean, std) in enumerate(zip(means, stds)):
                plt.bar(i + (j - 1.5) * bar_width, mean, bar_width, yerr=std, color=color_palette[j], label=strategy_labels[j] if i == 0 else None, capsize=5)
                if not np.isnan(mean):
                    plt.text(i + (j - 1.5) * bar_width, mean + std + 1, f'{mean:.2f}', ha='center', va='bottom', fontsize=11, color=color_palette[j])
        plt.xticks(x, ['Micro-F1', 'Macro-F1'], fontsize=15)
        plt.yticks(fontsize=13)
        plt.ylabel('Score (%)', fontsize=15)
        # plt.title(f'{dataset} Random Rewire Strategies', fontsize=17, pad=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.ylim(0, max([v for v in micro_means + macro_means if not np.isnan(v)]) * 1.15)
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),  # 图的中间正上方
            fontsize=10,
            frameon=False,
            ncol=4
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        if save_prefix:
            plt.savefig(f'{save_prefix}_{dataset}.png', dpi=180, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    # Micro-F1数据
    micro_data = {
        'Cora': {'Delaunay': (93.15, 0.49), 'random_same': (73.95, 8.68), 'random_double': (78.80, 6.95), 'random_half': (72.14, 12.27)},
        'CiteSeer': {'Delaunay': (89.17, 0.63), 'random_same': (60.27, 11.01), 'random_double': (62.51, 11.88), 'random_half': (61.35, 4.73)},
        'PubMed': {'Delaunay': (92.71, 1.46), 'random_same': (89.94, 1.15), 'random_double': (90.00, 1.28), 'random_half': (90.32, 0.70)},
        'Cornell': {'Delaunay': (89.95, 2.96), 'random_same': (66.56, 24.83), 'random_double': (76.39, 7.88), 'random_half': (72.79, 8.18)},
        'Texas': {'Delaunay': (95.19, 1.64), 'random_same': (83.39, 5.10), 'random_double': (82.30, 6.00), 'random_half': (83.93, 2.78)},
        'Wisconsin': {'Delaunay': (94.98, 1.46), 'random_same': (55.94, 32.50), 'random_double': (65.58, 19.52), 'random_half': (45.58, 30.42)},
        'Film': {'Delaunay': (63.56, 0.36), 'random_same': (48.00, 9.18), 'random_double': (53.88, 4.47), 'random_half': (49.67, 6.86)}
    }
    # Macro-F1数据
    macro_data = {
        'Cora': {'Delaunay': (92.44, 0.64), 'random_same': (55.55, 12.51), 'random_double': (61.54, 9.98), 'random_half': (55.30, 17.88)},
        'CiteSeer': {'Delaunay': (88.19, 0.42), 'random_same': (50.00, 11.54), 'random_double': (51.42, 12.21), 'random_half': (49.40, 5.84)},
        'PubMed': {'Delaunay': (92.62, 1.34), 'random_same': (89.38, 1.56), 'random_double': (89.45, 1.67), 'random_half': (89.89, 0.72)},
        'Cornell': {'Delaunay': (85.73, 4.42), 'random_same': (51.44, 29.03), 'random_double': (51.79, 13.70), 'random_half': (49.14, 15.90)},
        'Texas': {'Delaunay': (91.07, 9.03), 'random_same': (54.45, 8.37), 'random_double': (51.33, 11.28), 'random_half': (56.31, 4.63)},
        'Wisconsin': {'Delaunay': (89.49, 5.49), 'random_same': (33.55, 19.62), 'random_double': (39.12, 10.52), 'random_half': (29.17, 20.89)},
        'Film': {'Delaunay': (61.81, 0.44), 'random_same': (40.31, 9.97), 'random_double': (46.55, 3.64), 'random_half': (42.40, 7.13)}
    }
    plot_bar_grouped_per_dataset(micro_data, macro_data) 