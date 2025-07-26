import matplotlib.pyplot as plt
import numpy as np

def plot_bar_results(results, metric='test_acc', title='Test Accuracy Comparison', ylabel='Accuracy', save_path=None):
    # ... existing code ...
    pass  # 省略原有内容

def plot_train_val_curves(results, model_name, save_path=None):
    # ... existing code ...
    pass  # 省略原有内容

def plot_micro_macro_bar(data_dict, save_prefix=None):
    color_palette = ['#ff165d', '#3ec1d3']
    datasets = list(data_dict.keys())
    # 1. Micro-F1
    plt.figure(figsize=(10, 6))
    models = ['Dual Fusion', 'Concatenation']
    n_models = len(models)
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.7 / n_models
    for i, model in enumerate(models):
        means = []
        stds = []
        for dataset in datasets:
            val = data_dict[dataset].get('Micro-F1', {}).get(model, (np.nan, 0))
            means.append(val[0])
            stds.append(val[1])
        plt.bar(x + (i - n_models/2 + 0.5)*width, means, width, yerr=stds, label=model, color=color_palette[i], capsize=5, alpha=0.9)
        for xi, mean, std in zip(x, means, stds):
            if not np.isnan(mean):
                plt.text(xi + (i - n_models/2 + 0.5)*width, mean + std + 1, f'{mean:.2f}', ha='center', va='bottom', fontsize=11, color=color_palette[i])
    plt.xticks(x, datasets, fontsize=14, rotation=0)
    plt.yticks(fontsize=13)
    plt.ylabel('Score (%)', fontsize=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.25)
    plt.ylim(0, max([data_dict[d].get('Micro-F1', {}).get(m, (0,0))[0] for d in datasets for m in models if not np.isnan(data_dict[d].get('Micro-F1', {}).get(m, (np.nan,0))[0])] + [0]) * 1.15)
    plt.gca().set_facecolor('white')
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.18),
        fontsize=14,
        frameon=False,
        ncol=n_models
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save_prefix:
        plt.savefig(f'{save_prefix}_micro.png', dpi=180, bbox_inches='tight')
    plt.show()
    # 2. Macro-F1
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        means = []
        stds = []
        for dataset in datasets:
            val = data_dict[dataset].get('Macro-F1', {}).get(model, (np.nan, 0))
            means.append(val[0])
            stds.append(val[1])
        plt.bar(x + (i - n_models/2 + 0.5)*width, means, width, yerr=stds, label=model, color=color_palette[i], capsize=5, alpha=0.9)
        for xi, mean, std in zip(x, means, stds):
            if not np.isnan(mean):
                plt.text(xi + (i - n_models/2 + 0.5)*width, mean + std + 1, f'{mean:.2f}', ha='center', va='bottom', fontsize=11, color=color_palette[i])
    plt.xticks(x, datasets, fontsize=14, rotation=0)
    plt.yticks(fontsize=13)
    plt.ylabel('Score (%)', fontsize=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.25)
    plt.ylim(0, max([data_dict[d].get('Macro-F1', {}).get(m, (0,0))[0] for d in datasets for m in models if not np.isnan(data_dict[d].get('Macro-F1', {}).get(m, (np.nan,0))[0])] + [0]) * 1.15)
    plt.gca().set_facecolor('white')
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.18),
        fontsize=14,
        frameon=False,
        ncol=n_models
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save_prefix:
        plt.savefig(f'{save_prefix}_macro.png', dpi=180, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 示例数据
    data_dict = {
        'Cora': {
            'Micro-F1': {
                'Dual Fusion': (93.15, 0.49),
                'Concatenation': (71.29, 17.02)
            },
            'Macro-F1': {
                'Dual Fusion': (92.44, 0.64),
                'Concatenation': (57.87, 21.34)
            }
        },
        'CiteSeer': {
            'Micro-F1': {
                'Dual Fusion': (89.17, 0.63),
                'Concatenation': (62.56, 6.49)
            },
            'Macro-F1': {
                'Dual Fusion': (88.19, 0.42),
                'Concatenation': (52.82, 7.00)
            }
        },
        'PubMed': {
            'Micro-F1': {
                'Dual Fusion': (92.71, 1.46),
                'Concatenation': (90.62, 0.40)
            },
            'Macro-F1': {
                'Dual Fusion': (92.62, 1.34),
                'Concatenation': (90.19, 0.43)
            }
        },
        'Cornell': {
            'Micro-F1': {
                'Dual Fusion': (89.95, 2.96),
                'Concatenation': (71.15, 9.94)
            },
            'Macro-F1': {
                'Dual Fusion': (85.73, 4.42),
                'Concatenation': (47.69, 19.59)
            }
        },
        'Texas': {
            'Micro-F1': {
                'Dual Fusion': (95.19, 1.64),
                'Concatenation': (85.46, 6.87)
            },
            'Macro-F1': {
                'Dual Fusion': (91.07, 9.03),
                'Concatenation': (58.43, 11.65)
            }
        },
        'Wisconsin': {
            'Micro-F1': {
                'Dual Fusion': (94.98, 1.46),
                'Concatenation': (63.11, 19.40)
            },
            'Macro-F1': {
                'Dual Fusion': (89.49, 5.49),
                'Concatenation': (35.68, 13.31)
            }
        },
        'Film': {
            'Micro-F1': {
                'Dual Fusion': (63.56, 0.36),
                'Concatenation': (48.94, 7.34)
            },
            'Macro-F1': {
                'Dual Fusion': (61.81, 0.44),
                'Concatenation': (41.06, 8.79)
            }
        }
    }
    plot_micro_macro_bar(data_dict) 