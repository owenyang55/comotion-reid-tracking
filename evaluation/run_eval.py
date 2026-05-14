import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Numpy 兼容性修复
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# 2. 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
trackeval_root = os.path.join(current_dir, "TrackEval")
if trackeval_root not in sys.path:
    sys.path.insert(0, trackeval_root)

try:
    import trackeval.datasets.mot_challenge_2d_box as mod_dataset
    import trackeval.metrics as mod_metrics
    from trackeval.eval import Evaluator
    MotChallenge2DBox = mod_dataset.MotChallenge2DBox
    print("✅ TrackEval 模块加载成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def run_evaluation_and_plot():
    # === TrackEval 配置 ===
    eval_config = Evaluator.get_default_eval_config()
    dataset_config = MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}

    eval_config['PRINT_ONLY_COMBINED'] = True
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['OUTPUT_SUMMARY'] = True
    eval_config['PRINT_RESULTS'] = False 

    dataset_config['GT_FOLDER'] = os.path.join(current_dir, 'data/gt/mot_challenge/mot')
    dataset_config['TRACKERS_FOLDER'] = os.path.join(current_dir, 'data/trackers/mot_challenge/mot')
    dataset_config['OUTPUT_FOLDER'] = os.path.join(current_dir, 'data/results/')
    dataset_config['TRACKERS_TO_EVAL'] = ['original', 'improved'] 
    dataset_config['CLASSES_TO_EVAL'] = ['pedestrian']
    dataset_config['BENCHMARK'] = 'mot'
    dataset_config['SPLIT_TO_EVAL'] = 'test'
    dataset_config['INPUT_AS_ZIP'] = False
    dataset_config['DO_PREPROC'] = False
    dataset_config['TRACKER_SUB_FOLDER'] = 'data'
    dataset_config['OUTPUT_SUB_FOLDER'] = ''
    dataset_config['SEQMAP_FOLDER'] = os.path.join(current_dir, 'data/gt/mot_challenge/seqmaps')
    dataset_config['SEQMAP_FILE'] = os.path.join(current_dir, 'data/gt/mot_challenge/seqmaps/MOT-Test.txt') 
    dataset_config['SKIP_SPLIT_FOL'] = True

    # === 运行评测 ===
    evaluator = Evaluator(eval_config)
    dataset_list = [MotChallenge2DBox(dataset_config)]
    
    metrics_list = []
    for metric_name in metrics_config['METRICS']:
        if hasattr(mod_metrics, metric_name):
            metrics_list.append(getattr(mod_metrics, metric_name)())

    raw_results, _ = evaluator.evaluate(dataset_list, metrics_list)
    dataset_name = dataset_list[0].get_name()
    
    # === 数据提取 ===
    key_metrics = {
        'HOTA': ('HOTA', 'HOTA'),
        'DetA': ('HOTA', 'DetA'),
        'AssA': ('HOTA', 'AssA'),
        'IDF1': ('Identity', 'IDF1'),
        'IDSW': ('CLEAR', 'IDSW'),
        'Frag': ('CLEAR', 'Frag'),
        'Count': ('Count', 'IDs') 
    }

    print("\n" + "="*80)
    print(f"{'Metric':<10} | {'Original':<12} | {'Improved':<12} | {'Change':<20}")
    print("-" * 80)

    res_dict = {k: [] for k in key_metrics.keys()}
    
    # 提取 COMBINED 结果
    res_comb_orig = raw_results[dataset_name]['original']['COMBINED_SEQ']['pedestrian']
    res_comb_impr = raw_results[dataset_name]['improved']['COMBINED_SEQ']['pedestrian']
    
    for metric_display, (metric_class, metric_key) in key_metrics.items():
        val_orig = res_comb_orig[metric_class][metric_key]
        val_impr = res_comb_impr[metric_class][metric_key]
        
        # 稳健的数值转换函数 (处理 numpy array 或 scalar)
        def safe_convert(v):
            if hasattr(v, 'item') and v.size == 1: return v.item()
            elif hasattr(v, 'mean'): return v.mean()
            return v

        val_orig = float(safe_convert(val_orig))
        val_impr = float(safe_convert(val_impr))
        
        res_dict[metric_display] = [val_orig, val_impr]
        
        # 计算变化
        diff = val_impr - val_orig
        pct = (diff / val_orig * 100) if val_orig != 0 else 0
        
        # 打印控制台表格
        better = False
        if metric_display in ['IDSW', 'Frag', 'Count']:
            better = val_impr < val_orig
            change_str = f"{diff:+.0f} ({pct:+.1f}%)"
        else:
            better = val_impr > val_orig
            change_str = f"{diff:+.3f} ({pct:+.1f}%)"
            
        mark = "✅" if better else "  "
        print(f"{metric_display:<10} | {val_orig:<12.3f} | {val_impr:<12.3f} | {change_str:<20} {mark}")

    print("="*80)

    # === 调用优化后的绘图函数 ===
    plot_results_optimized(res_dict)

def plot_results_optimized(data):
    """
    生成高精度、自适应坐标轴的专业对比图
    """
    # 设置 Seaborn 风格
    sns.set_theme(style="whitegrid", context="talk") # context="talk" 会让字体稍微大一点，适合展示
    
    # 准备画布
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- 左图: 评分指标 (越高越好) ---
    score_metrics = ['HOTA', 'AssA', 'IDF1', 'DetA']
    score_data = {
        'Metric': [],
        'Score': [],
        'Method': []
    }
    for m in score_metrics:
        score_data['Metric'].extend([m, m])
        score_data['Score'].extend([data[m][0], data[m][1]])
        score_data['Method'].extend(['Original', 'Improved'])
    
    df_score = pd.DataFrame(score_data)
    
    # 绘图
    bar1 = sns.barplot(
        data=df_score, 
        x='Metric', 
        y='Score', 
        hue='Method', 
        ax=axes[0], 
        palette="viridis",
        edgecolor="black", # 给柱子加黑边，更清晰
        linewidth=1
    )
    
    axes[0].set_title("Tracking Scores (Higher is Better)", fontsize=18, fontweight='bold', pad=20)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score (0-1)", fontsize=14)
    
    # 【优化1】自适应 Y 轴
    # 找到最大值，并留出 15% 的头部空间放数字
    max_score = df_score['Score'].max()
    axes[0].set_ylim(0, max_score * 1.25) 
    
    # 【优化2】高精度数值标签
    for container in axes[0].containers:
        # fmt='%.3f' 保留三位小数 (如 0.173)
        axes[0].bar_label(container, fmt='%.3f', padding=3, fontsize=13, fontweight='bold')

    # --- 右图: 错误计数指标 (越低越好) ---
    count_metrics = ['IDSW', 'Count', 'Frag']
    count_data = {
        'Metric': [],
        'Count': [],
        'Method': []
    }
    for m in count_metrics:
        count_data['Metric'].extend([m, m])
        count_data['Count'].extend([data[m][0], data[m][1]])
        count_data['Method'].extend(['Original', 'Improved'])
        
    df_count = pd.DataFrame(count_data)

    bar2 = sns.barplot(
        data=df_count, 
        x='Metric', 
        y='Count', 
        hue='Method', 
        ax=axes[1], 
        palette="magma",
        edgecolor="black",
        linewidth=1
    )
    
    axes[1].set_title("Error Counts (Lower is Better)", fontsize=18, fontweight='bold', pad=20)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count", fontsize=14)
    
    # 自适应 Y 轴
    max_count = df_count['Count'].max()
    axes[1].set_ylim(0, max_count * 1.15)
    
    # 整数标签
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.0f', padding=3, fontsize=13, fontweight='bold')

    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(current_dir, 'comparison_result_optimized.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 高清图表已保存至: {save_path}")

if __name__ == '__main__':
    run_evaluation_and_plot()