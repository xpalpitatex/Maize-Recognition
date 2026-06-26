import os
import argparse
import numpy as np
import pandas as pd

# 设置matplotlib后端，避免tkinter线程问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime
from tqdm import tqdm
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import glob
import sys

# 将项目根目录添加到PATH
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(src_dir)
sys.path.insert(0, root_dir)

# 从改进的模型文件导入组件
from src.improved_corn_growth_model import (
    ImprovedCornGrowthStageModel, CONFIG, CornDatasetBalanced, 
    get_transforms, calculate_class_weights, FocalLoss,
    train_one_epoch, validate, get_lr_scheduler, save_model,
    MixupCutmixCallback
)

from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入数据处理器
from src.data_processor_pytorch import CornDataProcessor

# 获取项目根目录
def get_project_root():
    # 获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取src/trian目录
    trian_dir = os.path.dirname(current_file)
    # 获取src目录
    src_dir = os.path.dirname(trian_dir)
    # 获取项目根目录
    project_root = os.path.dirname(src_dir)
    return project_root

# 项目根目录路径
PROJECT_ROOT = get_project_root()

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='训练改进版玉米生育期识别模型')
    
    # 数据和模型相关参数
    parser.add_argument('--data_dir', type=str, default='s_classes',
                        help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='models/improved_model',
                        help='模型输出目录')
    parser.add_argument('--img_size', type=int, default=CONFIG['img_size'],
                        help='图像大小')
    parser.add_argument('--batch_size', type=int, default=16,  # 减小批量大小
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--model_type', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'],
                        help='特征提取网络类型')
    
    # 学习率和优化器参数
    parser.add_argument('--learning_rate', type=float, default=1e-4,  # 增大学习率
                        help='基础学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6,  # 调整最小学习率
                        help='最小学习率(学习率衰减下限)')
    parser.add_argument('--weight_decay', type=float, default=CONFIG['weight_decay'],
                        help='权重衰减系数(L2正则化)')
    parser.add_argument('--dropout', type=float, default=CONFIG['dropout_rate'],
                        help='Dropout比率')
    parser.add_argument('--grad_clip', type=float, default=0.5,  # 减小梯度裁剪阈值
                        help='梯度裁剪阈值')
    
    # 特征和损失函数相关参数
    parser.add_argument('--no_time', action='store_true', #store_false
                        help='不使用时间特征')
    parser.add_argument('--no_attention', action='store_true', 
                        help='不使用注意力机制')
    parser.add_argument('--no_focal_loss', action='store_true', 
                        help='不使用Focal Loss')
    parser.add_argument('--gamma', type=float, default=1.0,  # 减小gamma值
                        help='Focal Loss的gamma参数')
    parser.add_argument('--no_mixup', action='store_true', default=False,
                        help='不使用MixUp和CutMix')
    parser.add_argument('--no_balanced_sampling', action='store_true', 
                        help='不使用平衡采样')
    parser.add_argument('--class_weights_power', type=float, default=CONFIG['class_weights_power'],
                        help='类权重计算的幂参数')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='不使用预训练模型权重(从头训练)')
    parser.add_argument('--no_amp', action='store_true',
                        help='不使用自动混合精度训练(解决CUDA内存错误)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--early_stopping', type=int, default=10000,
                        help='早停轮数 (设为10000表示实际上禁用早停)')
    
    # 分布式训练相关参数 - 默认禁用分布式训练
    parser.add_argument('--distributed', action='store_true', default=False,  # 改为False
                        help='是否使用分布式训练')
    parser.add_argument('--world_size', type=int, default=1,  # 改为1
                        help='进程数量，通常为GPU数量')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29500',
                        help='初始化进程组的URL')
    parser.add_argument('--dist_backend', type=str, default='nccl',  # 改为nccl
                        help='分布式后端')
    parser.add_argument('--gpu', type=int, default=0,  # 指定使用GPU 0
                        help='要使用的GPU ID，不指定则使用所有可用GPU')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='多进程训练中的本地进程排名')
    
    # 学习率调度器选择 - 新增参数
    parser.add_argument('--scheduler_type', type=str, default='warm_restarts',
                        choices=['warm_restarts', 'cosine_smooth', 'exponential'],
                        help='学习率调度器类型: warm_restarts(原始有震荡), cosine_smooth(平滑余弦), exponential(指数衰减)')
    # 新增：只评估模式
    parser.add_argument('--eval_only', action='store_true', help='只做评估，不训练')
    
    # 新增：数据划分方式选择
    parser.add_argument('--split_by', type=str, default='random_stratified',
                        choices=['random_stratified', 'station_year'],
                        help='数据划分方式: random_stratified(随机分层), station_year(按站点×年份分组，避免泄漏)')
    
    parser.add_argument('--no_contrastive', action='store_true', 
                        help='不使用对比学习')
    parser.add_argument('--no_multiscale', action='store_true', 
                        help='不使用多尺度特征提取')
    parser.add_argument('--no_difficult_enhancement', action='store_true', 
                        help='不使用困难样本增强')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置CUDA的确定性(可能会影响性能)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_date_from_filename(filename):
    """从文件名中提取日期
    支持格式:
    1. O3543_010405_20230618160000_01.jpg (设备编号_相机ID_时间戳_序号)
    2. 任何包含8位数字日期的文件名 (YYYYMMDD)
    """
    try:
        # 尝试从格式 "O3543_010405_20230618160000_01.jpg" 提取日期
        parts = filename.split('_')
        if len(parts) >= 3:
            # 第三部分通常是时间戳
            timestamp = parts[2]
            if len(timestamp) >= 14 and timestamp[:14].isdigit():
                # 格式: YYYYMMDDHHMMSS
                date_str = timestamp[:8]
                time_str = timestamp[8:14]
                datetime_str = f"{date_str}{time_str}"
                return datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
        
        # 尝试查找任何包含8位数字的部分作为日期
        for part in parts:
            if len(part) >= 8 and part[:8].isdigit():
                date_str = part[:8]
                if date_str.startswith(('19', '20')):  # 确保这是一个有效的年份
                    return datetime.strptime(date_str, '%Y%m%d')
    except Exception as e:
        print(f"从文件名 '{filename}' 提取日期时出错: {str(e)}")
    
    return None

def prepare_data(data_dir, stage_to_idx):
    """准备数据集，支持多级目录结构 (站点/年份/生育期)"""
    # 确保使用绝对路径
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
    
    print(f"准备数据集: {data_dir}")
    data = []
    
    # 获取所有站点目录 (如O3543, O3544等)
    station_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"找到以下站点目录: {station_dirs}")
    
    for station in station_dirs:
        station_path = os.path.join(data_dir, station)
        
        # 获取年份目录
        year_dirs = [d for d in os.listdir(station_path) if os.path.isdir(os.path.join(station_path, d))]
        print(f"站点 {station}: 找到以下年份目录: {year_dirs}")
        
        for year in year_dirs:
            year_path = os.path.join(station_path, year)
            
            # 获取生育期目录
            stage_dirs = [d for d in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, d))]
            print(f"站点 {station}, 年份 {year}: 找到以下生育期目录: {stage_dirs}")
            
            for stage in stage_dirs:
                # 跳过不在映射中的阶段
                if stage not in stage_to_idx:
                    print(f"警告: 跳过未知生育期目录 {stage}，因为它不在stage_to_idx映射中")
                    continue
                    
                stage_path = os.path.join(year_path, stage)
                stage_idx = stage_to_idx[stage]
                
                # 获取该生育期目录下的所有图像
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files.extend(glob.glob(os.path.join(stage_path, f'*{ext}')))
                
                print(f"站点 {station}, 年份 {year}, 阶段 {stage}: 找到 {len(image_files)} 张图像")
                
                # 处理图像
                for img_path in image_files:
                    # 提取日期
                    filename = os.path.basename(img_path)
                    date = extract_date_from_filename(filename)
                    
                    # 如果无法提取日期，使用文件修改时间
                    if date is None:
                        file_mtime = os.path.getmtime(img_path)
                        date = datetime.fromtimestamp(file_mtime)
                    
                    # 添加记录
                    data.append({
                        'path': img_path,
                        'filename': filename,
                        'station': station,
                        'year': year,
                        'stage': stage,
                        'stage_idx': stage_idx,
                        'date': date,
                        'month': date.month,
                        'day': date.day
                    })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    if len(df) == 0:
        raise ValueError(f"数据集为空！请检查数据目录结构 '{data_dir}' 和stage_to_idx映射关系。")
    
    print(f"数据集准备完成，共 {len(df)} 张图像")
    
    # 打印各类别数量
    class_counts = df['stage_idx'].value_counts().sort_index()
    print("各生育期图像数量统计:")
    for idx, count in class_counts.items():
        stage = next((k for k, v in stage_to_idx.items() if v == idx), None)
        print(f"  阶段 {stage} (索引 {idx}): {count} 张图像")
    
    # 打印站点统计
    station_counts = df['station'].value_counts()
    print("各站点图像数量统计:")
    for station, count in station_counts.items():
        print(f"  站点 {station}: {count} 张图像")
    
    return df

def plot_class_distribution(df, stage_to_english, output_dir):
    """绘制类别分布图（使用英文）"""
    try:
        plt.figure(figsize=(12, 6))
        
        # 计算每个类别的样本数
        class_counts = df['stage'].value_counts().sort_index()
        
        # 准备标签 - 使用英文
        labels = [f"{stage} ({stage_to_english.get(stage, 'Unknown')})" for stage in class_counts.index]
        
        # 绘制柱状图 - 使用显式的x轴范围避免刻度错位
        x_positions = range(len(class_counts))
        sns.barplot(x=x_positions, y=class_counts.values, palette='rocket')
        plt.title('Growth Stage Sample Distribution', fontsize=16)
        plt.xlabel('Growth Stage', fontsize=14)
        plt.ylabel('Number of Samples', fontsize=14)
        plt.xticks(x_positions, labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()  # 确保关闭图形
        
    except Exception as e:
        print(f"绘制类别分布图时出错: {str(e)}")
        plt.close()  # 确保即使出错也关闭图形

def train_model(gpu, args):
    """训练模型的主函数"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 简化设备设置 - 单GPU训练
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 创建输出目录
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # 生育期映射 - 根据实际目录调整
    # 正确映射：O编号目录下有年份子目录，每个年份目录下有标准的生育期目录(11、21等)
    stage_to_name = {
        '11': '播种期',  
        '21': '出苗期',  
        '31': '三叶期',  
        '41': '七叶期',  
        '61': '拔节期',
        '71': '抽雄期',  
        '81': '乳熟期',
        '91': '成熟期',
        '99': '收割后'
    }
    
    # 英文标签映射
    stage_to_english = {
        '11': 'Seeding',  
        '21': 'Emergence',  
        '31': 'Three-leaf',  
        '41': 'Seven-leaf',  
        '61': 'Jointing',
        '71': 'Tasseling',  
        '81': 'Milk',
        '91': 'Maturity',
        '99': 'Post-harvest'
    }
    
    # 生育期映射到索引
    stage_to_idx = {
        '11': 0,  # 播种期
        '21': 1,  # 出苗期
        '31': 2,  # 三叶期 
        '41': 3,  # 七叶期
        '61': 4,  # 拔节期
        '71': 5,  # 抽雄期
        '81': 6,  # 乳熟期
        '91': 7,  # 成熟期
        '99': 8   # 收割后
    }
    
    # 更新模型的类别数量
    num_classes = len(stage_to_idx)
    
    print(f"检测到 {num_classes} 个生育期类别:")
    for stage, name in stage_to_english.items():
        print(f"  {stage}: {name} (索引: {stage_to_idx[stage]})")
    print(f"数据目录: {args.data_dir} 包含站点目录(O3543等)，每个站点目录下有年份目录，年份目录下有生育期目录(11等)")
    
    # 准备数据
    df = prepare_data(args.data_dir, stage_to_idx)
    
    # 检查数据是否为空
    if len(df) == 0:
        raise ValueError("数据集为空！请检查数据目录结构和stage_to_idx映射关系。")
    
    # 绘制类别分布图
    plot_class_distribution(df, stage_to_english, args.output_dir)
    
    # 划分训练集、验证集和测试集 - 根据用户选择使用不同策略
    if args.split_by == 'random_stratified':
        print("\n=== 使用随机分层划分策略 ===")
        print("优势：训练稳定，过拟合程度较轻")
        print("注意：存在一定程度的数据泄漏，但泛化评估更稳定")
        
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=args.seed, stratify=df['stage_idx']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df['stage_idx']
        )
        
    elif args.split_by == 'station_year':
        print("\n=== 使用站点×年份分组划分策略 ===")
        print("优势：避免数据泄漏，更真实的泛化评估")
        print("注意：可能导致类别分布不均，训练难度增加")
        
        from sklearn.model_selection import GroupShuffleSplit
        
        # 创建分组标识符 (station_year)
        df['group_id'] = df['station'] + '_' + df['year'].astype(str)
        groups = df['group_id'].values
        
        # 第一次划分：训练集 vs (验证集+测试集)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
        train_idx, temp_idx = next(gss1.split(df, df['stage_idx'], groups))
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df = df.iloc[temp_idx].reset_index(drop=True)
        temp_groups = temp_df['group_id'].values
        
        # 第二次划分：验证集 vs 测试集
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
        val_idx, test_idx = next(gss2.split(temp_df, temp_df['stage_idx'], temp_groups))
        
        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)
        
        print(f"训练组合: {sorted(train_df['group_id'].unique())}")
        print(f"验证组合: {sorted(val_df['group_id'].unique())}")
        print(f"测试组合: {sorted(test_df['group_id'].unique())}")
    
    print(f"训练集: {len(train_df)} 样本 ({len(train_df)/len(df):.1%})")
    print(f"验证集: {len(val_df)} 样本 ({len(val_df)/len(df):.1%})")
    print(f"测试集: {len(test_df)} 样本 ({len(test_df)/len(df):.1%})")
    
    # 分析划分后的数据分布
    print(f"\n=== 数据分布分析 ===")
    for name, subset in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        if len(subset) > 0:
            # 站点分布
            station_dist = subset['station'].value_counts()
            print(f"\n{name}站点分布:")
            for station, count in station_dist.items():
                percentage = count / len(subset) * 100
                print(f"  {station}: {count}张 ({percentage:.1f}%)")
            
            # 生育期分布
            stage_dist = subset['stage'].value_counts().sort_index()
            print(f"{name}生育期分布:")
            for stage, count in stage_dist.items():
                percentage = count / len(subset) * 100
                print(f"  {stage}: {count}张 ({percentage:.1f}%)")
    
    # 保存划分信息
    split_info = {
        'split_method': args.split_by,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_stations': sorted(train_df['station'].unique().tolist()),
        'val_stations': sorted(val_df['station'].unique().tolist()),
        'test_stations': sorted(test_df['station'].unique().tolist()),
        'train_years': sorted(train_df['year'].unique().tolist()),
        'val_years': sorted(val_df['year'].unique().tolist()),
        'test_years': sorted(test_df['year'].unique().tolist())
    }
    
    # 如果使用分组划分，还要保存分组信息
    if args.split_by == 'station_year':
        split_info.update({
            'train_groups': sorted(train_df['group_id'].unique().tolist()),
            'val_groups': sorted(val_df['group_id'].unique().tolist()),
            'test_groups': sorted(test_df['group_id'].unique().tolist())
        })
    
    with open(os.path.join(args.output_dir, 'data_split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # 保存划分后的DataFrames，支持eval_only模式
    train_df.to_csv(os.path.join(args.output_dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_df.csv'), index=False)
    print(f"✅ 已保存数据划分文件到 {args.output_dir}")
    
    # 更新配置
    config = CONFIG.copy()
    config['img_size'] = args.img_size
    config['base_lr'] = args.learning_rate
    config['min_lr'] = args.min_lr
    config['weight_decay'] = args.weight_decay
    config['dropout_rate'] = args.dropout
    config['gamma'] = args.gamma
    config['class_weights_power'] = args.class_weights_power
    config['patience'] = args.early_stopping
    # 保存模型类型到配置中，确保评估时使用正确的模型类型
    config['model_type'] = args.model_type
    config['base_model'] = args.model_type
    config['num_classes'] = num_classes
    config['use_temporal_info'] = not args.no_time
    
    # 创建并清空记录文件
    record_files = ['train_loss.txt', 'val_loss.txt', 'train_acc.txt', 'val_acc.txt']
    for file in record_files:
        with open(os.path.join(args.output_dir, file), 'w') as f:
            f.write(f"# 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 模型类型: {args.model_type}\n")
            f.write(f"# 批量大小: {args.batch_size}\n")
            f.write(f"# 学习率: {args.learning_rate}\n")
            f.write(f"# 使用时间特征: {not args.no_time}\n")
            f.write(f"# 使用注意力机制: {not args.no_attention}\n")
            f.write(f"# 使用自动混合精度: {not args.no_amp}\n\n")
    
    # 数据变换
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # 创建数据集 - 禁用Dataset内部的平衡采样，改用WeightedRandomSampler
    train_dataset = CornDatasetBalanced(
        train_df, 
        transform=train_transform, 
        include_time=not args.no_time,
        balanced_sampling=False,  # 总是禁用Dataset内部平衡采样
        class_weights_power=args.class_weights_power,
        img_size=args.img_size
    )
    
    val_dataset = CornDatasetBalanced(
        val_df, 
        transform=val_transform, 
        include_time=not args.no_time,
        balanced_sampling=False,
        img_size=args.img_size
    )
    
    # 创建数据加载器 - 单GPU训练，避免多进程序列化问题
    if not args.no_balanced_sampling:
        # 使用WeightedRandomSampler实现平衡采样
        from torch.utils.data import WeightedRandomSampler
        
        # 计算每个样本的权重 - 先缓存value_counts避免重复统计
        stage_counts = train_df['stage_idx'].value_counts()
        sample_weights = train_df['stage_idx'].map(
            lambda x: (1.0 / stage_counts[x]) ** args.class_weights_power
        ).values
        
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(train_df),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,  # 使用sampler时不能同时shuffle
            num_workers=0,  # 设为0避免Windows多进程pickle问题
            pin_memory=False,  # 禁用pin_memory避免潜在问题
            drop_last=True
        )
        print("✅ 使用WeightedRandomSampler实现平衡采样")
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # 设为0避免Windows多进程pickle问题
            pin_memory=False,  # 禁用pin_memory避免潜在问题
            drop_last=True
        )
        print("❌ 禁用平衡采样，使用随机采样")
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 设为0避免Windows多进程pickle问题
        pin_memory=False   # 禁用pin_memory避免潜在问题
    )
    
    # 计算类别权重 - calculate_class_weights已返回torch.tensor，无需重复包装
    class_weights = calculate_class_weights(train_df['stage_idx'].values, power=args.class_weights_power).to(device)
    
    print(f"类别权重: {class_weights}")
    
    # 创建模型
    model = ImprovedCornGrowthStageModel(
        num_classes=num_classes,
        model_type=args.model_type,
        pretrained=not args.no_pretrained,
        dropout_rate=args.dropout,
        include_time=not args.no_time,
        use_attention=not args.no_attention,
        use_contrastive=not args.no_contrastive,
        use_multiscale=not args.no_multiscale,
        use_difficult_enhancement=not args.no_difficult_enhancement
    )
    
    print(f"创建模型: {args.model_type}, 类别数: {num_classes}")
    print(f"预训练权重: {'禁用' if args.no_pretrained else '启用'}")
    print(f"时间特征: {'启用' if not args.no_time else '禁用'}")
    print(f"注意力机制: {'启用' if not args.no_attention else '禁用'}")
    print(f"多尺度特征: {'启用' if not args.no_multiscale else '禁用'}")
    print(f"困难样本增强: {'启用' if not args.no_difficult_enhancement else '禁用'}")
    print(f"对比学习: {'启用' if not args.no_contrastive else '禁用'}")
    print(f"Focal Loss: {'启用' if not args.no_focal_loss else '禁用'}")
    # 修正MixUp/CutMix状态显示逻辑
    mixup_enabled = not args.no_mixup and CONFIG.get('use_mixup', False)
    print(f"MixUp/CutMix: {'启用' if mixup_enabled else '禁用'}")
    print(f"平衡采样: {'启用' if not args.no_balanced_sampling else '禁用'}")
    print(f"学习率: {args.learning_rate}, 最小学习率: {args.min_lr}")
    print(f"权重衰减: {args.weight_decay}, Dropout率: {args.dropout}")
    print(f"Focal Loss gamma: {args.gamma}, 类权重幂: {args.class_weights_power}")
    print(f"早停轮数: {args.early_stopping}")
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 定义损失函数
    if args.no_focal_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("使用交叉熵损失函数")
    else:
        criterion = FocalLoss(gamma=args.gamma, alpha=class_weights)
        print(f"使用Focal Loss, gamma={args.gamma}")
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 定义学习率调度器 - 修改这部分
    if args.scheduler_type == 'warm_restarts':
        # 原始版本 - CosineAnnealingWarmRestarts (有震荡)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=args.min_lr
        )
        scheduler_name = "CosineAnnealingWarmRestarts (热重启，有震荡)"
        print(f"🔄 使用学习率调度器: {scheduler_name}")
        print("   特点: 周期性重启，可能在第75轮和155轮左右出现震荡")
        
    elif args.scheduler_type == 'cosine_smooth':
        # 平滑版本 - CosineAnnealingLR (无震荡)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs, 
            eta_min=args.min_lr
        )
        scheduler_name = "CosineAnnealingLR (平滑余弦退火)"
        print(f"📈 使用学习率调度器: {scheduler_name}")
        print("   特点: 平滑下降，无震荡，预期性能更稳定")
        
    elif args.scheduler_type == 'exponential':
        # 指数衰减版本 - ExponentialLR
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=0.98  # 每轮衰减2%
        )
        scheduler_name = "ExponentialLR (指数衰减)"
        print(f"📉 使用学习率调度器: {scheduler_name}")
        print("   特点: 指数衰减，稳定收敛")
    
    else:
        raise ValueError(f"不支持的调度器类型: {args.scheduler_type}")
    
    # MixUp和CutMix回调 - 修正逻辑
    mixup_cutmix = None
    use_mixup_cutmix = not args.no_mixup and CONFIG.get('use_mixup', False)
    if use_mixup_cutmix:
        mixup_cutmix = MixupCutmixCallback(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=0.5,
            switch_prob=0.5
        )
        print("✅ 已启用MixUp和CutMix数据增强")
    else:
        print("❌ 已禁用MixUp和CutMix数据增强")
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': [],
        'learning_rates': [],  # 添加学习率记录
        'scheduler_type': args.scheduler_type,  # 记录调度器类型
        'scheduler_name': scheduler_name  # 记录调度器名称
    }
    
    # 自动混合精度训练 - scaler在train_one_epoch内部创建和管理，这里不需要
    # scaler = torch.cuda.amp.GradScaler() if not args.no_amp else None
    
    for epoch in range(args.epochs):
        # 定期清理GPU缓存
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Epoch {epoch}: GPU内存清理完成")
        
        # 检查模型参数的健康状态
        if epoch % 20 == 0:
            nan_params = 0
            inf_params = 0
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params += 1
                if torch.isinf(param).any():
                    inf_params += 1
            
            if nan_params > 0 or inf_params > 0:
                print(f"警告: 检测到 {nan_params} 个NaN参数, {inf_params} 个Inf参数")
                # 如果参数异常，可以考虑重新加载最佳模型
                if nan_params > 5 or inf_params > 5:
                    print("参数异常过多，建议停止训练")
                    break
        
        # 训练一个epoch
        try:
            train_result = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                use_amp=not args.no_amp,
                use_mixup=use_mixup_cutmix,
                mixup_cutmix=mixup_cutmix,
                grad_clip=args.grad_clip
            )
            
            # 解包返回值
            if len(train_result) == 5:
                train_loss, train_acc, train_cls_loss, train_con_loss, difficult_accs = train_result
            else:
                train_loss, train_acc = train_result
                
        except Exception as e:
            print(f"训练epoch {epoch+1}失败: {str(e)}")
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 跳过这个epoch
            continue
        
        # 验证
        try:
            val_result = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=not args.no_amp
            )
            
            # 解包验证结果
            if len(val_result) == 7:
                val_loss, val_acc, val_cls_loss, val_con_loss, val_difficult_accs, val_labels, val_preds = val_result
            else:
                val_loss, val_acc, val_labels, val_preds = val_result
                
        except Exception as e:
            print(f"验证epoch {epoch+1}失败: {str(e)}")
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 使用上一次的验证结果或默认值
            val_loss, val_acc = float('inf'), 0.0
            val_labels, val_preds = [], []
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
             f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
             f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
             f"LR: {current_lr:.2e}")
        
        # 保存历史记录
        with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
        
        # 实时追加当前epoch的训练指标到txt文件
        with open(os.path.join(args.output_dir, 'train_loss.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: {train_loss}\n")
        
        with open(os.path.join(args.output_dir, 'val_loss.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: {val_loss}\n")
            
        with open(os.path.join(args.output_dir, 'train_acc.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: {train_acc}\n")
            
        with open(os.path.join(args.output_dir, 'val_acc.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: {val_acc}\n")
        
        # 每个epoch都更新训练曲线图
        plot_training_history(history, args.output_dir)
        
        # 保存最佳模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            # 保存最佳模型
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                stage_mapping=stage_to_idx,
                val_acc=val_acc,
                val_loss=val_loss,
                best_val_acc=best_val_acc,
                output_dir=args.output_dir
            )
        
        # 每5个epoch保存一次checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'config': config,
                'stage_mapping': stage_to_idx,
                'model_type': args.model_type,  # 明确保存模型类型
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 训练结束后进行最终评估，不需要再次绘制曲线
    evaluate_best_model(args.output_dir, test_df, stage_to_idx, stage_to_english, device)

def plot_training_history(history, output_dir):
    """绘制训练历史"""
    try:
        # 获取调度器信息
        scheduler_name = history.get('scheduler_name', 'Unknown Scheduler')
        scheduler_type = history.get('scheduler_type', 'unknown')
        
        # 创建并保存损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Curves - {scheduler_name}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建并保存准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title(f'Accuracy Curves - {scheduler_name}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建并保存学习率变化曲线
        if 'learning_rates' in history and history['learning_rates']:
            plt.figure(figsize=(10, 6))
            plt.plot(history['learning_rates'], label='Learning Rate', color='green', linewidth=2)
            plt.title(f'Learning Rate Schedule - {scheduler_name}', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Learning Rate', fontsize=14)
            plt.yscale('log')  # 使用对数尺度
            
            # 添加调度器类型说明
            if scheduler_type == 'warm_restarts':
                plt.text(0.02, 0.98, '注意: 可能在第75轮和155轮左右出现重启震荡', 
                        transform=plt.gca().transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        verticalalignment='top')
            elif scheduler_type == 'cosine_smooth':
                plt.text(0.02, 0.98, '特点: 平滑下降，无震荡', 
                        transform=plt.gca().transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                        verticalalignment='top')
            elif scheduler_type == 'exponential':
                plt.text(0.02, 0.98, '特点: 指数衰减，稳定收敛', 
                        transform=plt.gca().transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                        verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 创建综合训练历史图
        plt.figure(figsize=(18, 12))
        
        # 损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 准确率曲线
        plt.subplot(2, 3, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Curves', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 学习率变化
        if 'learning_rates' in history and history['learning_rates']:
            plt.subplot(2, 3, 3)
            plt.plot(history['learning_rates'], color='green', linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.yscale('log')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # 性能统计
        plt.subplot(2, 3, 4)
        if history['val_acc']:
            max_val_acc = max(history['val_acc'])
            max_val_acc_epoch = history['val_acc'].index(max_val_acc) + 1
            final_val_acc = history['val_acc'][-1]
            final_train_acc = history['train_acc'][-1]
            
            stats_text = f"""
调度器: {scheduler_name}

最佳验证准确率: {max_val_acc:.4f}
最佳轮次: Epoch {max_val_acc_epoch}

最终验证准确率: {final_val_acc:.4f}
最终训练准确率: {final_train_acc:.4f}

过拟合程度: {abs(final_train_acc - final_val_acc):.4f}

总训练轮次: {len(history['val_acc'])}
            """
            
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        
        # 训练进度对比 (最近20轮)
        plt.subplot(2, 3, 5)
        if len(history['val_acc']) > 20:
            recent_epochs = range(len(history['val_acc']) - 20, len(history['val_acc']))
            recent_train_acc = history['train_acc'][-20:]
            recent_val_acc = history['val_acc'][-20:]
            plt.plot(recent_epochs, recent_train_acc, label='Train Acc (Recent)')
            plt.plot(recent_epochs, recent_val_acc, label='Val Acc (Recent)')
            plt.title('Recent 20 Epochs', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # 学习率震荡分析 (如果是warm_restarts)
        plt.subplot(2, 3, 6)
        if scheduler_type == 'warm_restarts' and 'learning_rates' in history:
            plt.plot(history['learning_rates'], color='red', linewidth=2, alpha=0.7)
            plt.title('LR Restarts Analysis', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.yscale('log')
            
            # 标注可能的重启点
            if len(history['learning_rates']) > 75:
                plt.axvline(x=75, color='orange', linestyle='--', alpha=0.8, label='~Epoch 75')
            if len(history['learning_rates']) > 155:
                plt.axvline(x=155, color='purple', linestyle='--', alpha=0.8, label='~Epoch 155')
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, f'调度器类型: {scheduler_type}\n\n'
                             f'特点:\n'
                             f'{"平滑收敛，无震荡" if scheduler_type == "cosine_smooth" else ""}'
                             f'{"指数衰减，稳定" if scheduler_type == "exponential" else ""}',
                    transform=plt.gca().transAxes, fontsize=12,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.axis('off')
        
        plt.suptitle(f'Training History - {scheduler_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"绘制训练历史图时出错: {str(e)}")
        plt.close()  # 确保即使出错也关闭图形

def evaluate_best_model(output_dir, test_df, stage_to_idx, stage_to_english, device):
    """评估最佳模型"""
    print("\n评估最佳模型...")
    
    # 检查最佳模型文件是否存在
    best_model_path = os.path.join(output_dir, 'improved_corn_model_best.pth')
    if not os.path.exists(best_model_path):
        print(f"❌ 最佳模型文件不存在: {best_model_path}")
        print("可能原因：训练过程中所有epoch都失败，没有保存任何模型")
        print("建议：检查训练日志，修复训练问题后重新训练")
        return
    
    # 加载最佳模型
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        print(f"✅ 成功加载最佳模型: {best_model_path}")
    except Exception as e:
        print(f"❌ 加载最佳模型失败: {str(e)}")
        return
    
    # 提取配置
    config = checkpoint.get('config', CONFIG)
    
    # 打印加载的配置信息
    print(f"加载的模型配置: {config}")
    
    # 获取正确的模型类型，优先使用model_type，其次使用base_model，最后使用默认值
    model_type = checkpoint.get('model_type', 
                 config.get('model_type', 
                 config.get('base_model', CONFIG['base_model'])))
    
    print(f"使用模型类型: {model_type}")
    
    # 确定是否使用时间特征
    include_time = config.get('use_temporal_info', CONFIG['use_temporal_info'])
    print(f"是否包含时间特征: {include_time}")
    
    # 创建模型 - 使用checkpoint中的完整配置确保与训练时一致
    model = ImprovedCornGrowthStageModel(
        num_classes=config.get('num_classes', len(stage_to_idx)),
        model_type=model_type,
        pretrained=False,
        dropout_rate=config.get('dropout_rate', CONFIG['dropout_rate']),
        include_time=config.get('use_temporal_info', CONFIG['use_temporal_info']),
        use_attention=config.get('use_attention', CONFIG['use_attention']),
        use_contrastive=config.get('use_contrastive_learning', CONFIG['use_contrastive_learning']),
        use_multiscale=config.get('use_multiscale_features', CONFIG['use_multiscale_features']),
        use_difficult_enhancement=config.get('use_difficult_stage_enhancement', CONFIG['use_difficult_stage_enhancement'])
    )
    
    # 加载权重 - 使用strict=False以允许跳过不匹配的参数
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("成功加载模型权重（严格模式）")
    except RuntimeError as e:
        print(f"严格模式加载失败: {str(e)}")
        print("尝试使用非严格模式加载...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("成功以非严格模式加载模型权重（可能有参数缺失或多余）")
    
    model = model.to(device)
    model.eval()
    
    # 创建测试数据集和加载器
    test_transform = get_transforms(config, is_training=False)
    test_dataset = CornDatasetBalanced(
        test_df, 
        transform=test_transform, 
        include_time=include_time,
        balanced_sampling=False,
        img_size=config.get('img_size', 224)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=0  # 避免序列化问题
    )
    
    # 评估
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试"):
            # 正确处理数据格式
            if include_time:
                # 处理包含时间特征的情况
                (inputs, time_features), labels = batch
                inputs = inputs.to(device)
                time_features = time_features.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model((inputs, time_features))
            else:
                # 处理不包含时间特征的情况
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
            
            # 获取预测
            _, preds = torch.max(outputs, 1)
            
            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标 - 使用英文标签
    class_names = [f"{stage} ({stage_to_english.get(stage, 'Unknown')})" for stage in sorted(stage_to_idx.keys())]
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4
    )
    
    # 打印分类报告
    print("\n分类报告:")
    print(report)
    
    # 保存分类报告
    with open(os.path.join(output_dir, 'test_report.txt'), 'w') as f:
        f.write(report)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, stage_to_idx, stage_to_english, output_dir)

def plot_confusion_matrix(y_true, y_pred, stage_to_idx, stage_to_english, output_dir):
    """绘制混淆矩阵（使用英文）"""
    try:
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 英文标签映射
        stage_to_english = {
            '11': 'Seeding',  
            '21': 'Emergence',  
            '31': 'Three-leaf',  
            '41': 'Seven-leaf',  
            '61': 'Jointing',
            '71': 'Tasseling',  
            '81': 'Milk',
            '91': 'Maturity',
            '99': 'Post-harvest'
        }
        
        # 准备标签 - 使用英文
        idx_to_stage = {v: k for k, v in stage_to_idx.items()}
        labels = [f"{idx_to_stage[i]} ({stage_to_english.get(idx_to_stage[i], 'Unknown')})" for i in range(len(idx_to_stage))]
        
        # 创建热图
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Normalized Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建原始混淆矩阵热图
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Confusion Matrix (Raw Counts)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {str(e)}")
        plt.close()  # 确保即使出错也关闭图形

if __name__ == "__main__":
    # 设置CUDA调试模式以获得更详细的错误信息
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU内存状态: {torch.cuda.memory_summary()}")
    
    try:
        if args.eval_only:
            # 只评估模式：加载数据划分和配置，直接评估
            output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(get_project_root(), args.output_dir)
            # 加载数据划分信息
            split_info_path = os.path.join(output_dir, 'data_split_info.json')
            if not os.path.exists(split_info_path):
                raise FileNotFoundError(f"未找到数据划分信息: {split_info_path}")
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)
            # 加载stage_to_idx和stage_to_english
            stage_to_idx = {
                '11': 0, '21': 1, '31': 2, '41': 3, '61': 4, '71': 5, '81': 6, '91': 7, '99': 8
            }
            stage_to_english = {
                '11': 'Seeding',  '21': 'Emergence',  '31': 'Three-leaf',  '41': 'Seven-leaf',  '61': 'Jointing',
                '71': 'Tasseling',  '81': 'Milk', '91': 'Maturity', '99': 'Post-harvest'
            }
            # 加载测试集DataFrame
            test_df_path = os.path.join(output_dir, 'test_df.csv')
            if os.path.exists(test_df_path):
                test_df = pd.read_csv(test_df_path)
            else:
                # 回退：重新生成test_df
                data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(get_project_root(), args.data_dir)
                df = prepare_data(data_dir, stage_to_idx)
                # 按split_info划分
                test_years = split_info['test_years']
                test_stations = split_info['test_stations']
                test_df = df[df['year'].isin(test_years) & df['station'].isin(test_stations)]
            # 设备
            device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
            # 评估
            evaluate_best_model(output_dir, test_df, stage_to_idx, stage_to_english, device)
        else:
            # 直接调用训练函数，不使用分布式训练
            print(f"开始单GPU训练，使用GPU {args.gpu}")
            train_model(args.gpu, args)
    finally:
        # 清理matplotlib资源
        plt.close('all')  # 关闭所有图形
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理GPU缓存
        print("训练完成，资源清理完毕")

# 配置参数 - 可以直接在这里修改
MODEL_PATH = 'models/improved_model/improved_corn_model_best.pth'  # 您的模型路径
TESTDATA_DIR = 'test/testdata'  # 测试数据目录
OUTPUT_DIR = 'test/results/all_formats_test'  # 输出目录
SAMPLES_PER_FOLDER = 3  # 每个文件夹抽取的样本数
DEVICE = 'auto'  # 'auto', 'cuda', 'cpu' 