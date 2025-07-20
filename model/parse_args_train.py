from model.utils import *

def parse_args():
    """训练参数配置"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # 模型选择
    parser.add_argument('--model', type=str, default='DNANet')
    # DNANet模型参数
    parser.add_argument('--channel_size', type=str, default='three',
                        help='通道配置 (one, two, three, four)')
    parser.add_argument('--backbone', type=str, default='resnet_18')
    parser.add_argument('--deep_supervision', type=str, default='True', help='是否使用深度监督 (True, False)')

    # 数据集和预处理
    parser.add_argument('--mode', type=str, default='TXT', help='数据集加载模式')
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='数据集名称 (NUDT-SIRST, NUAA-SIRST, NUST-SIRST)')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='数据集划分方法 (50_50, 10000_100 for NUST-SIRST)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='数据加载线程数')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入通道数，预处理使用RGB=3')
    parser.add_argument('--base_size', type=int, default=256,
                        help='基础图像尺寸')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='裁剪图像尺寸')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='训练轮数 (默认: 1500)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='开始训练的轮次 (默认: 0)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        metavar='N', help='训练批次大小 (默认: 16)')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        metavar='N', help='测试批次大小 (默认: 16)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='最小学习率')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='优化器 (Adam, Adagrad)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='学习率 (默认: 0.001)')
    # CUDA和日志
    parser.add_argument('--gpus', type=str, default='0',
                        help='使用的GPU，例如指定 1,3')

    args = parser.parse_args()

    # 创建结果保存目录
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if args.deep_supervision:
        args.save_dir = "%s_%s_%s_wDS" % (args.dataset, args.model, dt_string)
    else:
        args.save_dir = "%s_%s_%s_woDS" % (args.dataset, args.model, dt_string)
    
    return args 
