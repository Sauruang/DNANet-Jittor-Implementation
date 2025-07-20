from model.utils import *

def parse_args():
    """测试参数配置"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # 模型选择
    parser.add_argument('--model', type=str, default='DNANet',
                        help='模型名称 (DNANet)')

    # DNANet模型参数
    parser.add_argument('--channel_size', type=str, default='three',
                        help='通道配置 (one, two, three, four)')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='骨干网络 (resnet_18, resnet_34)')
    parser.add_argument('--deep_supervision', type=str, default='True', help='是否使用深度监督 (True, False)')

    # 数据集和预处理
    parser.add_argument('--mode', type=str, default='TXT', help='数据集加载模式')
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='数据集名称 (NUDT-SIRST, NUAA-SIRST, NUST-SIRST)')
    parser.add_argument('--st_model', type=str, default='NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS',
                        help='模型标识名称，用于结果文件命名')
    parser.add_argument('--model_dir', type=str,
                        default = 'NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar',
                        help = '预训练模型路径 (相对于result/目录)')
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

    # 测试参数
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='测试批次大小 (默认: 1)')

    # CUDA和日志
    parser.add_argument('--gpus', type=str, default='0',
                        help='使用的GPU，例如指定 1,3')

    # ROC分析阈值
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='ROC分析阈值数量')

    args = parser.parse_args()

    return args 
