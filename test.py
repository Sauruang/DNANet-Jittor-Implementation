# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio
import os
import shutil
# Jittor and visulization
from jittor import transform as transforms
import jittor as jt

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

jt.flags.use_cuda = 1

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = os.path.join(args.root, args.dataset)
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data - 移除 ImageNormalize 避免广播错误，在测试循环中手动归一化
        input_transform = transforms.Compose([
                          transforms.ToTensor()])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = testset.set_attrs(batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = jt.load('result/' + args.model_dir)
        self.model.load_state_dict(checkpoint)

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with jt.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                # 修复维度顺序：从 (batch, height, width, channels) 到 (batch, channels, height, width)
                if len(data.shape) == 4 and data.shape[-1] == 3:
                    data = data.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                
                # 手动归一化 (ImageNet标准)
                mean = jt.array([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = jt.array([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                data = (data - mean) / std
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                num += 1

                losses.update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids))
            
            print(f'测试完成！')
            print(f'测试损失: {losses.avg:.4f}')
            print(f'平均IoU: {mean_IOU:.4f}')
            print(f'PD (检测概率): {PD:.4f}')
            print(f'FA (误报率): {FA:.8f}')
            
            try:
                scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
                             {'number_record1': FA, 'number_record2': PD})
                print(f'PD_FA结果已保存到: {dataset_dir}/value_result/{args.st_model}_PD_FA_255.mat')
            except Exception as e:
                print(f'保存PD_FA结果时出错: {e}')
            
            # 复制训练日志到数据集value_result目录
            model_dir_parts = args.model_dir.split('/')
            if len(model_dir_parts) > 0:
                result_dir = model_dir_parts[0]  # 例如: NUDT_BEST
                
                source_log = f'result/{result_dir}/epoch_continuous_log.txt'
                target_log = f'{dataset_dir}/value_result/{args.st_model}_epoch_continuous_log.txt'
                shutil.copy2(source_log, target_log)
                print(f'训练日志已复制到: {dataset_dir}/value_result/{args.st_model}_epoch_continuous_log.txt')


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args) 