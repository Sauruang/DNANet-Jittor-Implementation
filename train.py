# jittor and visulization
from tqdm             import tqdm
import jittor as jt
from jittor import optim
from jittor import lr_scheduler  # 修复导入语句
from jittor import nn
import numpy as np
from jittor import transform as transforms
from model.parse_args_train import  parse_args
import os
from datetime import datetime

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

jt.flags.use_cuda = 1

class AdagradCustom(jt.optim.Optimizer):
    """ Adagrad Optimizer """
    
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        super().__init__(params, lr)
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        
        # 初始化状态
        for pg in self.param_groups:
            pg["sum_squares"] = []
            pg["steps"] = []
            for p in pg["params"]:
                # 使用initial_accumulator_value初始化
                accumulator = jt.full_like(p, self.initial_accumulator_value).stop_grad()
                pg["sum_squares"].append(accumulator)
                pg["steps"].append(jt.zeros(1).stop_grad())  # 步数计数器
    
    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            lr_decay = pg.get("lr_decay", self.lr_decay)
            
            for p, g, s, step_counter in zip(pg["params"], pg["grads"], pg["sum_squares"], pg["steps"]):
                if p.is_stop_grad():
                    continue
                
                # 更新步数计数
                step_counter.update(step_counter + 1)
                step = step_counter.item()
                
                # 权重衰减
                if weight_decay != 0:
                    g = g + weight_decay * p
                
                # 学习率衰减
                clr = lr / (1 + (step - 1) * lr_decay)
                
                # 累积梯度平方
                s.update(s + g * g)
                
                # 参数更新
                std = jt.sqrt(s) + eps
                p.update(p - clr * g / std)
                
        self.post_step()

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PD_FA = PD_FA(1, 10) # Bins from original paper, 10 thresholds
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        
        os.makedirs(os.path.join('result', self.save_dir), exist_ok=True)

        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = os.path.join(args.root, args.dataset)
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data - 移除 ImageNormalize 避免广播错误，在训练循环中手动归一化
        input_transform = transforms.Compose([
            transforms.ToTensor()])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = trainset.set_attrs(batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = testset.set_attrs(batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)
        self.testset = testset

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        model.apply(weights_init_kaiming)
        print("Model Initializing")
        self.model      = model

        # 优化器选择
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            # 完全匹配PyTorch Adagrad的默认参数
            self.optimizer = AdagradCustom(
                list(filter(lambda p: p.requires_grad, model.parameters())), 
                lr=args.lr, 
                lr_decay=0, 
                weight_decay=0, 
                initial_accumulator_value=0, 
                eps=1e-10
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()
        
        # Evaluation metrics
        self.best_iou       = 0
        
    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data, total=len(self.train_data) // self.args.train_batch_size)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            # 修复维度顺序：从 (batch, height, width, channels) 到 (batch, channels, height, width)
            if len(data.shape) == 4 and data.shape[-1] == 3:
                data = data.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            
            # 手动归一化 (ImageNet标准)
            mean = jt.array([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = jt.array([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            data = (data - mean) / std
            if self.args.deep_supervision == 'True':  # ✅ 修复：使用self.args
                preds= self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                loss /= len(preds)
                pred = preds[-1]  # ✅ 确保pred被定义
            else:
               pred = self.model(data)
               loss = SoftIoULoss(pred, labels)
            
            # 优化器更新 - 兼容自定义Adagrad
            if isinstance(self.optimizer, AdagradCustom):
                # 自定义Adagrad：直接传递loss
                self.optimizer.step(loss)
            else:
                # 原生Jittor优化器：使用标准API
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data, total=(len(self.test_data) + self.args.test_batch_size - 1) // self.args.test_batch_size)
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        losses = AverageMeter()

        with jt.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                # 修复维度顺序：从 (batch, height, width, channels) 到 (batch, channels, height, width)
                if len(data.shape) == 4 and data.shape[-1] == 3:
                    data = data.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                
                # 手动归一化 (ImageNet标准)
                mean = jt.array([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = jt.array([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                data = (data - mean) / std
                
                if self.args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                
                losses.update(loss.item(), pred.size(0))

                # --- Update epoch-level metrics ---
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
            test_loss=losses.avg
            
            Final_FA, Final_PD = self.PD_FA.get(img_num=len(self.testset))

            # --- Start: Continuous logging for every epoch ---
            try:
                current_lr = self.optimizer.lr
                continuous_log_path = os.path.join('result', self.save_dir, 'epoch_continuous_log.txt')
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                log_entry = (
                    f"Timestamp: {dt_string} | Epoch: {epoch:04d} | "
                    f"Train Loss: {self.train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                    f"mIoU: {mean_IOU:.4f} | PD: {Final_PD:.4f} | FA: {Final_FA:.8f} | "
                    f"LR: {current_lr:.6f}\n"
                )
                with open(continuous_log_path, 'a') as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"!! WARNING: Failed to write to continuous epoch log: {e}")
            # --- End: Continuous logging ---

            # Save the best model and its specific logs if it's a new record.
            if mean_IOU > self.best_iou:
                self.best_iou = mean_IOU
                print(f"New best mIoU: {self.best_iou:.4f}. Saving best model and logs...")
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

                # 1. Append to the history log of best scores
                history_log_path = os.path.join('result', self.save_dir, 'best_iou_history.log')
                try:
                    with open(history_log_path, 'a') as f:
                        f.write(f"Timestamp: {dt_string} | Epoch: {epoch:04d} | New Best mIoU: {self.best_iou:.4f}\n")
                except Exception as e:
                    print(f"!! WARNING: Failed to write to best IoU history log: {e}")

                # 2. Overwrite the log for the latest best metrics
                best_log_path = os.path.join('result', self.save_dir, 'best_metrics_latest.log')
                try:
                    with open(best_log_path, 'w') as f:
                        f.write(f"Last update: {dt_string} (Epoch: {epoch})\n")
                        f.write(f"Best mIoU: {self.best_iou:.4f}\n")
                        f.write(f"Train Loss: {self.train_loss:.4f}\n")
                        f.write(f"Test Loss: {test_loss:.4f}\n")
                        f.write('Recall: ' + ' '.join([f'{r:.4f}' for r in recall]) + '\n')
                        f.write('Precision: ' + ' '.join([f'{p:.4f}' for p in precision]) + '\n')
                        f.write(f'PD: {Final_PD:.4f}\n')
                        f.write(f'FA: {Final_FA:.8f}\n')
                except Exception as e:
                    print(f"!! WARNING: Failed to write best metrics log: {e}")

                # 3. Overwrite the best model checkpoint
                model_path = os.path.join('result', self.save_dir, 'mIoU_best.pth.tar')
                try:
                    save_checkpoint(self.model.state_dict(), model_path)
                    print(f"    --> Best model and logs saved for epoch {epoch}.")
                except Exception as e:
                    print(f"!! WARNING: Failed to save best model checkpoint: {e}")


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args) 