#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集自动准备脚本
自动解压和组织NUAA-SIRST和NUDT-SIRST数据集文件结构
"""

import os
import zipfile
import shutil
from pathlib import Path

def check_dataset_files():
    """检查数据集文件是否存在"""
    print("检查数据集文件...")
    
    expected_files = [
        "NUAA-SIRST.zip",
        "NUDT-SIRST.zip"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_name in expected_files:
        if os.path.exists(file_name):
            existing_files.append(file_name)
            print(f"   找到: {file_name}")
        else:
            missing_files.append(file_name)
            print(f"   缺失: {file_name}")
    
    return existing_files, missing_files

def create_dataset_structure():
    """创建数据集目录结构"""
    print("\n 创建数据集目录结构...")
    
    # 创建主数据集目录
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    print(f"   创建目录: {dataset_dir}")
    
    # 创建各数据集子目录
    datasets = ["NUAA-SIRST", "NUDT-SIRST"]
    
    for dataset_name in datasets:
        dataset_path = dataset_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        # 创建必要的子目录
        subdirs = ["images", "masks", "50_50", "value_result"]
        for subdir in subdirs:
            (dataset_path / subdir).mkdir(exist_ok=True)
        
        print(f"   创建数据集目录: {dataset_path}")

def extract_dataset(zip_file, dataset_name):
    """解压数据集文件"""
    print(f"\n📦 解压 {zip_file} ...")
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # 解压到临时目录
            temp_dir = f"temp_{dataset_name}"
            zip_ref.extractall(temp_dir)
            print(f"   解压完成: {zip_file}")
            
            # 查找解压后的实际目录结构
            temp_path = Path(temp_dir)
            extracted_dirs = list(temp_path.iterdir())
            
            if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
                # 如果解压后只有一个目录，进入该目录
                source_dir = extracted_dirs[0]
            else:
                # 如果解压后是多个文件/目录，使用temp_dir作为源目录
                source_dir = temp_path
            
            return source_dir, temp_dir
            
    except zipfile.BadZipFile:
        print(f"   错误: {zip_file} 不是有效的ZIP文件")
        return None, None
    except Exception as e:
        print(f"   解压失败: {e}")
        return None, None

def organize_dataset_files(source_dir, dataset_name):
    """整理数据集文件到标准结构"""
    print(f"\n🔧 整理 {dataset_name} 数据集文件...")
    
    source_path = Path(source_dir)
    target_path = Path("dataset") / dataset_name
    
    # 查找images和masks目录
    images_source = None
    masks_source = None
    split_source = None
    
    # 递归查找images和masks目录
    for path in source_path.rglob("*"):
        if path.is_dir():
            if path.name.lower() == "images":
                images_source = path
            elif path.name.lower() == "masks":
                masks_source = path
            elif path.name in ["50_50", "split"] or "split" in path.name.lower():
                split_source = path
    
    # 复制images目录
    if images_source and images_source.exists():
        target_images = target_path / "images"
        if target_images.exists():
            shutil.rmtree(target_images)
        shutil.copytree(images_source, target_images)
        print(f"   复制images目录: {len(list(target_images.glob('*')))} 个文件")
    else:
        print(f"   未找到images目录")
    
    # 复制masks目录  
    if masks_source and masks_source.exists():
        target_masks = target_path / "masks"
        if target_masks.exists():
            shutil.rmtree(target_masks)
        shutil.copytree(masks_source, target_masks)
        print(f"   复制masks目录: {len(list(target_masks.glob('*')))} 个文件")
    else:
        print(f"   未找到masks目录")
    
    # 复制数据集划分文件
    if split_source and split_source.exists():
        target_split = target_path / "50_50"
        if target_split.exists():
            shutil.rmtree(target_split)
        shutil.copytree(split_source, target_split)
        print(f"   复制数据集划分文件")
    else:
        # 查找train.txt和test.txt文件
        train_txt = None
        test_txt = None
        for path in source_path.rglob("*.txt"):
            if "train" in path.name.lower():
                train_txt = path
            elif "test" in path.name.lower():
                test_txt = path
        
        if train_txt and test_txt:
            target_split = target_path / "50_50"
            target_split.mkdir(exist_ok=True)
            shutil.copy2(train_txt, target_split / "train.txt")
            shutil.copy2(test_txt, target_split / "test.txt")
            print(f"   复制数据集划分文件")
        else:
            print(f"    未找到数据集划分文件 (train.txt, test.txt)")

def cleanup_temp_files(temp_dirs):
    """清理临时文件"""
    print("\n🧹 清理临时文件...")
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"   删除临时目录: {temp_dir}")

def verify_dataset_structure():
    """验证数据集结构是否正确"""
    print("\n 验证数据集结构...")
    
    datasets = ["NUAA-SIRST", "NUDT-SIRST"]
    all_good = True
    
    for dataset_name in datasets:
        print(f"\n  检查 {dataset_name}:")
        dataset_path = Path("dataset") / dataset_name
        
        # 检查必要目录
        required_dirs = ["images", "masks", "50_50", "value_result"]
        for dirname in required_dirs:
            dir_path = dataset_path / dirname
            if dir_path.exists():
                if dirname == "images":
                    file_count = len(list(dir_path.glob("*.png")))
                    print(f"     {dirname}/ ({file_count} 张图像)")
                elif dirname == "masks":
                    file_count = len(list(dir_path.glob("*.png")))
                    print(f"     {dirname}/ ({file_count} 个标签)")
                elif dirname == "50_50":
                    train_txt = dir_path / "train.txt"
                    test_txt = dir_path / "test.txt"
                    if train_txt.exists() and test_txt.exists():
                        print(f"     {dirname}/ (train.txt, test.txt)")
                    else:
                        print(f"     {dirname}/ (缺少train.txt或test.txt)")
                        all_good = False
                else:
                    print(f"     {dirname}/")
            else:
                print(f"     {dirname}/ (目录不存在)")
                all_good = False
    
    return all_good

def print_download_instructions():
    """打印下载说明"""
    print("\n📥 数据集下载说明:")
    print("=" * 50)
    print("请手动下载以下数据集并保存到项目根目录:")
    print("📁 NUDT-SIRST数据集:")
    print("  链接: https://pan.quark.cn/s/c87c1148de39?pwd=AQtj")
    print("  提取码: AQtj")
    print()
    print("📁 NUAA-SIRST数据集:")
    print("  链接: https://pan.quark.cn/s/55066db3363d?pwd=DVb1")  
    print("  提取码: DVb1")
    print()
    print("下载完成后，重新运行此脚本进行自动解压和整理。")

def main():
    """主函数"""
    
    # 检查数据集文件
    existing_files, missing_files = check_dataset_files()
    
    if missing_files:
        print_download_instructions()
        return
    
    # 创建目录结构
    create_dataset_structure()
    
    # 处理每个数据集
    temp_dirs = []
    
    for zip_file in existing_files:
        dataset_name = zip_file.replace(".zip", "")
        
        # 解压数据集
        source_dir, temp_dir = extract_dataset(zip_file, dataset_name)
        if source_dir is None:
            continue
            
        temp_dirs.append(temp_dir)
        
        # 整理文件结构
        organize_dataset_files(source_dir, dataset_name)
    
    # 清理临时文件
    cleanup_temp_files(temp_dirs)
    
    # 验证结构
    success = verify_dataset_structure()
    
    # 最终提示
    if success:
        print("数据集准备完成！")
    else:
        print("据集准备过程中发现问题，请检查上述错误信息。")

if __name__ == "__main__":
    main() 
