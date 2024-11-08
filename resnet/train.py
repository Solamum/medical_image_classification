import json
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 超参数
    batch_size = 16
    epochs = 30
    lr = 0.0001

    # 数据加载和预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data")  # data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "2-MedImage-TrainSet"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # 获取类别标签
    # {'disease':0, 'normal':1}
    classes_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "2-MedImage-TestSet"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print("using {} images for training, {} images for validation.\n".format(train_num, val_num))

    # 载入预训练模型参数
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)  # change output layer to 2 classes
    net.to(device)

    # 定义损失函数和优化器
    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    # 记录每个epoch的指标，用于画图
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    val_auc_list = []
    combined_metric_list = []

    # 开始训练
    best_metric = 0.0  # 最佳综合指标，这里取 Accuracy, F1-Score, AUC 的加权平均值
    saved_acc = 0.0  # 保存的最佳模型的准确率
    save_path = '../model/temp/resNet34.pth'
    train_steps = len(train_loader)

    # 早停法：当连续 patience 个 epoch 验证集指标没有提升时，提前停止训练
    patience = 10  # 根据需要调整
    counter = 0
    early_stop = False
    stop_epoch = 0

    print(f'开始训练，总 Epochs {epochs}, Batch Size {batch_size}, Learning Rate {lr}\n')
    start = time.time()
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # 记录训练损失
            running_loss += loss.item()

            # 计算训练过程中的准确率
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 平均训练损失
        train_loss = running_loss / train_steps
        train_loss_list.append(train_loss)
        # 训练过程中的准确率：预测正确的样本数 / 总样本数
        train_accuracy = train_acc / train_num
        train_acc_list.append(train_accuracy)

        # validate
        net.eval()
        all_preds = []  # 所有预测值
        all_labels = []  # 实际标签
        all_probs = []  # 所有预测概率值
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]

                # 累积所有预测值和标签
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                # 收集概率值以用于 AUC 计算
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        # 计算 Accuracy, Precision, Recall, F1-Score, AUC
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        # 记录指标
        val_acc_list.append(accuracy)
        val_precision_list.append(precision)
        val_recall_list.append(recall)
        val_f1_list.append(f1)
        val_auc_list.append(auc)

        # 综合考虑 Accuracy, F1, AUC 的加权平均值
        combined_metric = 0.6 * accuracy + 0.2 * f1 + 0.2 * auc  # 可以调整权重
        combined_metric_list.append(combined_metric)

        print("| {:^6} | {:^10} | {:^17} | {:^19} | {:^9} | {:^6} | {:^8} | {:^6} | {:^15} |".format(
            "Epoch", "Train Loss", "Training Accuracy", "Validation Accuracy", "Precision", "Recall", "F1-Score",
            "AUC", "Combined Metric"))
        print(
            "| {:^6} | {:^10.3f} | {:^17.3f} | {:^19.3f} | {:^9.3f} | {:^6.3f} | {:^8.3f} | {:^6.3f} | {:^15.3f} |\n".format(
                epoch + 1, train_loss, train_accuracy, accuracy, precision, recall, f1, auc, combined_metric))

        # 保存最佳模型
        if combined_metric > best_metric:
            best_metric = combined_metric
            saved_acc = accuracy
            torch.save(net.state_dict(), save_path)
            print(f'Save model, Epoch {epoch + 1}, Combined Metric {combined_metric:.3f}, Accuracy {accuracy:.3f}\n')
            counter = 0

            # 绘制 ROC 曲线
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC Curve (area = {auc:.3f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join('../metrics_result/temp', 'ROC Curve.png'))
            plt.show()

        else:
            counter += 1
            if counter >= patience:
                early_stop = True
                stop_epoch = epoch + 1
                break

    end = time.time()
    minutes = (end - start) / 60
    if early_stop:
        print(
            f'早停法：训练提前停止，Stop Epoch {stop_epoch + 1}，用时 {minutes:.3f} 分钟，Best Combined Metric {best_metric:.3f}, Accuracy {saved_acc:.3f}\n')
    else:
        stop_epoch = epochs
        print(
            f'模型训练结束，用时 {minutes:.3f} 分钟，Best Combined Metric {best_metric:.3f}, Accuracy {saved_acc:.3f}\n')

    # 绘制损失和准确率曲线
    plt.figure()
    plt.plot(range(1, stop_epoch + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, stop_epoch + 1), val_acc_list, label='Validation Accuracy')
    plt.plot(range(1, stop_epoch + 1), train_acc_list, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Loss and Accuracy')
    plt.legend()
    plt.savefig(os.path.join('../metrics_result/temp', 'Loss and Accuracy.png'))
    plt.show()
    # 绘制曲线
    plot(val_precision_list, stop_epoch, 'Precision')
    plot(val_recall_list, stop_epoch, 'Recall')
    plot(val_f1_list, stop_epoch, 'F1-Score')
    plot(val_auc_list, stop_epoch, 'AUC')
    plot(combined_metric_list, stop_epoch, 'Combined Metric')


def plot(plt_list, epochs, label, path='../metrics_result/temp'):
    # 绘制曲线
    plt.figure()
    plt.plot(range(1, epochs + 1), plt_list)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.ylim(0, 1)
    plt.title(label)
    # plt.legend()
    plt.savefig(os.path.join(path, label + '.png'))
    plt.show()


if __name__ == '__main__':
    main()
