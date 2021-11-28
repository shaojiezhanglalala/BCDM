from __future__ import print_function
import argparse
import torch.optim as optim
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
from torchvision import transforms, datasets
import torch.nn.functional as F
import os
import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='save/mcd', metavar='B',
                    help='board dir')
parser.add_argument('--train_path', type=str, default=r'D:\PythonWorkSpace\pythonProject\BCDM-master\classification\dataset\train', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default=r'D:\PythonWorkSpace\pythonProject\BCDM-master\classification\dataset\validation', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--resnet', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.cuda.set_device(args.gpu)
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
save_path = args.save + '_' + str(args.num_k)

data_transforms = {
    train_path: transforms.Compose([
        # 图像变换：先改变形状、再依概率水平翻转、最后中心裁剪大小为224的正方形。
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        # 将图像转为tensor格式
        transforms.ToTensor(),
        # 将图像标准化
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        # 同上
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# 根据训练集和验证集路径加载数据
'''
datasets.ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字。
ImageFolder(root,transform=None,target_transform=None,loader=default_loader)
root : 在指定的root路径下面寻找图片
transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象
target_transform :对label进行变换
loader: 指定加载图片的函数，默认操作是读取PIL image对象
'''
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
dset_classes= ['aeroplane',
 'bicycle',
 'bus',
 'car',
 'horse',
 'knife',
 'motorcycle',
 'person',
 'plant',
 'skateboard',
 'train',
 'truck']
classes_acc = {}
for i in dset_classes:
    classes_acc[i] = []
    classes_acc[i].append(0)
    classes_acc[i].append(0)
print('classes' + str(dset_classes))
print('lr', args.lr)
#   判断GPU是否可用
use_gpu = torch.cuda.is_available()
# 设置随机种子，用于保存随机初始化状态，使每次初始化都是固定的。
torch.manual_seed(args.seed)
#   若GPU可用，则为GPU设置随机种子
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# 开始加载训练集数据，train_path为源域数据，val_path为目标域数据
train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=True, drop_last=True)
#   经过train_loader.load_data()处理，返回'S': S, 'S_label': S_paths,'T': T, 'T_label': T_paths
dataset = train_loader.load_data()
# 加载测试集数据，细节同上
test_loader = CVDataLoader()
opt = args
test_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=True, drop_last=False)
dataset_test = test_loader.load_data()
# 选择生成器的神经网络结构，实验使用的是resnet 101
option = 'resnet' + args.resnet
# 特征生成器
G = ResBottle(option)
# 两个分类器，ResClassifier为作者自己设计的分类器，总共四层网络，各层单元数分别为G.output_num()、1000、1000、12
F1 = ResClassifier(num_classes=12, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
F2 = ResClassifier(num_classes=12, num_layer=num_layer, num_unit=G.output_num(), middle=1000)

D = AdversarialNetwork(in_feature=G.output_num(), hidden_size=1024)
# 初始化两个分类器的权重
F1.apply(weights_init)
F2.apply(weights_init)
# 学习率
lr = args.lr
# 如果使用GPU，将神经网络转化到GPU上
if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()
    D.cuda()
# 选择优化方式 momentum：使用momentumSGD，adam：使用adam优化方式，否则使用Adadelta
# weight_decay为权重衰减率，防止模型过拟合
if args.optimizer == 'momentum':
    optimizer_g = optim.SGD(list(G.features.parameters()), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                           weight_decay=0.0005)
    optimizer_d = optim.SGD(list(D.parameters()), lr=args.lr, weight_decay=0.0005, momentum=0.9)
elif args.optimizer == 'adam':
    optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
    optimizer_d = optim.Adam(list(D.parameters()), lr=args.lr, weight_decay=0.0005)
else:
    optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
    optimizer_d = optim.Adadelta(D.parameters(), lr=args.lr, weight_decay=0.0005)


# 定义训练过程
def train(num_epoch):
    # 使用交叉熵损失函数。
    criterion = nn.CrossEntropyLoss().cuda()
    # 记录训练过程中最佳准确率
    best_acc=0
    for ep in range(num_epoch):
        since = time.time()
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    #     开始按批训练
        for batch_idx, data in enumerate(dataset):
            G.train()
            F1.train()
            F2.train()
            D.train()
            # 根据迭代次数，衰减学习率
            adjust_learning_rate(optimizer_f, ep, batch_idx, 4762, 0.001)
            data_s = data['S']
            label_s = data['S_label']
            data_t = data['T']
            label_t = data['T_label']

            if dataset.stop_S:
                break
            # 若使用GPU，进行转换
            if args.cuda:
                data_s, label_s = data_s.cuda(), label_s.cuda()
                data_t, label_t = data_t.cuda(), label_t.cuda()
            # 权衡超参数：α，用于权衡Lcss和Lcdd
            eta=0.01
            data_all = Variable(torch.cat((data_s, data_t), 0))
            label_s = Variable(label_s)
            bs = len(label_s)
            s_domain_labels = torch.ones(bs, dtype=torch.long).cuda()
            t_domain_labels = torch.zeros(bs, dtype=torch.long).cuda()
            domain_labels = Variable(torch.cat((s_domain_labels, t_domain_labels), 0))

            """source domain discriminative"""
            # 训练两个分类器，使其对于源域数据有较好的区分能力
            # Step A train all networks to minimize loss on source
            # 初始化梯度
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            # 经过特征生成器g,产生的源域数据和目标域数据的特征
            features_source=output[:bs, :]
            features_target=output[bs:, :]
            # 经过两个分类器F1,F2，得到输出
            output1 = F1(output)
            output2 = F2(output)
            # 源域数据的分类器输出
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            # 目标域数据的分类器输出
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1 = F.softmax(output_t1)
            output_t2 = F.softmax(output_t2)
            # 计算分类损失
            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            all_loss = loss1 + loss2 + 0.01*entropy_loss
            # 反向传播
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            """target domain discriminative"""
            # Step B train classifier to maximize discrepancy
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            # 生成特征
            output = G(data_all)
            # 两个分类器的输出
            output1 = F1(output)
            output2 = F2(output)
            # 源域数据的分类器输出
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            # 目标域数据的分类器输出
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1 = F.softmax(output_t1)
            output_t2 = F.softmax(output_t2)
            # 计算损失
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            # 使用了论文的CDD度量
            loss_dis = cdd(output_t1,output_t2)

            F_loss = loss1 + loss2 - eta * loss_dis+0.01*entropy_loss
            F_loss.backward()
            optimizer_f.step()

            # Step C train genrator to minimize discrepancy
            for i in range(num_k):
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                output = G(data_all)
                features_source = output[:bs, :]
                features_target = output[bs:, :]
                output1 = F1(output)
                output2 = F2(output)
                output_s1 = output1[:bs, :]
                output_s2 = output2[:bs, :]
                output_t1 = output1[bs:, :]
                output_t2 = output2[bs:, :]
                output_t1 = F.softmax(output_t1)
                output_t2 = F.softmax(output_t2)

                entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
                loss_dis = cdd(output_t1,output_t2)
                D_loss = eta*loss_dis+0.01*entropy_loss

                D_loss.backward()
                optimizer_g.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f} '.format(
                        ep, batch_idx, len(dataset.data_loader_S), 100. * batch_idx / len(dataset.data_loader_S),
                        loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))

        # test
        temp_acc=test(ep + 1)
        if temp_acc > best_acc:
            best_acc = temp_acc
        print('\tbest:', best_acc)
        print('time:', time.time() - since)
        print('-' * 100)


def test(epoch):
    # 将模型转换为评估模式
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    print('-' * 100, '\nTesting')
    for batch_idx, data in enumerate(dataset_test):
        if dataset_test.stop_T:
            break
        if args.cuda:
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.cuda()
        img, label = Variable(img, volatile=True), Variable(label)
        output = G(img)
        output1 = F1(output)
        output2 = F2(output)
        test_loss += F.nll_loss(output1, label).item()
        output_add = output1 + output2
        pred = output_add.data.max(1)[1]
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]
        for i in range(len(label)):
            key_label = dset_classes[label.long()[i].item()]
            key_pred = dset_classes[pred.long()[i].item()]
            # 预测总次数
            classes_acc[key_label][1] += 1
            if key_pred == key_label:
                # 预测正确的次数
                classes_acc[key_pred][0] += 1

    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))
    avg = []
    for i in dset_classes:
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    temp_acc=np.average(avg)
    print('\taverage:', temp_acc)
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0
    return temp_acc


if __name__ == '__main__':
    train(args.epochs + 1)

