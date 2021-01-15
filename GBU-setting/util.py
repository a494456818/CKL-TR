# import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import pandas as pd
import random
from copy import deepcopy


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()]
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()]
        val_unseen_loc = fid['val_unseen_loc'][()]
        test_seen_loc = fid['test_seen_loc'][()]
        test_unseen_loc = fid['test_unseen_loc'][()]
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc]
            self.train_label = label[trainval_loc]
            self.test_unseen_feature = feature[test_unseen_loc]
            self.test_unseen_label = label[test_unseen_loc]
            self.test_seen_feature = feature[test_seen_loc]
            self.test_seen_label = label[test_seen_loc]
        else:
            self.train_feature = feature[train_loc]
            self.train_label = label[train_loc]
            self.test_unseen_feature = feature[val_unseen_loc]
            self.test_unseen_label = label[val_unseen_loc]

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        if opt.dataset == "FLO":
            trainval_loc = matcontent['train_loc'].squeeze() - 1
        else:
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.attribute_unseen = self.attribute[self.unseenclasses]
        self.attribute_seen = self.attribute[self.seenclasses]
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

        self.all_feature = torch.cat((self.train_feature, self.test_unseen_feature, self.test_seen_feature))
        self.both_feature = torch.cat((self.train_feature, self.test_unseen_feature))

        self.tr_cls_centroid = {}
        for i in range(self.ntrain_class):
            self.tr_cls_centroid[i] = torch.mean(self.train_feature[self.train_mapped_label == i], dim=0)

        # 增加科属种，只在训练集中加噢
        biologicalTaxonomyMatrix = pd.read_csv("./data/" + opt.dataset + ".csv", encoding="gbk")  # AWA2.csv表格
        family = biologicalTaxonomyMatrix["Family"].values
        family = list(set(family))  # 加载科，为一个字典
        family = clearNanInList(family)

        genus = biologicalTaxonomyMatrix["Genus"].values  # 加载属
        genus = list(set(genus))
        genus = clearNanInList(genus)

        familyDict = {}  # 某个科中所属的全部标签，标签从0开始
        genusDict = {}  # 某个属中所属的全部标签，标签从0开始

        for label in np.unique(self.train_class):
            f = biologicalTaxonomyMatrix.loc[label, "Family"]  # 科
            g = biologicalTaxonomyMatrix.loc[label, "Genus"]  # 属
            # 把 train_label 映射到对应的科中，标签为i
            if f in family:
                try:
                    familyDict[f].append(label)
                except:
                    familyDict[f] = [label]
            # 把 train_label 映射到对应的属中，标签为i
            if g in genus:
                try:
                    genusDict[g].append(label)
                except:
                    genusDict[g] = [label]

        # 删除数量为1的科和属, key:名称 value:对应的标签
        familyDict = deleteOne(familyDict)
        genusDict = deleteOne(genusDict)

        self.familyToText = {}  # key：科标签801开始，value：文本特征，一个列表
        self.genusToText = {}  # value：属标签901开始，value：文本特征，一个列表
        self.familyLabelToBirdLabel = {}  # key: 科的标签，value：科中所有鸟的标签
        self.genusLabelToBirdLabel = {}  # key: 属的标签，value：科中所有鸟的标签

        self.train_origin_label = torch.from_numpy(self.train_label.numpy().copy())  # 真实的标签，科属中的鸟类也是对应的其真实类表

        # 加载科的数据, 标签从801开始
        familyLabelStart = 801
        for key in familyDict.keys():
            for labelBelontoF in familyDict[key]:
                textLabels = []
                birdLabels = []
                if labelBelontoF not in textLabels:
                    try:
                        self.familyToText[familyLabelStart].append(
                            self.attribute[labelBelontoF])  # 标签为labelBelontoF的添加文本
                    except:
                        self.familyToText[familyLabelStart] = [self.attribute[labelBelontoF]]
                    textLabels.append(labelBelontoF)

                if labelBelontoF not in birdLabels:
                    try:
                        self.familyLabelToBirdLabel[familyLabelStart].append(labelBelontoF)  # 标签为labelBelontoF的添加文本
                    except:
                        self.familyLabelToBirdLabel[familyLabelStart] = [labelBelontoF]
                    birdLabels.append(labelBelontoF)
            familyLabelStart += 1

        # 加载属的数据, 标签从901开始
        genusLabelStart = 901
        for key in genusDict.keys():
            for labelBelontoG in genusDict[key]:
                textLabels = []
                birdLabels = []
                if labelBelontoG not in textLabels:
                    try:
                        self.genusToText[genusLabelStart].append(
                            self.attribute[labelBelontoG])  # 标签为labelBelontoF的添加文本
                    except:
                        self.genusToText[genusLabelStart] = [self.attribute[labelBelontoG]]
                    textLabels.append(labelBelontoG)

                if labelBelontoG not in birdLabels:
                    try:
                        self.genusLabelToBirdLabel[genusLabelStart].append(labelBelontoG)  # 标签为labelBelontoF的添加文本
                    except:
                        self.genusLabelToBirdLabel[genusLabelStart] = [labelBelontoG]
                    birdLabels.append(labelBelontoG)
            genusLabelStart += 1

        self.birdLabelToGenusLabel = {l: key for key in self.genusLabelToBirdLabel.keys() for l in
                                      self.genusLabelToBirdLabel[key]}
        self.birdLabelToFamilyLabel = {l: key for key in self.familyLabelToBirdLabel.keys() for l in
                                       self.familyLabelToBirdLabel[key]}

        # 计算所有类别的中心损失(原始类别，科和属的)

        # 科和属
        for i in range(801, familyLabelStart):
            try:
                self.tr_cls_centroid[i] = np.mean(self.train_feature[np.array(
                    [int(label.numpy()) in self.familyLabelToBirdLabel[i] for label in self.train_label])].numpy(),
                                                  axis=0)
            except:
                self.tr_cls_centroid[i] = np.mean(self.train_feature.numpy()[np.array(
                    [int(label.numpy()) in self.familyLabelToBirdLabel[i] for label in self.train_label])].numpy(),
                                                  axis=0)

        for i in range(901, genusLabelStart):
            try:
                self.tr_cls_centroid[i] = np.mean(self.train_feature[np.array(
                    [int(label.numpy()) in self.genusLabelToBirdLabel[i] for label in self.train_label])].numpy(),
                                                  axis=0)
            except:
                self.tr_cls_centroid[i] = np.mean(self.train_feature.numpy()[np.array(
                    [int(label.numpy()) in self.genusLabelToBirdLabel[i] for label in self.train_label])].numpy(),
                                                  axis=0)

        self.familyLabelEnd = familyLabelStart
        self.familyLabelStart = 801

        self.genusLabelEnd = genusLabelStart
        self.genusLabelStart = 901

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # family-genus-species样本
    def next_batch_3level(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        # 属和科的
        batch_label_genus = torch.from_numpy(np.array([(self.birdLabelToGenusLabel[
                                                            int(label.numpy())] if label in list(
            self.birdLabelToGenusLabel.keys()) else -1) for label in batch_label]))
        batch_label_family = torch.from_numpy(np.array([(self.birdLabelToFamilyLabel[
                                                             int(label.numpy())] if label in list(
            self.birdLabelToFamilyLabel.keys()) else -1) for label in batch_label]))
        # 语义特征
        batch_att_genus = deepcopy(batch_att)
        batch_att_family = deepcopy(batch_att)
        for i, genus_label in enumerate(batch_label_genus):
            if genus_label != -1:
                batch_att_genus[i] = random.sample(self.genusToText[int(genus_label)], 1)[0]
        for i, family_label in enumerate(batch_label_family):
            if family_label != -1:
                batch_att_family[i] = random.sample(self.familyToText[int(family_label)], 1)[0]

        return batch_feature, batch_label, batch_att, batch_label_genus, batch_att_genus, batch_label_family, batch_att_family

    def next_batch_transductive(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_seen_feature = self.train_feature[idx]
        batch_seen_label = self.train_label[idx]
        batch_seen_att = self.attribute[batch_seen_label]

        idx = torch.randperm(self.all_feature.shape[0])[0:batch_size]
        batch_both_feature = self.all_feature[idx]
        idx_both_att = torch.randint(0, self.attribute.shape[0], (batch_size,))
        batch_both_att = self.attribute[idx_both_att]

        return batch_seen_feature, batch_seen_label, batch_seen_att, batch_both_feature, batch_both_att

    def next_batch_transductive_both(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_seen_feature = self.train_feature[idx]
        batch_seen_label = self.train_label[idx]
        batch_seen_att = self.attribute[batch_seen_label]

        idx = torch.randperm(self.both_feature.shape[0])[0:batch_size]
        batch_both_feature = self.both_feature[idx]
        idx_both_att = torch.randint(0, self.attribute.shape[0], (batch_size,))
        batch_both_att = self.attribute[idx_both_att]

        return batch_seen_feature, batch_seen_label, batch_seen_att, batch_both_feature, batch_both_att

    def next_batch_MMD(self, batch_size):
        # idx = torch.randperm(self.ntrain)[0:batch_size]
        index = torch.randint(self.seenclasses.shape[0], (2,))
        while index[0] == index[1]:
            index = torch.randint(self.seenclasses.shape[0], (2,))
        select_labels = self.seenclasses[index]
        X_features = self.train_feature[self.train_label == select_labels[0]]
        Y_features = self.train_feature[self.train_label == select_labels[1]]

        idx_X = torch.randperm(X_features.shape[0])[0:batch_size]
        X_features = X_features[idx_X]

        idx_Y = torch.randperm(Y_features.shape[0])[0:batch_size]
        Y_features = Y_features[idx_Y]

        return X_features, Y_features

    def next_batch_MMD_all(self):
        # idx = torch.randperm(self.ntrain)[0:batch_size]
        index = torch.randint(self.seenclasses.shape[0], (2,))
        while index[0] == index[1]:
            index = torch.randint(self.seenclasses.shape[0], (2,))
        select_labels = self.seenclasses[index]
        X_features = self.train_feature[self.train_label == select_labels[0]]
        Y_features = self.train_feature[self.train_label == select_labels[1]]

        return X_features, Y_features

    def next_batch_unseenatt(self, batch_size, unseen_batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        # idx = torch.randperm(data)[0:batch_size]
        idx_unseen = torch.randint(0, self.unseenclasses.shape[0], (unseen_batch_size,))
        unseen_label = self.unseenclasses[idx_unseen]
        unseen_att = self.attribute[unseen_label]

        return batch_feature, batch_label, batch_att, unseen_label, unseen_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att


def clearNanInList(list):
    temp = []
    for l in list:
        if l is not np.nan:
            temp.append(l)
    return temp


# 删除list为1的数据, dict: key: str; values: list
def deleteOne(dict):
    temp = {}
    for key in dict.keys():
        if len(dict[key]) > 1:
            temp[key] = dict[key]
    return temp