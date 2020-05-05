import numpy as np
import scipy.io as sio
from termcolor import cprint
import pickle
import sys
import pandas as pd
import random

class LoadDataset(object):

    def __init__(self, opt):
        txt_feat_path = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train.pkl'
            pfc_label_path_test = 'data/CUB2011/labels_test.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train.mat'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test.mat'
            train_cls_num = 150
            test_cls_num = 50
        else:
            train_test_split_dir = 'data/CUB2011/train_test_split_hard.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train_hard.pkl'
            pfc_label_path_test = 'data/CUB2011/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test_hard.mat'
            train_cls_num = 160
            test_cls_num = 40

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        cprint("pfc_feat_file: {} || {} ".format(pfc_feat_path_train, pfc_feat_path_test), 'red')

        self.train_cls_num = train_cls_num
        self.test_cls_num  = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]
        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test  = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test  = pickle.load(fout2)

        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir)
        self.text_dim = self.train_text_feature.shape[1]  # 7551

        train_test_split = sio.loadmat(train_test_split_dir)
        biologicalTaxonomyMatrix = pd.read_csv("./data/cub.csv", index_col="id", encoding="gbk")
        family = biologicalTaxonomyMatrix["Family"].values
        family = list(set(family))
        family = clearNanInList(family)

        genus = biologicalTaxonomyMatrix["Genus"].values
        genus = list(set(genus))
        genus = clearNanInList(genus)

        familyDict = {}
        genusDict = {}

        for (i, train_label) in enumerate(train_test_split["train_cid"].squeeze()):
            f = biologicalTaxonomyMatrix.loc[train_label, "Family"]
            g = biologicalTaxonomyMatrix.loc[train_label, "Genus"]
            
            if f in family:
                try:
                    familyDict[f].append(i)
                except:
                    familyDict[f] = [i]
            
            if g in genus:
                try:
                    genusDict[g].append(i)
                except:
                    genusDict[g] = [i]

        
        familyDict = deleteOne(familyDict)
        genusDict = deleteOne(genusDict)

        self.familyToText = {}
        self.genusToText = {}
        self.familyLabelToBirdLabel = {}
        self.genusLabelToBirdLabel = {}

        self.labels_origin_train = self.labels_train.copy()
        
        familyLabelStart = 201
        for key in familyDict.keys():
            for labelBelontoF in familyDict[key]:
                textLabels = []
                birdLabels = []
                n = len(familyDict[key])
                selectedVisualFeatures = self.pfc_feat_data_train[self.labels_train == labelBelontoF]
                tempVisualFeatures = selectedVisualFeatures[
                    random.sample([i for i in range(selectedVisualFeatures.shape[0])]
                                  , selectedVisualFeatures.shape[0] // n + 1)]

                self.pfc_feat_data_train = np.r_[self.pfc_feat_data_train, tempVisualFeatures]
                self.labels_origin_train = np.r_[self.labels_origin_train, [labelBelontoF]*tempVisualFeatures.shape[0]]
                self.labels_train = np.r_[self.labels_train, [familyLabelStart]*tempVisualFeatures.shape[0]]
                if labelBelontoF not in textLabels:
                    try:
                        self.familyToText[familyLabelStart].append(self.train_text_feature[labelBelontoF])
                    except:
                        self.familyToText[familyLabelStart] = [self.train_text_feature[labelBelontoF]]
                    textLabels.append(labelBelontoF)

                if labelBelontoF not in birdLabels:
                    try:
                        self.familyLabelToBirdLabel[familyLabelStart].append(labelBelontoF)
                    except:
                        self.familyLabelToBirdLabel[familyLabelStart] = [labelBelontoF]
                    birdLabels.append(labelBelontoF)
            familyLabelStart += 1

        genusLabelStart = 301
        for key in genusDict.keys():
            for labelBelontoG in genusDict[key]:
                textLabels = []
                birdLabels = []
                tempVisualFeatures = self.pfc_feat_data_train[self.labels_train == labelBelontoG]
                self.pfc_feat_data_train = np.r_[self.pfc_feat_data_train, tempVisualFeatures]
                self.labels_origin_train = np.r_[self.labels_origin_train, [labelBelontoG]*tempVisualFeatures.shape[0]]
                self.labels_train = np.r_[self.labels_train, [genusLabelStart]*tempVisualFeatures.shape[0]]
                if labelBelontoG not in textLabels:
                    try:
                        self.genusToText[genusLabelStart].append(self.train_text_feature[labelBelontoG])
                    except:
                        self.genusToText[genusLabelStart] = [self.train_text_feature[labelBelontoG]]
                    textLabels.append(labelBelontoG)

                if labelBelontoG not in birdLabels:
                    try:
                        self.genusLabelToBirdLabel[genusLabelStart].append(labelBelontoG)
                    except:
                        self.genusLabelToBirdLabel[genusLabelStart] = [labelBelontoG]
                    birdLabels.append(labelBelontoG)
            genusLabelStart += 1

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        self.tr_cls_centroid = {}
        for i in range(train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)

        for i in range(201, familyLabelStart):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)
        for i in range(301, genusLabelStart):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)
        self.familyLabelEnd = familyLabelStart
        self.familyLabelStart = 201

        self.genusLabelEnd = genusLabelStart
        self.genusLabelStart = 301

def deleteOne(dict):
    temp = {}
    for key in dict.keys():
        if len(dict[key]) > 1:
            temp[key] = dict[key]
    return temp

def clearNanInList(list):
    temp = []
    for l in list:
        if l is not np.nan:
            temp.append(l)
    return temp

class LoadDataset_NAB(object):
    def __init__(self, opt):
        txt_feat_path = 'data/NABird/NAB_Porter_13217D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = 'data/NABird/train_test_split_NABird_easy.mat'
            pfc_label_path_train = 'data/NABird/labels_train.pkl'
            pfc_label_path_test = 'data/NABird/labels_test.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_easy.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_easy.mat'
            train_cls_num = 323
            test_cls_num = 81
        else:
            train_test_split_dir = 'data/NABird/train_test_split_NABird_hard.mat'
            pfc_label_path_train = 'data/NABird/labels_train_hard.pkl'
            pfc_label_path_test = 'data/NABird/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_hard.mat'
            train_cls_num = 323
            test_cls_num = 81

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        cprint("pfc_feat_file: {} || {} ".format(pfc_feat_path_train, pfc_feat_path_test), 'red')

        self.train_cls_num = train_cls_num
        self.test_cls_num  = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]
        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test  = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test  = pickle.load(fout2)

        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir)
        self.text_dim = self.train_text_feature.shape[1] # 13, 217

        train_test_split = sio.loadmat(train_test_split_dir)
        biologicalTaxonomyMatrix = pd.read_csv("./data/nab.csv", index_col="id", encoding="gbk")
        family = biologicalTaxonomyMatrix["Family"].values
        family = list(set(family))
        family = clearNanInList(family)

        genus = biologicalTaxonomyMatrix["Genus"].values
        genus = list(set(genus))
        genus = clearNanInList(genus)

        familyDict = {}
        genusDict = {}

        for (i, train_label) in enumerate(train_test_split["train_cid"].squeeze()):
            f = biologicalTaxonomyMatrix.loc[train_label, "Family"]
            g = biologicalTaxonomyMatrix.loc[train_label, "Genus"]

            if f in family:
                try:
                    familyDict[f].append(i)
                except:
                    familyDict[f] = [i]

            if g in genus:
                try:
                    genusDict[g].append(i)
                except:
                    genusDict[g] = [i]


        familyDict = deleteOne(familyDict)
        genusDict = deleteOne(genusDict)

        self.familyToText = {}
        self.genusToText = {}
        self.familyLabelToBirdLabel = {}
        self.genusLabelToBirdLabel = {}

        self.labels_origin_train = self.labels_train.copy()
       
        familyLabelStart = 601
        for key in familyDict.keys():
            for labelBelontoF in familyDict[key]:
                textLabels = []
                birdLabels = []
                n = len(familyDict[key])
                selectedVisualFeatures = self.pfc_feat_data_train[self.labels_train == labelBelontoF]
                tempVisualFeatures = selectedVisualFeatures[
                    random.sample([i for i in range(selectedVisualFeatures.shape[0])]
                                  , selectedVisualFeatures.shape[0] // n + 1)]

                self.pfc_feat_data_train = np.r_[self.pfc_feat_data_train, tempVisualFeatures]
                self.labels_origin_train = np.r_[
                    self.labels_origin_train, [labelBelontoF] * tempVisualFeatures.shape[0]]
                self.labels_train = np.r_[self.labels_train, [familyLabelStart] * tempVisualFeatures.shape[0]]
                if labelBelontoF not in textLabels:
                    try:
                        self.familyToText[familyLabelStart].append(
                            self.train_text_feature[labelBelontoF]) 
                    except:
                        self.familyToText[familyLabelStart] = [self.train_text_feature[labelBelontoF]]
                    textLabels.append(labelBelontoF)

                if labelBelontoF not in birdLabels:
                    try:
                        self.familyLabelToBirdLabel[familyLabelStart].append(
                            labelBelontoF) 
                    except:
                        self.familyLabelToBirdLabel[familyLabelStart] = [labelBelontoF]
                    birdLabels.append(labelBelontoF)
            familyLabelStart += 1

        genusLabelStart = 701
        for key in genusDict.keys():
            for labelBelontoG in genusDict[key]:
                textLabels = []
                birdLabels = []
                tempVisualFeatures = self.pfc_feat_data_train[self.labels_train == labelBelontoG]
                self.pfc_feat_data_train = np.r_[self.pfc_feat_data_train, tempVisualFeatures]
                self.labels_origin_train = np.r_[
                    self.labels_origin_train, [labelBelontoG] * tempVisualFeatures.shape[0]]
                self.labels_train = np.r_[self.labels_train, [genusLabelStart] * tempVisualFeatures.shape[0]]
                if labelBelontoG not in textLabels:
                    try:
                        self.genusToText[genusLabelStart].append(
                            self.train_text_feature[labelBelontoG])
                    except:
                        self.genusToText[genusLabelStart] = [self.train_text_feature[labelBelontoG]]
                    textLabels.append(labelBelontoG)

                if labelBelontoG not in birdLabels:
                    try:
                        self.genusLabelToBirdLabel[genusLabelStart].append(
                            labelBelontoG)
                    except:
                        self.genusLabelToBirdLabel[genusLabelStart] = [labelBelontoG]
                    birdLabels.append(labelBelontoG)
            genusLabelStart += 1

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        self.tr_cls_centroid = {}
        for i in range(train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)

        for i in range(601, familyLabelStart):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)
        for i in range(701, genusLabelStart):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)
        self.familyLabelEnd = familyLabelStart
        self.familyLabelStart = 601

        self.genusLabelEnd = genusLabelStart
        self.genusLabelStart = 701




class FeatDataLayer(object):
    def __init__(self, label, feat_data,  opt):
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels':minibatch_label}
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs


def get_text_feature(dir, train_test_split_dir):
    train_test_split = sio.loadmat(train_test_split_dir)
    # get training text feature
    train_cid = train_test_split['train_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    train_text_feature = text_feature[train_cid - 1]  # 0-based index
    # get testing text feature
    test_cid = train_test_split['test_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    test_text_feature = text_feature[test_cid - 1]  # 0-based index
    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)


def getFamilyAndGenusTextFeatures(train_text_feature, familyToText, genusToText):
    temp = {}
    for i in range(train_text_feature.shape[0]):
        temp[i] = [train_text_feature[i].tolist()]
    for key in familyToText.keys():
        temp[key] = np.array(familyToText[key]).tolist()
    for key in genusToText.keys():
        temp[key] = np.array(genusToText[key]).tolist()
    return temp

class FeatDataLayer_add_FG(object):

    def __init__(self, label, feat_data, opt, train_text_feature, familyToText,
                 genusToText, familyLabelToBirdLabel, genusLabelToBirdLabel,
                 labels_origin, dataset="CUB"):

        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self.familyLabelToBirdLabel = familyLabelToBirdLabel
        self.genusLabelToBirdLabel = genusLabelToBirdLabel
        self.labels_origin = labels_origin
        self.dbname = dataset
        # ·µ»ØÒ»¸ö×Öµä, key:label, values: text feature
        self.train_text_feature = getFamilyAndGenusTextFeatures(train_text_feature, familyToText, genusToText)

        if dataset == "CUB":
            self.thresh = 200
        elif dataset == "NAB":
            self.thresh = 600

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds]) 
        minibatch_origin_label = minibatch_label.copy()

        if self.dbname == "CUB":
            for i in range(len(minibatch_label)):
                if minibatch_label[i] > 300:
                    minibatch_label[i] = random.sample(self.genusLabelToBirdLabel[minibatch_label[i]], 1)[0]
                elif minibatch_label[i] > 200:
                    minibatch_label[i] = random.sample(self.familyLabelToBirdLabel[minibatch_label[i]], 1)[0]

        if self.dbname == "NAB":
            for i in range(len(minibatch_label)):
                if minibatch_label[i] > 700:
                    minibatch_label[i] = random.sample(self.genusLabelToBirdLabel[minibatch_label[i]], 1)[0]
                elif minibatch_label[i] > 600:
                    minibatch_label[i] = random.sample(self.familyLabelToBirdLabel[minibatch_label[i]], 1)[0]

        text_feat = np.array([np.array(random.sample(self.train_text_feature[i], 1)[0]) for i in minibatch_label])

        blobs = {'data': minibatch_feat, 'labels': minibatch_label,
                 'text_feat': text_feat, "minibatch_origin_label": minibatch_origin_label}

        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs
