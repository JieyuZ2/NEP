import sys
import numpy as np
import gensim
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.enabled = True

#
# class MLPClassifier(nn.Module):
#     def __init__(self, embed,
#                  embed_size,
#                  classifier_hidden_dim,
#                  classifier_output_dim):
#         super(MLPClassifier, self).__init__()
#
#         self.embeds = Variable(embed, requires_grad=False).cuda()
#
#         self.classifier = nn.Sequential(nn.Linear(embed_size, classifier_hidden_dim, bias=True),
#                                         nn.ReLU(inplace=True),
#                                         nn.Linear(classifier_hidden_dim, classifier_output_dim, bias=True))
#
#     def look_up_embed(self, id):
#         return self.embeds[id].view(1,-1)
#
#     def look_up_embeds(self, ids):
#         return self.embeds.index_select(0, ids)
#
#     def forward(self, batch):
#         input = []
#         for id in batch:
#             input.append(self.look_up_embed(id))
#         inputs = torch.cat(input, 0)
#
#         # inputs = self.look_up_embeds(batch)
#         output = self.classifier(inputs)
#         return output


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer=1):
        super(MLPClassifier, self).__init__()
        if layer == 1:
            self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim, bias=True))
        elif layer == 2:
            self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim, bias=True))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, label):
        return self.loss_fn(self.classifier(batch), label)

    def predict(self, batch, label):
        self.eval()
        scores = self.classifier(batch)
        predicted = scores.argmax(dim=1)
        c = (predicted == label).sum(dim=0).item()
        acc = c / len(label)
        self.train()
        return acc


class MLP(nn.Module):
    def __init__(self, num_node, embedding_size, output_dim):
        super(MLP, self).__init__()
        self.num_node = num_node
        self.embedding_size = embedding_size
        self.embeds = nn.Embedding(self.num_node, self.embedding_size)
        self.embeds.weight = nn.Parameter(torch.FloatTensor(self.num_node, self.embedding_size).uniform_(
                                                           -0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.classifier = nn.Linear(embedding_size, output_dim, bias=True)

    def look_up_embeds(self, ids):
        return self.embeds(ids)

    def forward(self, ids):
        X = self.look_up_embeds(ids)
        return self.classifier(X)


class ModuleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModuleBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, bias=None):
        if bias is None:
            return F.relu(F.linear(input, self.weight))
        else:
            return F.relu(F.linear(input, self.weight, bias))


class ModuleNet(nn.Module):
    def __init__(self, num_node, embedding_size, num_module, target_embedding=None):
        super(ModuleNet, self).__init__()
        self.num_node = num_node
        self.embedding_size = embedding_size

        if target_embedding is None:
            self.embeds = nn.Embedding(self.num_node, self.embedding_size)
            self.embeds.weight = nn.Parameter(torch.FloatTensor(self.num_node, self.embedding_size).uniform_(
                                                           -0.5 / self.embedding_size, 0.5 / self.embedding_size))
        else:self.embeds = nn.Embedding.from_pretrained(target_embedding.data, freeze=False)
        self.target_embeds = nn.Embedding.from_pretrained(self.embeds.weight.data, freeze=True)

        self.module_dict = nn.ModuleDict(
            {str(module_id): ModuleBlock(in_features=embedding_size, out_features=embedding_size)
             for module_id in range(num_module)})

    def copy_embedding(self):
        self.target_embeds.weight.data.copy_(self.embeds.weight.data)

    def forward(self, path, nodes):
        X = self.embeds(nodes[:,0])
        for i, edge in enumerate(path):
            X = self.module_dict[edge](X)
        target = self.target_embeds(nodes[:,-1])
        loss = ((target-X)**2).sum(1)
        return loss.mean()

    def save_embedding(self, id_to_name, path, binary):
        target_embed = self.embeds.weight.data.cpu().numpy()
        learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(self.embedding_size)
        learned_embed.add(id_to_name[:self.num_node], target_embed)
        learned_embed.save_word2vec_format(fname=path, binary=binary, total_vec=self.num_node)
        return learned_embed

    def return_embedding(self):
        return self.embeds.weight.data.cpu().numpy()


class LabelEncoder(object):
    def __init__(self, dataset, args):
        self.model = MLP(len(dataset.type_to_node[args.target_node_type]), args.embedding_size, dataset.num_class).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.X = dataset.train_nodes
        self.X_var = Variable(torch.LongTensor(self.X)).cuda()
        self.y = [dataset.node_to_label[node] for node in self.X]
        self.label_var = Variable(torch.LongTensor(self.y)).cuda()
        self.num = len(self.X)
        self.batch_size = 32
        self.num_epoch = 200

    def next_batch(self):
        for i in np.arange(0, self.num, self.batch_size):
            yield self.X_var[i:i + self.batch_size], self.label_var[i:i + self.batch_size]

    def train(self):
        self.model.train()
        best_train_acc = 0
        for epoch in range(self.num_epoch):
            for batch in self.next_batch():
                X_batch, y_batch = batch
                self.optimizer.zero_grad()
                scores = self.model(X_batch)
                loss = self.loss_fn(scores, y_batch)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            scores = self.model(self.X_var)
            preds = np.argmax(scores.data.cpu().numpy(), axis=1)
            num_correct = np.sum(preds == self.y)
            train_acc = float(num_correct) / self.num
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1
            msg = '\rEpoch:{}/{} train acc={}, best train acc={} @epoch:{}'.\
                format(epoch + 1, self.num_epoch, train_acc, best_train_acc, best_train_acc_epoch)
            print(msg, end='')
            sys.stdout.flush()
        print('')
        return self.model.embeds.weight.data.cpu()
