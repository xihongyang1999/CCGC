import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import Encoder_Net
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=4, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=500, help='feature dim')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='cora', help='name of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='number of cluster.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
parser.add_argument('--threshold', type=float, default=0.5, help='the threshold of high-confidence')
parser.add_argument('--alpha', type=float, default=0.5, help='trade-off of loss')
args = parser.parse_args()

#load data
adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()

# Laplacian Smoothing
adj_norm_s = preprocess_graph(adj, args.t, norm='sym', renorm=True)
smooth_fea = sp.csr_matrix(features).toarray()
for a in adj_norm_s:
    smooth_fea = a.dot(smooth_fea)
smooth_fea = torch.FloatTensor(smooth_fea)

acc_list = []
nmi_list = []
ari_list = []
f1_list = []

for seed in range(10):

    setup_seed(seed)

    # init
    best_acc, best_nmi, best_ari, best_f1, predict_labels, dis= clustering(smooth_fea, true_labels, args.cluster_num)

    # MLP
    model = Encoder_Net(args.linlayers, [features.shape[1]] + [args.dims])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # GPU
    model.to(args.device)
    smooth_fea = smooth_fea.to(args.device)
    sample_size = features.shape[0]
    target = torch.eye(smooth_fea.shape[0]).to(args.device)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        z1, z2 = model(smooth_fea)
        if epoch > 50:

            high_confidence = torch.min(dis, dim=1).values
            threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]

            # pos samples
            index = torch.tensor(range(smooth_fea.shape[0]), device=args.device)[high_confidence_idx]
            y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]
            index = index[torch.argsort(y_sam)]
            class_num = {}

            for label in torch.sort(y_sam).values:
                label = label.item()
                if label in class_num.keys():
                    class_num[label] += 1
                else:
                    class_num[label] = 1
            key = sorted(class_num.keys())
            if len(class_num) < 2:
                continue
            pos_contrastive = 0
            centers_1 = torch.tensor([], device=args.device)
            centers_2 = torch.tensor([], device=args.device)


            for i in range(len(key[:-1])):
                class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
                now = index[class_num[key[i]]:class_num[key[i + 1]]]
                pos_embed_1 = z1[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                pos_embed_2 = z2[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()
                centers_1 = torch.cat([centers_1, torch.mean(z1[now], dim=0).unsqueeze(0)], dim=0)
                centers_2 = torch.cat([centers_2, torch.mean(z2[now], dim=0).unsqueeze(0)], dim=0)

            pos_contrastive = pos_contrastive / args.cluster_num
            if pos_contrastive == 0:
                continue
            if len(class_num) < 2:
                loss = pos_contrastive
            else:
                centers_1 = F.normalize(centers_1, dim=1, p=2)
                centers_2 = F.normalize(centers_2, dim=1, p=2)
                S = centers_1 @ centers_2.T
                S_diag = torch.diag_embed(torch.diag(S))
                S = S - S_diag
                neg_contrastive = F.mse_loss(S, torch.zeros_like(S))
                loss = pos_contrastive + args.alpha * neg_contrastive

        else:
            S = z1 @ z2.T
            loss = F.mse_loss(S, target)

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            z1, z2 = model(smooth_fea)

            hidden_emb = (z1 + z2) / 2

            acc, nmi, ari, f1, predict_labels, dis = clustering(hidden_emb, true_labels, args.cluster_num)
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1

    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)
print(acc_list.mean(), "±", acc_list.std())
print(nmi_list.mean(), "±", nmi_list.std())
print(ari_list.mean(), "±", ari_list.std())
print(f1_list.mean(), "±", f1_list.std())
