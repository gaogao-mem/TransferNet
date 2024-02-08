import os
import torch
import torch.nn as nn
import pickle
import math
import random

from utils.BiGRU import GRU, BiGRU

class TransferNet(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.max_active = args.max_active
        self.ent_act_thres = args.ent_act_thres
        self.aux_hop = args.aux_hop
        dim_word = args.dim_word
        dim_hidden = args.dim_hidden
        
        with open(os.path.join(args.input_dir, 'wiki.pt'), 'rb') as f:
            self.kb_pair = torch.LongTensor(pickle.load(f))
            self.kb_range = torch.LongTensor(pickle.load(f))
            self.kb_desc = torch.LongTensor(pickle.load(f))

        print('number of triples: {}'.format(len(self.kb_pair)))

        num_words = len(vocab['word2id'])
        num_entities = len(vocab['entity2id'])
        self.num_steps = args.num_steps

        self.desc_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        
        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.2)
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh(),
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        self.rel_classifier = nn.Linear(dim_hidden, 1)

        self.q_classifier = nn.Linear(dim_hidden, num_entities)
        self.hop_selector = nn.Linear(dim_hidden, self.num_steps)


    def follow(self, e, pair, p):
        """
        Args:
            e [num_ent]: entity scores
            pair [rsz, 2]: pairs that are taken into consider
            p [rsz]: transfer probabilities of each pair
        """
        sub, obj = pair[:, 0], pair[:, 1]
        obj_p = e[sub] * p
        out = torch.index_add(torch.zeros_like(e), 0, obj, obj_p)
        return out
     
'''
question:  tensor([  153,  2865,     4,   173,   327,   160,  9726,   122,   113, 17318,
            0,     0,     0,     0,     0,     0], device='cuda:0') 
topic_entity:  tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0') 
answer:  tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0') 
hop:  tensor(3, device='cuda:0')
forward
torch.Size([64, 16]) 16 tensor([ 6,  4,  3, 11,  4, 10,  2,  8,  7, 11, 11, 10,  3,  8,  8,  4,  3,  5,
        10, 11,  8,  3,  9,  9,  3,  5,  4,  6,  9,  1,  6,  3,  8,  8,  3, 10,
        10,  9, 11, 10,  9,  8,  8,  9,  8, 11,  9,  5,  8, 12,  7,  8,  3,  8,
         6,  6,  8, 10,  6,  6, 10,  5,  7,  7], device='cuda:0')
tensor([10, 12, 13,  5, 12,  6, 14,  8,  9,  5,  5,  6, 13,  8,  8, 12, 13, 11,
         6,  5,  8, 13,  7,  7, 13, 11, 12, 10,  7, 15, 10, 13,  8,  8, 13,  6,
         6,  7,  5,  6,  7,  8,  8,  7,  8,  5,  7, 11,  8,  4,  9,  8, 13,  8,
        10, 10,  8,  6, 10, 10,  6, 11,  9,  9], device='cuda:0')
last_e:  tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0') torch.Size([42907])
sort_score:  tensor(1., device='cuda:0') torch.Size([42907]) tensor(1492, device='cuda:0') torch.Size([42907])
desc  torch.Size([114, 16]) 16 tensor([ 5,  7,  1,  5, 10,  5,  0,  5,  5,  7,  7,  7,  9,  6,  1,  0,  5,  5,
         0,  0,  7,  5,  0,  3,  2,  5,  4,  0, 10,  8, 10,  1,  6,  0,  5,  5,
         5,  9,  4,  2,  6,  6,  0,  7, 10,  6,  3,  5,  9,  7,  6,  0,  7,  0,
         8,  0,  6,  7,  9,  5,  5,  2,  2,  3,  5,  0,  5,  4,  8,  1,  3,  7,
         7,  0,  2,  5,  4,  8,  0,  8,  6,  6,  5,  8,  5,  7,  5, 10,  5,  2,
         4,  7,  5,  6,  0,  2,  5,  0,  5,  6,  9,  0,  0,  5,  6,  6,  6,  1,
         7,  6,  0,  9,  5,  1], device='cuda:0')
tensor([11,  9, 15, 11,  6, 11, 16, 11, 11,  9,  9,  9,  7, 10, 15, 16, 11, 11,
        16, 16,  9, 11, 16, 13, 14, 11, 12, 16,  6,  8,  6, 15, 10, 16, 11, 11,
        11,  7, 12, 14, 10, 10, 16,  9,  6, 10, 13, 11,  7,  9, 10, 16,  9, 16,
         8, 16, 10,  9,  7, 11, 11, 14, 14, 13, 11, 16, 11, 12,  8, 15, 13,  9,
         9, 16, 14, 11, 12,  8, 16,  8, 10, 10, 11,  8, 11,  9, 11,  6, 11, 14,
        12,  9, 11, 10, 16, 14, 11, 16, 11, 10,  7, 16, 16, 11, 10, 10, 10, 15,
         9, 10, 16,  7, 11, 15], device='cuda:0')
torch.Size([114, 16, 300])
torch.Size([114, 16, 768]) torch.Size([114, 768])
e_stack  [tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0',
       grad_fn=<IndexAddBackward0>)] 1 42907
       '''

    def forward(self, questions, e_s, answers=None, hop=None):
        print("forward")
        print(questions.shape, questions.size(1), questions.eq(0).long().sum(dim=1))
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        print(question_lens)
        q_word_emb = self.word_dropout(self.word_embeddings(questions)) # [bsz, max_q, dim_hidden]
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens) # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]


        device = q_word_h.device
        bsz, dim_h = q_embeddings.size()
        last_e = e_s
        word_attns = []
        ent_probs = []
        
        path_infos = [] # [bsz, num_steps]
        for i in range(bsz):
            path_infos.append([])
            for j in range(self.num_steps):
                path_infos[i].append(None)

        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_embeddings) # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [bsz, 1, max_q]
            q_dist = q_dist * questions.ne(0).float().unsqueeze(1)
            q_dist = q_dist / (torch.sum(q_dist, dim=2, keepdim=True) + 1e-6) # [bsz, 1, max_q]
            word_attns.append(q_dist.squeeze(1))
            ctx_h = (q_dist @ q_word_h).squeeze(1) # [bsz, dim_h]
            ctx_h = ctx_h + cq_t

            e_stack = []
            cnt_trunc = 0
            for i in range(bsz):
                # e_idx = torch.topk(last_e[i], k=1, dim=0)[1].tolist() + \
                #         last_e[i].gt(self.ent_act_thres).nonzero().squeeze(1).tolist()
                # TRY
                # if self.training and t > 0 and random.random() < 0.005:
                #     e_idx = last_e[i].gt(0).nonzero().squeeze(1).tolist()
                #     random.shuffle(e_idx)
                # else:
                print("last_e: ", last_e[i], last_e[i].shape)
                sort_score, sort_idx = torch.sort(last_e[i], dim=0, descending=True)
                print("sort_score: ", sort_score[0], sort_score.shape, sort_idx[0], sort_idx.shape)
                e_idx = sort_idx[sort_score.gt(self.ent_act_thres)].tolist()
                e_idx = set(e_idx) - set([0])
                if len(e_idx) == 0:
                    # print('no active entity at step {}'.format(t))
                    pad = sort_idx[0].item()
                    if pad == 0:
                        pad = sort_idx[1].item()
                    e_idx = set([pad])

                rg = []
                for j in e_idx:
                    rg.append(torch.arange(self.kb_range[j,0], self.kb_range[j,1]).long().to(device))
                rg = torch.cat(rg, dim=0) # [rsz,]
                # print(len(e_idx), len(rg))
                if len(rg) > self.max_active: # limit the number of next-hop
                    rg = rg[:self.max_active]
                    # TRY
                    # rg = rg[torch.randperm(len(rg))[:self.max_active]]
                    cnt_trunc += 1
                    # print('trunc: {}'.format(cnt_trunc))

                # print('step {}, desc number {}'.format(t, len(rg)))
                pair = self.kb_pair[rg] # [rsz, 2]
                desc = self.kb_desc[rg] # [rsz, max_desc]
                print("desc ", desc.shape, desc.size(1), desc.eq(0).long().sum(dim=1))
                desc_lens = desc.size(1) - desc.eq(0).long().sum(dim=1)
                print(desc_lens)
                desc_word_emb = self.word_dropout(self.word_embeddings(desc))
                print(desc_word_emb.shape)
                desc_word_h, desc_embeddings, _ = self.desc_encoder(desc_word_emb, desc_lens) # [rsz, dim_h]
                print(desc_word_h.shape, desc_embeddings.shape)
                d_logit = self.rel_classifier(ctx_h[i:i+1] * desc_embeddings).squeeze(1) # [rsz,]
                d_prob = torch.sigmoid(d_logit) # [rsz,]
                # transfer probability
                e_stack.append(self.follow(last_e[i], pair, d_prob))
                print("e_stack ", e_stack, len(e_stack), len(e_stack[0]))

                # collect path
                act_idx = d_prob.gt(0.9)
                act_pair = pair[act_idx].tolist()
                act_desc = [' '.join([self.vocab['id2word'][w] for w in d if w > 0]) for d in desc[act_idx].tolist()]
                path_infos[i][t] = [(act_pair[_][0], act_desc[_], act_pair[_][1]) for _ in range(len(act_pair))]
                exit()

            last_e = torch.stack(e_stack, dim=0)
            print("last_e ", last_e.shape)

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            # Specifically for MetaQA: reshape cycle entities to 0, because A-r->B-r_inv->A is not allowed
            if t > 0:
                ent_m = torch.zeros_like(last_e)
                for i in range(bsz):
                    prev_inv = set()
                    for (s, r, o) in path_infos[i][t-1]:
                        prev_inv.add((o, r.replace('__subject__', 'obj').replace('__object__', 'sub'), s))
                    for (s, r, o) in path_infos[i][t]:
                        element = (s, r.replace('__subject__', 'sub').replace('__object__', 'obj'), o)
                        if r != '__self_rel__' and element in prev_inv:
                            ent_m[i, o] = 1
                            # print('block cycle: {}'.format(' ---> '.join(list(map(str, element)))))
                last_e = (1-ent_m) * last_e

            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
        hop_logit = self.hop_selector(q_embeddings)
        hop_attn = torch.softmax(hop_logit, dim=1) # [bsz, num_hop]
        last_e = torch.sum(hop_res * hop_attn.unsqueeze(2), dim=1) # [bsz, num_ent]

        # Specifically for MetaQA: for 2-hop questions, topic entity is excluded from answer
        m = hop_attn.argmax(dim=1).eq(1).float().unsqueeze(1) * e_s
        last_e = (1-m) * last_e

        # question mask, incorporate language bias
        q_mask = torch.sigmoid(self.q_classifier(q_embeddings))
        last_e = last_e * q_mask
        
        if not self.training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'ent_probs': ent_probs,
                'path_infos': path_infos
            }
        else:
            weight = answers * 9 + 1
            loss_score = torch.mean(weight * torch.pow(last_e - answers, 2))

            loss = {'loss_score': loss_score}

            if self.aux_hop:
                loss_hop = nn.CrossEntropyLoss()(hop_logit, hop-1)
                loss['loss_hop'] = 0.01 * loss_hop

            return loss
