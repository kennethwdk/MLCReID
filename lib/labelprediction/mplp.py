import torch

class MPLP(object):
    def __init__(self, t=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t = t

    def predict(self, memory, targets):
        mem_vec = memory[targets]
        mem_sim = mem_vec.mm(memory.t())
        m, n = mem_sim.size()
        mem_sim_sorted, index_sorted = torch.sort(mem_sim, dim=1, descending=True)
        multilabel = torch.zeros(mem_sim.size()).to(self.device)
        mask_num = torch.sum(mem_sim_sorted > self.t, dim=1)

        for i in range(m):
            topk = int(mask_num[i].item())
            topk = max(topk, 10)
            topk_idx = index_sorted[i, :topk]
            vec = memory[topk_idx]
            sim = vec.mm(memory.t())
            _, idx_sorted = torch.sort(sim.detach().clone(), dim=1, descending=True)
            step = 1
            for j in range(topk):
                pos = torch.nonzero(idx_sorted[j] == index_sorted[i, 0]).item()
                if pos > topk: break
                step = max(step, j)
            step = step + 1
            step = min(step, mask_num[i].item())
            if step <= 0: continue
            multilabel[i, index_sorted[i, 0:step]] = float(1)

        targets = torch.unsqueeze(targets, 1)
        multilabel.scatter_(1, targets, float(1))
        
        return multilabel