import torch
import torch.nn.functional as F

class Calibrater:
    def __init__(self, logits, acc):
        self.logits = logits
        self.acc = acc
    
    @torch.no_grad()
    def compute_confidence(self, T):
        logits = self.logits / T
        probability = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probability, dim=-1)
        confidence = confidence.mean().item()

        return confidence
    
    def search_T(self, thres=0.001):
        acc = self.acc
        T_min, T_max = 0.1, 10
        T_best, confidence = self.search_recurse(T_min, T_max, acc, thres)

        # print(f"   T {T_best:.4f} confidence {100*confidence:.2f}% accuracy {100*acc:.2f}% ")
        return T_best, confidence

    @torch.no_grad()
    def search_recurse(self, T_min, T_max, acc, thres):
        conf_min = self.compute_confidence(T_min)
        gap_min = abs(conf_min - acc)

        conf_max = self.compute_confidence(T_max)
        gap_max = abs(conf_max - acc)

        alpha = gap_min / (gap_min + gap_max)
        T_mid = T_min * (1 - alpha) + T_max * alpha
        conf_mid = self.compute_confidence(T_mid)
        gap_mid = abs(conf_mid - acc)

        if gap_mid <= thres:
            return T_mid, conf_mid

        if conf_max > acc:
            T_max *= 1.1
        elif conf_min < acc:
            T_min /= 1.1
        elif gap_min < gap_max:
            T_max = T_mid
        else:
            T_min = T_mid

        return self.search_recurse(T_min, T_max, acc, thres)
