"""Binary Dawid–Skene with posterior-mean Beta(8,2) worker regularization."""
from collections import defaultdict
import numpy as np
from scipy.special import expit, logsumexp

WORKERS = ('CROP', 'CLIP', 'FRAME_BOX', 'IDENTITY')


class Reliability:
    def __init__(self, alpha=8., beta=2., iterations=20):
        self.alpha, self.beta, self.iterations = alpha, beta, iterations
        self.answers = defaultdict(lambda: defaultdict(list))
        self.values = {w: (.8, .8) for w in WORKERS}
        self.prevalence = .5
        self.posteriors = {}

    def add(self, key, view, answer, identity_answer):
        item = self.answers[key]
        if 'IDENTITY' not in item: item['IDENTITY'].append(int(identity_answer))
        item[view].append(int(answer))

    def fit(self):
        if not self.answers: return self.values
        keys = sorted(self.answers, key=str)
        yes = np.zeros((len(keys), len(WORKERS))); no = np.zeros_like(yes)
        for i, key in enumerate(keys):
            for j, worker in enumerate(WORKERS):
                observed = self.answers[key].get(worker, [])
                yes[i, j] = sum(observed); no[i, j] = len(observed)-sum(observed)
        s = np.array([self.values[w][0] for w in WORKERS]); c = np.array([self.values[w][1] for w in WORKERS])
        prevalence = self.prevalence
        for _ in range(self.iterations):
            log1 = np.log(prevalence) + yes@np.log(s) + no@np.log1p(-s)
            log0 = np.log1p(-prevalence) + yes@np.log1p(-c) + no@np.log(c)
            positive = np.exp(log1-np.logaddexp(log1, log0))
            trials = yes+no
            s = (positive@yes+self.alpha)/(positive@trials+self.alpha+self.beta)
            c = ((1-positive)@no+self.alpha)/((1-positive)@trials+self.alpha+self.beta)
            prevalence = float((positive.sum()+1)/(len(positive)+2))
        self.values = {w: (float(s[j]), float(c[j])) for j, w in enumerate(WORKERS)}
        self.prevalence = prevalence
        self.posteriors = dict(zip(keys, map(float, positive)))
        return self.values

    def receipt(self):
        return dict(queried_pairs=len(self.answers), prevalence=self.prevalence,
                    reliability={k: dict(sensitivity=v[0], specificity=v[1]) for k, v in self.values.items()},
                    EM_iterations=self.iterations, Beta_prior=[self.alpha, self.beta],
                    repeated_same_view_answers='conditionally independent observation factors; correlation is a modeling limitation')

    def __getstate__(self):
        value = dict(self.__dict__)
        value['answers'] = {key: dict(answers) for key, answers in self.answers.items()}
        return value

    def __setstate__(self, value):
        self.__dict__.update(value)
        self.answers = defaultdict(lambda: defaultdict(list),
            {key: defaultdict(list, answers) for key, answers in self.answers.items()})
