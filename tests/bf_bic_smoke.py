import numpy as np, time, causalget as cg
rng = np.random.default_rng(3)
n = 2000
def sim():
    X = np.zeros((n, 6))
    X[:,0] = rng.standard_normal(n)
    X[:,1] = rng.standard_normal(n)
    X[:,2] = X[:,0]**2 + 0.5*rng.standard_normal(n)              # 0 -> 2 (zero linear correlation)
    X[:,3] = np.tanh(2*X[:,1]) + 0.3*X[:,2] + 0.5*rng.standard_normal(n)  # 1,2 -> 3
    X[:,4] = np.sin(2*X[:,3]) + 0.5*rng.standard_normal(n)      # 3 -> 4
    X[:,5] = X[:,4]*X[:,0] + 0.5*rng.standard_normal(n)         # 4,0 -> 5
    return X
true = {(0,2),(1,3),(2,3),(3,4),(4,5),(0,5)}
def edges(dag):  # dag[child, parent] = 1
    return {(int(pa), int(ch)) for ch, pa in zip(*np.nonzero(dag))}
def report(name, dag, t):
    E = edges(dag); und = {frozenset(e) for e in E}; tund = {frozenset(e) for e in true}
    print(f"{name:14s} {t*1000:6.1f} ms  edges={sorted(E)}\n{'':14s} skeleton TP={len(und&tund)} FP={len(und-tund)} FN={len(tund-und)}  correctly oriented={len(E&true)}")
X = sim()
R = np.corrcoef(X, rowvar=False)
t=time.time(); d = cg.boss(R, n=n, discount=2, seed=1); report("linear BOSS", d, time.time()-t)
for T in (2,3,4):
    t=time.time(); d = cg.boss_bf(X, truncation_limit=T, discount=2, seed=1); report(f"BF-BIC T={T}", d, time.time()-t)
