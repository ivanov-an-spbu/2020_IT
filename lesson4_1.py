import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

X1 = np.random.multivariate_normal((0, 0),
                                   np.array([[0.06, 0], [0, 0.08]]), size=300)
X2 = np.random.multivariate_normal((0.7, 1),
                                   np.array([[0.15, 0], [0, 0.18]]), size=500)


X = np.vstack((X1, X2))
X = np.random.permutation(X)

[i, j] = np.random.random_integers(0, X.shape[0], 2)


# plt.scatter(X[:, 0], X[:, 1])

c1 = X[i]
c2 = X[j]

dist1 = np.sum((X - c1)**2, axis=1)
dist2 = np.sum((X - c2)**2, axis=1)

print(dist1[:5])
print(dist2[:5])

res = dist1<dist2


plt.scatter(X[res, 0], X[res, 1])
plt.scatter(X[~res, 0], X[~res, 1])



# plt.scatter(X[:300, 0], X[:300, 1])

# plt.scatter(X1[:, 0], X1[:, 1])
# plt.scatter(X2[:, 0], X2[:, 1])

plt.scatter(X[i, 0], X[i, 1], marker='*', color='y')
plt.scatter(X[j, 0], X[j, 1], marker='*', color='m')


plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()