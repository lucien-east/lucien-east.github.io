---
title: Implement SVM with SMO from scratch in Python
date: 2022-07-30 18:39:12
categories: [Mathematics, Machine Learning]
tags: [Optimisation, Kernel Method, SVM, Python]
lang: en
mathjax: true
---

**Support vector machine** (SVM) plays an important role in machine learning. Actually, SVM is one of my favorite models because of its analytical property. Its main idea is to find the optimal hyperplane that can linearly separate the data, or maximise **margin** in the feature space. It is one of the most robust models based on statistical learning framework in **VC theory** (Vapnik–Chervonenkis theory). The determination of the model parameters is a quadratic programming problem, or **convex optimisation** more specifically. Its solution is usually **sparse**, and the new input prediction depends only on the evaluation of a subset of training data with kernel function. One of the most common and efficient approaches to train SVM is **sequential minimal optimisation** (SMO), which breaks down the problem into solving a pair of parameters analytically at each step. Besides, SMO also eliminates the need for matrix storage issue when the training data size is huge.

<!--more-->


## Recall the Objective Function of SVM

{% asset_img svm_1.png Figure 1 %}

We want to find a hyperplane $w^T x + b = 0$ and maximise the margin width $2M$ in order to separate the data labeled as $y = \pm 1$. All data points should be correctly classified, i.e. $y_i(w^T x_i + b) > 0$ for all $i$,

{% raw %}
$$
\displaylines {
\max_{w,b} M \\
\textrm{ s.t. } \frac{y_i(w^T x_i + b)}{ \Vert w \Vert} \ge M, \forall i=1,2,...,N
}
$$
{% endraw %}

The above optimisation problem can be converted into a dual form of Lagrangian function, maximising $L$:

{% raw %}
$$
\displaylines {
L(\alpha)=\sum^{N}_{i} \alpha_i - \frac{1}{2} \sum^{N}_{i}\sum^{N}_{j} y_i y_j \alpha_i \alpha_j {x_i}^T x_j \\
\textrm{ s.t. } 0 \le \alpha_i,  \forall i \\
\textrm{ and } \sum^{N}_{i}y_i \alpha_i = 0
}
\tag{1}
$$
{% endraw %}

The solution for $w$ is

$$
w = \sum^{N}_{i} y_i \alpha_i x_i \tag{2}
$$

If we allow some points are misclassified with penalty $C$, then the constraints of $\alpha_i$ become

$$
0 \le \alpha_i \le C,  \forall i
$$


> For points with $0 \lt \alpha_i \lt C$, they just lie on the edge of the margin. In contrast, when $\alpha_i = 0$, it is in the decision boundary and does not contribute to the prediction. The points with $\alpha_i = C$ lie inside the margin and might either be classified correctly or not.

Points with non-zero $\alpha_i$ are called **support vectors**.  These can be demonstrated as below:

{% asset_img svm_2.png Figure 2 %}



If we apply the feature transformation with kernel function $K(x_i, x_j)$, then the Lagrangian function $(1)$ turns into

$$
L(\alpha)=\sum^{N}_{i} \alpha_i - \frac{1}{2} \sum^{N}_{i}\sum^{N}_{j} y_i y_j \alpha_i \alpha_j K(x_i, x_j) \tag{3}
$$

The constraints of Lagrangian multipliers are the same as before.

The predicted $\hat{y}$ of new input is

$$
\hat{y} = \sum^{N}_{i} y_i \alpha_i K(x_i, x) + b  \tag{4}
$$

## Sequential Minimal Optimisation

SMO process mainly contains two parts: one is to solve two Lagrangian parameters analytically at a step, and then decide how to choose these two parameters heuristically for speed up.

### Solving Lagrangian Multipliers

Using the constraint $\sum^{N}_{i}y_i \alpha_i = 0$ in $(1)$, we can get

$$
0 = y_1 \alpha^{old}_1 + y_2 \alpha^{old}_2 + \sum^{N}_{i=3}y_i \alpha_i = y_1 \alpha^{new}_1 + y_2 \alpha^{new}_2 + \sum^{N}_{i=3}y_i \alpha_i
$$

$$
\Rightarrow y_1 \alpha^{old}_1 + y_2 \alpha^{old}_2 = k = y_1 \alpha^{new}_1 + y_2 \alpha^{new}_2  \tag{5}
$$

where $k = -\sum^{N}_{i=3}y_i \alpha_i$.

Remember that $y$ is either $1$ or $-1$ , with the constraint $0 \le \alpha_i \le C$, $\alpha_1$ and $\alpha_2$ can only lie on the diagonal line segment shown as below:

{% asset_img svm_3.png Figure 3 %}

The Lagrangian multiplier $\alpha_2$ can be solved by the first derivative of the objective function to find its extremum. The analytical form to solve $\alpha_2$ is

$$
\alpha^{new}_2 = \alpha^{old}_2 + y_2 \frac{E_2 - E_1}{\eta},
$$

$$
E_i = \hat{y_i} - y_i  \text{, } \eta = K_{11} + K_{22} - 2K_{12} \tag{6}
$$
$$
K_{ij} = K(x_i, x_j)
$$


{% asset_img svm_4.png Figure 4 %}

#### Case 1: $y_1$ $\neq$ $y_2$

$$
L = \max(0, \alpha_2 - \alpha_1)
$$

$$
H = \min(C, C + \alpha_2 - \alpha_1)
$$


#### Case 2: $y_1 = y_2$

$$
L = \max(0, \alpha_2 + \alpha_1 - C)
$$

$$
H = \min(C, \alpha_2 + \alpha_1)
$$

The $\alpha^{new}_2$ should be bounded by $L$ and $H$. 

{% raw %}
$$
  \alpha^{new}_2=\begin{cases}
    L, & \text{if $\alpha^{new}_2 \lt L$}.\\
    \alpha^{new}_2, & \text{if $L \le \alpha^{new}_2 \le H$}.\\
    H, & \text{if $\alpha^{new}_2 \gt H$}
  \end{cases}
$$
{% endraw %}

We can then obtain $\alpha^{new}_1$ by multiplying $y_1$ on both sides in $(5)$,

$$
\alpha^{new}_1 = \alpha^{old}_1 + y_1 y_2 (\alpha^{old}_2 - \alpha^{new}_2) \tag{7}
$$


#### Abnormal Case for $\eta$

Normally, $\eta$ should be greater than 0. However, if we encounter the abnormal case that $\eta \le 0$, e.g. picking the same points or an incorrect kernel that does not obey Mercer's condition, the full version of SMO algorithm will move the Lagrangian multiplier to the end of the line segment that can maximise the objective function.[^1] [^2]

Another simple way to handle this is to treat the scenario as no progress being made for this pair of $\alpha$.


### Comupting the Threshold b

We can update the threshold $b$ after getting $\alpha$ at each step.

When $0 \lt \alpha_1 \lt C$, $b_1$ is a valid threshold because it makes the output $\hat{y_1}$ be the same as $y_1$ when the input is $x_1$

$$
E_1 = (y_1 \alpha^{old}_1 K_{11} + y_2 \alpha^{old}_2 K_{12} + b) - (y_1 \alpha^{new}_1 K_{11} + y_2 \alpha^{new}_2 K_{12} + b_1)
$$

$$
\Rightarrow b_1 = b - E_1 - y_1 (\alpha^{new}_1 - \alpha^{old}_1) K_{11} - y_2 (\alpha^{new}_2  - \alpha^{old}_2) K_{12}
$$

Similarly, when $\alpha_2$ is not at bounds, $b_2$ is a valid threshold

$$
b_2 = b - E_2 - y_1 (\alpha^{new}_1 - \alpha^{old}_1) K_{12} - y_2 (\alpha^{new}_2  - \alpha^{old}_2) K_{22}
$$

When both $b_1$ and $b_2$ are valid, they will be equal because $\hat{y_i} y_i = 1$, and the new $E_1$ and $E_2$ will be 0. This can be easily verified with $(6)$ and $(7)$. Intuitively, when $y_1 = y_2$, they are both at bounds and is trivial, whereas for $y_1 \neq y_2$, they both try to maximise the margin width, and this results in $b_1$ and $b_2$ being equal.

For other cases, we could choose the halfway between $b_1$ and $b_2$.

{% raw %}
$$
  b=\begin{cases}
    b_1, & \text{if $0 \lt \alpha^{new}_1 \lt C$}.\\
    b_2, & \text{if $0 \lt \alpha^{new}_2 \lt C$}.\\
    \frac{(b_1+b_2)}{2}, & \text{otherwise}.
  \end{cases}
$$
{% endraw %}


### Choosing the Multipliers to Optimise

In order to speed up the training rate, the main idea of choosing the multipliers in SMO can be briefly summarised as the following.

> Firstly, choose the multiplier that are likely to violate the KKT conditions to optimise, i.e. $0 < \alpha_i < C$. When one multiplier is chosen, another multiplier would be the one that can maximise the step size, $\vert E_2 - E_1 \vert$.

Then SMO will scan the entire data sets until the algorithm terminates.


### Other Tricks to Make the Training Process Faster

#### Error Cache Update

We can reduce the computational cost to compute the error cache, which stores $E_i$, after Lagrangian multipliers update. From $(6)$,

$$
E^{old}_i = y_1 \alpha^{old}_1 K_{1i} + y_2 \alpha^{old} K_{2i} + \sum^N_{j=3} y_j \alpha_j K_{ij} + b - y_i
$$

$$
E^{new}_i = y_1 \alpha^{new}_1 K_{1i} + y_2 \alpha^{new} K_{2i} + \sum^N_{j=3} y_j \alpha_j K_{ij} + b_{new} - y_i
$$

$$
\Rightarrow E^{new}_i = E^{old}_i + y_1 (\alpha^{new}_1 - \alpha^{old}_1) K_{1i} + y_2 (\alpha^{new}_2 - \alpha^{old}_2) K_{2i} + (b_{new} - b)
$$


#### Linear SVM Optimisation

The linear SVM only needs to store a single weight vector, $w$. It can also be updated using similar mechanism as error cache. From $(2)$,

$$
w^{old} = y_1 \alpha^{old}_1 x_1 + y_2 \alpha^{old}_2 x_2 + \sum^N_{j=3} \alpha_j x_j
$$

$$
w^{new} = y_1 \alpha^{new}_1 x_1 + y_2 \alpha^{new}_2 x_2 + \sum^N_{j=3} \alpha_j x_j
$$


$$
\Rightarrow w^{new} = w^{old} + y_1 (\alpha^{new}_1 - \alpha^{old}_1) x_1 + y_2 (\alpha^{new}_2 - \alpha^{old}_2) x_2
$$



## Implementation

### Source Code

```Python =
# File: MySVM.py

import numpy as np

class SVM:
    def __init__(self, X, y, C=1, kernel='linear', b=0, max_iter=300, tol=1e-5, eps=1e-8):
        self.X = X
        self.y = y
        self.m, self.n = np.shape(self.X)
        self.C = C

        self.alphas = np.zeros(self.m)
        self.b = b

        self.kernel = kernel       # 'linear', 'rbf'
        if kernel == 'linear':
            self.kernel_func = self.linear_kernel
        elif kernel == 'gaussian' or kernel == 'rbf':
            self.kernel_func = self.gaussian_kernel
        else:
            raise ValueError('unknown kernel type')

        self.error = np.zeros(self.m)

        self.max_iter=max_iter
        self.tol = tol
        self.eps = eps

        self.is_linear_kernel = True if self.kernel == 'linear' else False
        self.w = np.zeros(self.n)  # used by linear kernel
    
    def linear_kernel(self, x1, x2, b=0):
        return x1 @ x2.T + b
    
    def gaussian_kernel(self, x1, x2, sigma=1):
        if np.ndim(x1) == 1 and np.ndim(x2) == 1:
            return np.exp(-(np.linalg.norm(x1-x2,2))**2/(2*sigma**2))
        elif(np.ndim(x1)>1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2)>1):
            return np.exp(-(np.linalg.norm(x1-x2, 2, axis=1)**2)/(2*sigma**2))
        elif np.ndim(x1) > 1 and np.ndim(x2) > 1 :
            return np.exp(-(np.linalg.norm(x1[:, np.newaxis] \
                             - x2[np.newaxis, :], 2, axis = 2) ** 2)/(2*sigma**2))
        return 0.
    
    def predict(self, x):
        result = (self.alphas * self.y) @ self.kernel_func(self.X, x) + self.b
        return result

    def get_error(self, i):
        return self.predict(self.X[i,:]) - self.y[i]

    def take_step(self, i1, i2):
        if (i1 == i2):
            return 0

        x1 = self.X[i1, :]
        x2 = self.X[i2, :]

        y1 = self.y[i1]
        y2 = self.y[i2]

        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]

        b = self.b

        E1 = self.get_error(i1)
        E2 = self.get_error(i2)

        s = y1 * y2

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return 0

        k11 = self.kernel_func(x1, x1)
        k12 = self.kernel_func(x1, x2)
        k22 = self.kernel_func(x2, x2)

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:
            # Abnormal case for eta <= 0, treat this scenario as no progress
            return 0

        # Numerical tolerance
        # if abs(alpha2_new - alpha2) < self.eps:   # this is slower
        # below is faster, not degrade the SVM performance
        if abs(alpha2_new - alpha2) < self.eps * (alpha2 + alpha2_new + self.eps):
            return 0

        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)

        # Numerical tolerance
        if alpha1_new < self.eps:
            alpha1_new = 0
        elif alpha1_new > (self.C - self.eps):
            alpha1_new = self.C

        # Update threshold
        b1 = b - E1 - y1 * (alpha1_new - alpha1) * k11 - y2 * (alpha2_new - alpha2) * k12
        b2 = b - E2 - y1 * (alpha1_new - alpha1) * k12 - y2 * (alpha2_new - alpha2) * k22
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = 0.5 * (b1 + b2)

        # Update weight vector for linear SVM
        if self.is_linear_kernel:
            self.w = self.w + y1 * (alpha1_new - alpha1) * x1 \
                            + y2 * (alpha2_new - alpha2) * x2

        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        # Error cache update
        ## if alpha1 & alpha2 are not at bounds, the error will be 0
        self.error[i1] = 0
        self.error[i2] = 0

        i_list = [idx for idx, alpha in enumerate(self.alphas) \
                      if 0 < alpha and alpha < self.C]
        for i in i_list:
            self.error[i] += \
                  y1 * (alpha1_new - alpha1) * self.kernel_func(x1, self.X[i,:]) \
                + y2 * (alpha2_new - alpha2) * self.kernel_func(x2, self.X[i,:]) \
                + (self.b - b)

        return 1


    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.get_error(i2)
        r2 = E2 * y2

        # Choose the one that is likely to violiate KKT
        # if (0 < alpha2 < self.C) or (abs(r2) > self.tol):  # this is slow
        # below is faster, not degrade the SVM performance
        if ((r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0)):
            if len(self.alphas[(0 < self.alphas) & (self.alphas < self.C)]) > 1:
                if E2 > 0:
                    i1 = np.argmin(self.error)
                else:
                    i1 = np.argmax(self.error)

                if self.take_step(i1, i2):
                    return 1

            # loop over all non-zero and non-C alpha, starting at a random point
            i1_list = [idx for idx, alpha in enumerate(self.alphas) \
                           if 0 < alpha and alpha < self.C]
            i1_list = np.roll(i1_list, np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

            # loop over all possible i1, starting at a random point
            i1_list = np.roll(np.arange(self.m), np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

        return 0
    
    def fit(self):
        loop_num = 0
        numChanged = 0
        examineAll = True
        while numChanged > 0 or examineAll:
            if loop_num >= self.max_iter:
                break

            numChanged = 0
            if examineAll:
                for i2 in range(self.m):
                    numChanged += self.examine_example(i2)
            else:
                i2_list = [idx for idx, alpha in enumerate(self.alphas) \
                                if 0 < alpha and alpha < self.C]
                for i2 in i2_list:
                    numChanged += self.examine_example(i2)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

            loop_num += 1

```

### Demo

```Python =
# File: SVM_test.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import MySVM


def gen_circle(n=50, center_x=0, center_y=0, radius=1, label=0):
    
    """
    A simple function that generates circular distribution
    n: number of points (default=50)
    center_x: the center for X (default=0)
    center_y: the center for Y (default=0)
    radius: the radius of circle (default=1)
    """

    # random angle
    alpha = 2 * np.pi * np.random.rand(n)
    # random radius
    r = radius * np.sqrt(np.random.rand(n))
    # calculating coordinates
    x = r * np.cos(alpha) + center_x
    y = r * np.sin(alpha) + center_y

    label = np.ones(n) * label

    return [x, y, label]


if __name__ == '__main__':
    np.random.seed(5)   # to reproduce

    n = 100
    C0 = gen_circle(n, center_x=1, center_y=1, radius=1.05, label=1)
    C1 = gen_circle(n, center_x=-1, center_y=-1, radius=1.05, label=-1)

    x0 = np.append(C0[0], C1[0])
    x1 = np.append(C0[1], C1[1])

    X = np.c_[x0, x1]
    Y = np.append(C0[2], C1[2])

    scaler = StandardScaler()
    train_x = scaler.fit_transform(X)

    model = MySVM.SVM(train_x, Y, C=1, kernel='linear', max_iter=600, tol=1e-5, eps=1e-5)
    # model = MySVM.SVM(train_x, Y, C=1, kernel='rbf', max_iter=600, tol=1e-5, eps=1e-5)
    model.fit()

    train_y = model.predict(train_x)

    print('support vector: {} / {}'\
        .format(len(model.alphas[model.alphas != 0]), len(model.alphas)))
    sv_idx = []
    for idx, alpha in enumerate(model.alphas):
        if alpha != 0:
            print('index = {}, alpha = {:.3f}, predict y={:.3f}'\
                .format(idx, alpha, train_y[idx]))
            sv_idx.append(idx)


    print(f'bias = {model.b}')
    print('training data error rate = {}'.format(len(Y[Y * train_y < 0])/len(Y)))

    ## Draw the Plot
    plt.plot(C0[0], C0[1], 'o', markerfacecolor='r', markeredgecolor='None', alpha=0.55)
    plt.plot(C1[0], C1[1], 'o', markerfacecolor='b', markeredgecolor='None', alpha=0.55)

    resolution = 50
    dx = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    dy = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    dx, dy = np.meshgrid(dx, dy)
    plot_x = np.c_[dx.flatten(), dy.flatten()]

    dz = model.predict(scaler.transform(plot_x))
    dz = dz.reshape(dx.shape)

    plt.contour(dx, dy, dz, alpha=1, colors=('b', 'k', 'r'), \
                levels=(-1, 0, 1), linestyles = ('--', '-', '--'))

    label_cnt = 0
    for i in sv_idx:
        if label_cnt == 0:
            plt.scatter(X[i, 0], X[i, 1], marker='*', color='k', \
                        s=120, label='Support vector')
            label_cnt += 1
            continue

        plt.scatter(X[i, 0], X[i, 1], marker='*', color='k', s=120)

    plt.legend()
    plt.show()
```


{% asset_img svm_5.png Figure 5 %}



[^1]: [Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)

[^2]: [The implementation of Support Vector Machines using the sequential minimal optimization algorithm](https://www.cs.mcgill.ca/~hv/publications/99.04.McGill.thesis.gmak.pdf)


