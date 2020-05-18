几个通用优化问题求解

1. 更新（字典）$D$：
   $$
   \min\limits_{D} a||X-DY||_F^2\quad s.t. ||D_{:,i}||_2^2\le c,\quad \forall i=1,2,...,k
   $$
   其中$D\in R^{n\times k}$
   
   定义拉格朗日函数：
   $$
   \mathcal{L}(D,\overrightarrow{\lambda})=a·tr((X-DY)^T(X-DY))+\sum\limits_{i=1}^k\lambda_i(\sum\limits_{j=1}^nD_{j,i}^2-c)\\
=a·tr(X^TX-X^TDY-Y^TD^TX+Y^TD^TDY)+tr(D^TD\Lambda)-tr(c\Lambda)
   $$
   其中$\overrightarrow{\lambda}=[\lambda_1,...,\lambda_k]^T$为拉格朗日乘子向量，$\Lambda=diag\{\overrightarrow{\lambda}\}$，$\lambda_i\ge 0$，令$\frac{\partial \mathcal{L}(D,\overrightarrow{\lambda})}{\partial D}=0$，得$D=aXY^T(aYY^T+\Lambda)^{-1}$，转换为拉格朗日对偶问题：
   $$
   \mathcal{D}(\overrightarrow{\lambda})=tr(aX^TX-a^2XY^T(aYY^T+\Lambda)^{-1}(XY^T)^T-c\Lambda)
   $$
   利用牛顿法求对偶函数的极大值
   
   
   
   事实上，系数$a$并不起作用
   
2. 使用软阈值解优化问题：$\min\limits_{X}||X-B||_2^2+2\lambda ||X||_1$

   $x_i=soft(B,\lambda)_i=\begin{cases}B_i+\lambda,\quad B_i\lt-\lambda \\0,\quad |B_i|\le \lambda \\B_i-\lambda,\quad B_i\gt \lambda\end{cases}=sign(B_i)\max\{|B_i|-\lambda,0\}$

   当遇到$\min\limits_{X}||X-B||_2^2+\lambda ||X||_1$时，则$x_i=soft(B,\frac{\lambda}{2})_i$

3. 

