# Fixed-Effect Model

[TOC]

## Ordinary Linear Model

Denote $h_{\theta}(x)$ as the linear model hypothesis function:

$$
\begin{equation*}
h_{\theta}( x) =\theta _{0} +\theta _{1} x_{1} +\theta _{2} x_{2} +\dotsc +\theta _{n} x_{n} =\theta ^{\top } x
\end{equation*}
$$

where,

$$
\begin{equation*}
x_{0} =1,x=\begin{bmatrix}
x_{0}\\
x_{1}\\
\vdots \\
x_{n}
\end{bmatrix} ,\theta =\begin{bmatrix}
\theta _{0}\\
\theta _{1}\\
\vdots \\
\theta _{n}
\end{bmatrix}
\end{equation*}
$$

## Fixed-Effect Model

Denote $\beta$ as coefficient instead of $\theta$.
Denote $\alpha$ as fixed-effect of group $i$.
Denote $e$ as error term.

Fixed-effect model can be written as:

$$
\begin{gather*}
y_{it} =\alpha _{i} +( \beta _{0} +\beta _{1} x_{1,it} +\beta _{2} x_{2,it} +\dotsc +\beta _{n} x_{n,it}) +e_{it}\\
y_{it} =\alpha _{i} +\beta ^{\top } x_{it} +e_{it}
\end{gather*}
$$

where,

$$
\begin{equation*}
x_{0,it} =1,x_{it} =\begin{bmatrix}
x_{0,it}\\
x_{1,it}\\
\vdots \\
x_{n,it}
\end{bmatrix} ,\beta =\begin{bmatrix}
\beta _{0}\\
\beta _{1}\\
\vdots \\
\beta _{n}
\end{bmatrix}
\end{equation*}
$$

## Apply Mean

Denote $m_{i}$ as number of training examples of group $i$.

Take average over group $i$:

$$
\begin{equation*}
\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1} y_{it} =\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1}\left( \alpha _{i} +\beta ^{\top } x_{it} +e_{it}\right)
\end{equation*}
$$


$$
\begin{equation*}
\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1} y_{it} =\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1} \alpha _{i} +\beta ^{\top }\begin{bmatrix}
\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1} x_{0,it}\\
\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1} x_{1,it}\\
\vdots \\
\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1} x_{n,it}
\end{bmatrix} +\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1} e_{it}
\end{equation*}
$$

Simplify mean using bar notation:

$$	
\begin{equation*}
\overline{y_{i}} =\alpha _{i} +\beta ^{\top }\begin{bmatrix}
\overline{x_{0,i}}\\
\overline{x_{1,i}}\\
\vdots \\
\overline{x_{n,i}}
\end{bmatrix} +\overline{e_{i}}
\end{equation*}
$$

$$
\begin{equation*}
\overline{y_{i}} =\alpha _{i} +\beta ^{\top }\overline{x_{i}} +\overline{e_{i}}
\end{equation*}
$$

where,

$$
\begin{equation*}
\overline{x_{i}} =\begin{bmatrix}
\overline{x_{0,i}}\\
\overline{x_{1,i}}\\
\vdots \\
\overline{x_{n,i}}
\end{bmatrix}
\end{equation*}
$$

## Predictive Model

Now, we have two equations:

$$
\begin{equation*}
\begin{array}{ c c c c }
y_{it} =\alpha _{i} +\beta ^{\top } x_{it} +e_{it} &  &  & ( 1.a)\\
\overline{y_{i}} =\alpha _{i} +\beta ^{\top }\overline{x_{i}} +\overline{e_{i}} &  &  & ( 1.b)
\end{array}
\end{equation*}
$$

Apply $(1.a)-(1.b)$:

$$
\begin{gather*}
y_{it} -\overline{y_{i}} =\beta ^{\top }\left( x_{it} -\overline{x_{i}}\right) +\left( e_{it} -\overline{e_{i}}\right)\\
\ddot{y_{it}} =\beta ^{\top }\ddot{x_{it}} +\ddot{e_{it}}
\end{gather*}
$$

where,

$$
\begin{equation*}
\ddot{y_{it}} =\left( y_{it} -\overline{y_{i}}\right) ,\ddot{x_{it}} =\left( x_{it} -\overline{x_{i}}\right) ,\ddot{e_{it}} =\left( e_{it} -\overline{e_{i}}\right)
\end{equation*}
$$

Since error term cannot be predicted, so we define the predictive model to be a linear model hypothesis function in terms of $\ddot{x_{it}}$:

$$
\begin{gather*}
\widehat{\ddot{y_{it}}} =\ddot{y_{it}} -\ddot{e_{it}} =\beta ^{\top }\ddot{x_{it}} =h_{\beta }\left(\ddot{x_{it}}\right)\\
\widehat{\ddot{y_{it}}} =h_{\beta }\left(\ddot{x_{it}}\right) =\beta ^{\top }\ddot{x_{it}}
\end{gather*}
$$

The hypothesis function $h_{\beta}(\ddot{x_{it}})$ will be use in machine learning to train for the coefficients $\beta$.

## Actual Prediction of y

Once $\beta$ has been learned, we can proceed to the prediction of $\ddot{y_{it}}$ and then the actual prediction of $y_{it}$.

### Prediction Method 1

We know that $\ddot{y_{it}}$ is approximately equal to $\widehat{\ddot{y_{it}}}$.

$$
\begin{gather*}
y_{it} -\overline{y_{i}} =\ddot{y_{it}} \approx \widehat{\ddot{y_{it}}} =h_{\beta }\left(\ddot{x_{it}}\right) =h_{\beta }\left( x_{it} -\overline{x_{i}}\right)\\
y_{it} -\overline{y_{i}} \approx h_{\beta }\left( x_{it} -\overline{x_{i}}\right)\\
y_{it} \approx h_{\beta }\left( x_{it} -\overline{x_{i}}\right) +\overline{y_{i}}
\end{gather*}
$$

So, we define the actual prediction of $y_{it}$ to be:

$$
\begin{equation*}
\widehat{y_{it}} =h_{\beta }\left( x_{it} -\overline{x_{i}}\right) +\overline{y_{i}}
\end{equation*}
$$

### Prediction Method 2

From the original fixed-effect model, we define the prediction of $y_{it}$ by removing the error term $e_{it}$:

$$
\begin{equation*}
\widehat{y_{it}} =y_{it} -e_{it} =\alpha _{i} +\beta ^{\top } x_{it}
\end{equation*}
$$

We define cost function $J$ of $y_{it}$ prediction by using Mean Square Error (MSE):

$$
\begin{equation*}
J( \alpha _{i}) =\frac{1}{2m_{i}}\sum ^{m_{i}}_{t=1}\left(\widehat{y_{it}} -y_{it}\right)^{2}
\end{equation*}
$$

We need to find $\alpha_{i}$ that minimizes cost function $J(\alpha_{i})$:

$$
\begin{equation*}
\min_{a_{i}} J( \alpha _{i}) =\min_{\alpha _{i}}\frac{1}{2m_{i}}\sum ^{m_{i}}_{t=1}\left(\ddot{y_{it}} -y_{it}\right)^{2}
\end{equation*}
$$

Take partial derivative of $J(\alpha_{i})$ with respect to $\alpha_{i}$:

$$
\begin{equation*}
\frac{\partial }{\partial \alpha _{i}} J( \alpha _{i}) =\frac{\partial }{\partial \alpha _{i}}\left(\frac{1}{2m_{i}}\sum ^{m_{i}}_{t=1}\left(\widehat{y_{it}} -y_{it}\right)^{2}\right) =\frac{1}{m}\sum ^{m_{i}}_{t=1}\left(\widehat{y_{it}} -y_{it}\right)
\end{equation*}
$$

It shall minimizes when the gradient = $0$:

$$
\begin{equation*}
\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1}\left(\widehat{y_{it}} -y_{it}\right) =\frac{1}{m_{i}}\sum ^{m_{i}}_{t=1}\left( \alpha _{i} +\beta ^{\top } x_{it} -y_{it}\right) =0
\end{equation*}
$$

$$
\begin{equation*}
\alpha _{i} =\frac{1}{m}\sum ^{m_{i}}_{t=1}\left( y_{it} -\beta ^{\top } x_{it}\right) =\overline{y_{i}} -\beta ^{\top }\overline{x_{i}}
\end{equation*}
$$

Now, we have two equations:

$$
\begin{equation*}
\begin{array}{ c c c c }
\widehat{y_{it}} =\alpha _{i} +\beta ^{\top } x_{it} &  &  & ( 2.a)\\
\alpha _{i} =\overline{y_{i}} -\beta ^{\top }\overline{x_{i}} &  &  & ( 2.b)
\end{array}
\end{equation*}
$$

Substitute $(2.b)$ into $(2.a)$:

$$
\begin{equation*}
\widehat{y_{it}} =\left(\overline{y_{i}} -\beta ^{\top }\overline{x_{i}}\right) +\beta ^{\top } x_{it} =\left(\overline{y_{i}} -\beta ^{\top }\overline{x_{i}}\right) +h_{\beta }( x_{it})
\end{equation*}
$$

### Cross-Checking Method 1 and Method 2

Denote $A$ as prediction using method $1$.
Denote $B$ as prediction using method $2$.

$$
\begin{equation*}
\begin{array}{ c c c c }
\widehat{y_{it}}^{( A)} =h_{\beta }\left( x_{it} -\overline{x_{i}}\right) +\overline{y_{i}} &  &  & ( 3.a)\\
\widehat{y_{it}}^{( B)} =\overbrace{\left(\overline{y_{i}} -\beta ^{\top }\overline{x_{i}}\right)}^{\alpha _{i}} +h_{\beta }( x_{it}) &  &  & ( 3.b)
\end{array}
\end{equation*}
$$

We prove that $\widehat{y_{it}}^{( A)} = \widehat{y_{it}}^{( B)}$.

From $(3.a)$:

$$
\begin{equation*}
LHS=h_{\beta }\left( x_{it} -\overline{x_{i}}\right) +\overline{y_{i}} =\beta ^{\top }\left( x_{it} -\overline{x_{i}}\right) +\overline{y_{i}} =\overline{y_{i}} -\beta ^{\top }\overline{x_{i}} +\beta ^{\top } x_{it}
\end{equation*}
$$

From $(3.b)$:

$$
\begin{equation*}
RHS=\left(\overline{y_{i}} -\beta ^{\top }\overline{x_{i}}\right) +h_{\beta }( x_{it}) =\overline{y_{i}} -\beta ^{\top }\overline{x_{i}} +\beta ^{\top } x_{it}
\end{equation*}
$$

Therefore, we conclude that both method 1 and method 2 are the same in doing the actual prediction of $y_{it}$.
