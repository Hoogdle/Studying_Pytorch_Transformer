import numpy as np

x = np.array(
    [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],
     [11],[12],[13],[14],[15],[16],[17],[18],[19],[20],
     [21],[22],[23],[24],[25],[26],[27],[28],[29],[30]]
)
y = np.array(
    [[0.94],[1.98],[2.88],[3.92],[3.96],[4.55],[5.64],[6.3],[7.44],[9.1],
     [8.46],[9.5],[10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],
     [17.46],[19.8],[18],[21.34],[22],[22.5],[24.57],[26.04],[21.6],[28.8]]
)

weight = 0.0
bias = 0.0
learning_rate = 0.005

for epoch in range(10000):
    y_hat = weight * x + bias
    cost = ((y-y_hat)**2).mean()

    weight = weight - learning_rate * ((y_hat-y)*x).mean()
    bias = bias - learning_rate * (y_hat-y).mean() #편향으로 미분하면 미분이 살짝 달라짐

    if (epoch + 1) % 1000 == 0:
        print(f'Epoc : {epoch+1:4d}, Weight : {weight:.3f}, Bias : {bias:.3f}, Cost : {cost:.3f}')


# Epoc : 1000, Weight : 0.872, Bias : -0.290, Cost : 1.377
# Epoc : 2000, Weight : 0.877, Bias : -0.391, Cost : 1.373
# Epoc : 3000, Weight : 0.878, Bias : -0.422, Cost : 1.372
# Epoc : 4000, Weight : 0.879, Bias : -0.432, Cost : 1.372
# Epoc : 5000, Weight : 0.879, Bias : -0.435, Cost : 1.372
# Epoc : 6000, Weight : 0.879, Bias : -0.436, Cost : 1.372
# Epoc : 7000, Weight : 0.879, Bias : -0.436, Cost : 1.372
# Epoc : 8000, Weight : 0.879, Bias : -0.436, Cost : 1.372
# Epoc : 9000, Weight : 0.879, Bias : -0.436, Cost : 1.372
# Epoc : 10000, Weight : 0.879, Bias : -0.436, Cost : 1.372
        
# 하이퍼파라미터 값이 적절 한다면 더 많은 학습이 필요하지 않는다.
# learning_rate == 0.001 이였다면 오차가 더 천천히 감소한다.
# learning_rate == 0.006 이였다면 오차가 감소하지만 가중치나 편향이 지그재그 형태로 움직이게 된다.
# learning_rate == 0.007 이였다면 오차가 점점 증가하며 편향이 적절한 값에서 멀어지게 된다.
# 가중치와 편향의 값이 매우 큰 경우는 매우 많은 학습을 요구하게 된다.
        
