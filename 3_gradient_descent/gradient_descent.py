import numpy as np

def gradient_descent(x,y):
    # initial value: 0
    m_curr = b_curr = 0
    # define the number of iterations
    iterations = 10000
    # n is the length of these data points
    n = len(x)
    # whatever you want (note that the suitable one is let the cost be constant or reduce not to much)
    learning_rate = 0.08 # the smaller the better (0.01 -> 0.1)

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        # val**2 = val^2 (square)
        # if the cost is reduce in each of these steps => good
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        # calculate m derive and b derive
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr, cost, i))


# we want to derive the best fit line or an equation using m and b

# use numpy array instead of simple python list
# (cuz matrix multiplication is very convenient, also np array is faster)

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)