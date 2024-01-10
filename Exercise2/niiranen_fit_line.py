from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np

# Linear solver
def mylinfit(x, y):
    n = len(y)
    xy = [x*y for x,y in zip(x,y)]
    xx = [x*x for x,x in zip(x,x)]
    b = ((sum(xy)*sum(x))/sum(xx)-sum(y))*1/((sum(x)-sum(xx)*n))/sum(xx)
    a = (sum(xy)-b*sum(x))/sum(xx)
    return a, b

# Separate x and y coordinates and return the vectors
def separatexy(xy):
    i = 0
    x = [0] * len(xy)
    y = [0] * len(xy)
    while(i < len(xy)):
        x[i] = xy[i][0]
        y[i] = xy[i][1]
        i += 1
    return x, y

def main():
    # Plot empty coordinate system
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')
    ax.spines['bottom'].set_position('zero') # type: ignore
    ax.spines['left'].set_position('zero') # type: ignore
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Select N points from the coordinate system
    xy = plt.ginput(-1, mouse_stop=MouseButton.RIGHT)
    x, y = separatexy(xy)

    # Fit linear model to data points 
    a, b = mylinfit(x, y)
    plt.show() 
    plt.plot(x, y, 'kx')
    xp = np.arange(-5, 5, 0.1)
    plt.plot(xp, a*xp+b, 'r-')
    print (f"My fit : a={a} and b={b}")
    plt.show( )

main()
