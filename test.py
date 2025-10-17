import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
N = 100
half_n = N // 2

r = 10
x0_gt, y0_gt = 2, 3 # Center
s = r / 16
t = np.random.uniform(0, 2*np.pi, half_n)
n = s*np.random.randn(half_n)
x, y = x0_gt + (r + n) * np.cos(t), y0_gt + (r + n) * np.sin(t)
X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

s = 1.
m, b = -1, 2
x = np.linspace(-12, 12, half_n)
y = m*x + b + s*np.random.randn(half_n)
X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

X = np.vstack((X_circ, X_line)) # All points

def RANSAC(data, num_samples, params_from_samples, error_given_params, inlier_thresh, iterations, min_inlier_count = 30):
    '''
    data is an array of tuples, each of num_samples elements
    min_inlier_count must be less than len(data)
    '''
    max_inlier_count = 0
    best_samples = None
    best_inliers = None
    best_params = None

    for _ in range(iterations):
        # Generate num_samples random integers between 0 and len(data), use these as indices to extract num_samples random samples from the data 
        sample_indices = np.random.randint(0, len(data), num_samples)
        samples = data[sample_indices]

        # Compute the model parameters using the above sampled points
        params = params_from_samples(samples)

        # Count the number of inliers based on the model described by the above parameters, and the specified threshold
        inliers = data[np.abs(error_given_params(params, data)) < inlier_thresh]
        
        # Keep track of which parameters produced the most inliers
        if ((len(inliers) > max_inlier_count)) and (len(inliers) >= min_inlier_count):
            max_inlier_count = len(inliers)
            best_samples = samples
            best_inliers = inliers
            best_params = params

    return best_samples, best_inliers, best_params

def line_params_from_samples(samples):
    x1, y1 = samples[0]
    x2, y2 = samples[1]

    theta = np.atan2((x2 - x1), (y1 - y2))
    d = x1*np.cos(theta) + y1*np.sin(theta)

    return (theta, d)

def line_error(params, data):
    theta, d = params
    x, y = data[:, 0], data[:, 1]

    return x*np.cos(theta) + y*np.sin(theta) - d

def circle_params_from_samples(samples):
    x1, y1 = samples[0]
    x2, y2 = samples[1]
    x3, y3 = samples[2]

    A1 = 2*(x2 - x1)
    A2 = 2*(x3 - x1)
    B1 = 2*(y2 - y1)
    B2 = 2*(y3 - y1)
    C1 = x2**2 - x1**2 + y2**2 - y1**2
    C2 = x3**2 - x1**2 + y3**2 - y1**2

    h = (B2*C1 - B1*C2) / (A1*B2 - A2*B1)
    k = (A2*C1 - A1*C2) / (A2*B1 - A1*B2)
    r = np.sqrt((x1 - h)**2 + (y1 - k)**2)

    return (h, k, r)

def circle_error(params, data):
    h, k, r = params
    x, y = data[:, 0], data[:, 1]

    return r - np.sqrt((x - h)**2 + (y - k)**2)


print ("fitting line first then circle")
line_best_samples, line_best_inliers, (theta_0, d_0) = RANSAC(X, 2, line_params_from_samples, line_error, 1, 1000)
a_0 = np.cos(theta_0)
b_0 = np.sin(theta_0)

remaining_points = np.array([x for x in X if x not in line_best_inliers])

circle_best_samples, circle_best_inliers, (h_0, k_0, r_0) = RANSAC(remaining_points, 3, circle_params_from_samples, circle_error, 1, 1000)

xbefore = X

print ("RANSAC Estimates")
print()

print ("Line: ax + by = d")
print ("[with a^2 + b^2 = 1, i.e., a = cos(theta), b = sin(theta)]")
print()

print ("theta = {:0.2f} deg,".format(theta_0*180/np.pi), "d = {:0.2f}".format(d_0))
print ()

print ("{:0.2f}x + {:0.2f}y = {:0.2f}".format(a_0, b_0, d_0))

print("-" * 50)

print ("Circle: (x - h)^2 + (y - k)^2 = r^2")
print()

print ("h = {:0.2f},".format(h_0), "k = {:0.2f},".format(k_0), "r = {:0.2f}".format(r_0))
print ()

print ("(x - {:0.2f})^2 + (y - {:0.2f})^2 = {:0.2f}^2".format(h_0, k_0, r_0))

print ("------------------"*5)

fig, ax = plt.subplots(1, 2, figsize=(8, 16))

ax[0].scatter(X_line[:, 0], X_line[:, 1], label='Line')
ax[0].scatter(X_circ[:, 0], X_circ[:, 1], label='Circle')

circle_gt = plt.Circle((x0_gt, y0_gt), r, color='g', fill=False, label='Ground truth circle')
ax[0].add_patch(circle_gt)

circle_ransac = plt.Circle((h_0, k_0), r_0, color='b', fill=False, label='RANSAC Circle')
ax[0].add_patch(circle_ransac)

x_min, x_max = ax[0].get_xlim()
x_ = np.array([x_min, x_max])

print ("line best samples", line_best_samples)
print ("circle best samples", circle_best_samples)

y_ = m*x_ + b

ax[0].plot(x_, y_, color='m', label='Ground truth line')

y_0 = -(a_0/b_0)*x_ + (d_0/b_0)
ax[0].plot(x_, y_0, color='r', label='RANSAC Line')

ax[0].plot(line_best_inliers[:, 0], line_best_inliers[:, 1], '+', color="#27147C", markersize=10, label='Line Best Inliers 1')
ax[0].plot(circle_best_inliers[:, 0], circle_best_inliers[:, 1], '+', color="#47B3B1", markersize=10, label='Circle Best Inliers 1')

ax[0].plot(line_best_samples[:, 0], line_best_samples[:, 1], '*', color="#FF03D5", markersize=10, label='Line Best Samples 1')
ax[0].plot(circle_best_samples[:, 0], circle_best_samples[:, 1], '*', color="#000000", markersize=10, label='Circle Best Samples 1')

ax[0].legend()












print ("fitting circle first then line")

for xbef in xbefore:
    if xbef not in X:
        print ("X CHANGED!")
print ("checked all xbefore and X")

circle_best_samples_, circle_best_inliers_, (h_2, k_2, r_2) = RANSAC(X, 3, circle_params_from_samples, circle_error, 1, 1000)

print ("Circle Estimates:")
print ("h = %.2f, k = %.2f, r = %.2f" % (h_2, k_2, r_2))

remaining_points_ = np.array([x for x in X if x not in circle_best_inliers_])

line_best_samples_, line_best_inliers_, (theta_2, d_2) = RANSAC(remaining_points_, 2, line_params_from_samples, line_error, 1, 1000)
a_2 = np.cos(theta_0)
b_2 = np.sin(theta_0)

print ("Line Estimates:")
print ("a = %.2f, b = %.2f, d = %.2f" % (a_2, b_2, d_2))

# Raw Data
ax[1].scatter(X_line[:, 0], X_line[:, 1], label='Line')
ax[1].scatter(X_circ[:, 0], X_circ[:, 1], label='Circle')

# Circle
circle_gt_2 = plt.Circle((x0_gt, y0_gt), r, color='g', fill=False, label='Ground truth circle 2')
ax[1].add_patch(circle_gt_2)

circle_ransac_ = plt.Circle((h_2, k_2), r_2, color='b', fill=False, label='RANSAC Circle 2')
ax[1].add_patch(circle_ransac_)

ax[1].plot(circle_best_inliers_[:, 0], circle_best_inliers_[:, 1], '+', color="#47B3B1", markersize=10, label='Circle Best Inliers 2')

# Line

x_min, x_max = ax[1].get_xlim()
x_1 = np.array([x_min, x_max])

y_1 = m*x_1 + b
ax[1].plot(x_1, y_1, color='m', label='Ground Truth Line 2')

y_2 = -(a_2/b_2)*x_1 + (d_2/b_2)
ax[1].plot(x_1, y_2, color='r', label='RANSAC Line 2')

print ("line best samples", line_best_samples_)
print ("circle best samples", circle_best_samples_)

ax[1].plot(line_best_inliers_[:, 0], line_best_inliers_[:, 1], '+', color="#27147C", markersize=10, label='Line Best Inliers 2')

ax[1].plot(line_best_samples_[:, 0], line_best_samples_[:, 1], '*', color="#FF03D5", markersize=10, label='Line Best Samples 2')
ax[1].plot(circle_best_samples_[:, 0], circle_best_samples_[:, 1], '*', color="#000000", markersize=10, label='Circle Best Samples 2')

ax[1].legend()
plt.show()
