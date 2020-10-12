import numpy as np
import matplotlib.pyplot as plt

dx = 0.001
x0 = np.arange(0, 1.00, dx)



def u(x, b=0.5, c=0.1):
    return np.exp(-(x-b)**2/(2*c**2))


def cut(x, u):
    diff_x = np.gradient(x)
    start = np.argmax(diff_x <= 0)
    end = len(diff_x) - np.argmax(diff_x[::-1] < 0)

    return x[start:end], u[start:end], start>0


def calc_thresold(x, u, x_range):
    
    all_ind = np.arange(0, len(x))
    
    min_area_diff = np.PINF
    x_opt = None
    i_opt = None
    k_opt = None
    for xp in x_range:
        i = np.argmax( x >= xp)
        j = np.argmax(np.logical_and( x <= xp, all_ind>i))
        k = np.argmax(np.logical_and(x > xp, all_ind>j))

        right_area = poly_area(x[i:j], u[i:j])
        left_area = poly_area(x[j:k], u[j:k])

        area_diff = np.abs(right_area - left_area)
        if area_diff < min_area_diff:
            min_area_diff = area_diff
            x_opt = xp
            i_opt = i
            k_opt = k
    
    mask = np.logical_or( all_ind <= i_opt, all_ind>= k_opt)
    return x_opt, mask


def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


u0 = u(x0) # u(t, x) -- u(t0=0, x)

fig, axs = plt.subplots(5, 1, sharex=True)
axs = axs.ravel()

axs[0].plot(x0, u0) 
axs[0].set_ylabel("u(0, x)")


for i in range(1, 5):
    x = x0 + u0*0.1*i
    axs[i].plot(x, u0) 
    x_cut, u_cut, is_postproc = cut(x, u0)
    if is_postproc:
        # axs[i].plot(x_cut, u_cut) 
        # todo:
        x_opt, mask = calc_thresold(x, u0, x_cut)
        # x_avg = np.average(x_cut)
        # quit()
        # axs[i].axvline(x_opt)
        # axs[i].axvline(x_avg, color='r')
        
        axs[i].plot(x[mask], u0[mask]) 


    tstep = np.round(0.1*i, 2)
    axs[i].set_ylabel(f"u({tstep}, x)")




plt.xlabel("x")

for ax in axs:
    ax.grid()
plt.show()

