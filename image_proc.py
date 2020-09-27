import matplotlib.pyplot as plt
import numpy as np

def load_image(file_path='img/image.jpg'):
    img = plt.imread(file_path)
    print(f'image: {img.shape}')
    return img/255

def gaus_filter(window_size=11, std=3):
    # todo: implement (currently simple average)
    M = np.ones((window_size, window_size))

    for i in range(window_size):
        for j in range(window_size):
            #  = ...np.exp() mean=0, std=std
            pass
    M = M/M.sum() # normalize to sum == 1
    return M


def sobel_filter():
    M = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    return M


def apply_filter(img, M):
    # img: 2D matrix
    filter_width = M.shape[0]
    ws2 = filter_width//2 # or int(filter_width/2)

    filtered_image = np.zeros_like(img)

    for i in range(ws2, img.shape[0]-ws2): #for each row
        for j in range(ws2, img.shape[1]-ws2): #for each column
            filtered_image[i,j] = np.sum(A[i-ws2: i+ws2+1, j-ws2:j+ws2+1]*M)
    
    print(filtered_image.shape)
    return filtered_image


img = load_image()
fig, axs  = plt.subplots(2,2)
axs[0,0].imshow(img)
axs[0,1].imshow(img[:, ::-1]) # inverse order for all columns (step=-1)


A = np.sum(img, axis=2)/3 # sum of all color channels / 2 --> gray scale
print('image in gray scale:', A.shape)
# axs[1,0].imshow(A)
axs[1,0].imshow(A, cmap='gray')

A = A[:, ::-1]
ax_sum = np.sum(A, axis=0) # calc sum for each column (along row axis=0)
print(ax_sum.shape)
axs[1,1].stem(ax_sum)



fig, axs  = plt.subplots(1,3)
axs[0].imshow(A)

M = gaus_filter()
B = apply_filter(A, M)
axs[1].imshow(B)

M = sobel_filter()
C = apply_filter(A, M)
axs[2].imshow(C)


plt.show()
