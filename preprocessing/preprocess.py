import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
import gc

def loadData(X_train_path, y_train_path, X_test_path, y_test_path):

    X_train = sio.loadmat(X_train_path)['pavia']
    y_train = sio.loadmat(y_train_path)['pavia_gt']

    X_test = sio.loadmat(X_test_path)['paviaU']
    y_test = sio.loadmat(y_test_path)['paviaU_gt']

    return X_train, y_train, X_test, y_test

def gen_random_pix_samples(X, y, window_size=25, ns_per_class=50): 
    
    """
    -----------------------
    grabs random pixel samples (stratified) 
    Specific for 3D-CNN-LSTM-AE model. Training chips are 
    images, where the centre pixel is the label (class id). 

    Obtains random pixel coordinates for each class in the image. 
    At each pixel coordinate, the label-ID is obtained, and a window
    is drawn around each pixel - extracted as a sample. 
    -----------------------
    :param X: (np.ndarray) representing hyperspectral image (rows, cols, bands)
    :param y: (np.array) representing 2d labeled image (rows, cols) 
    :param window_size: (int) window size (should be odd number)   
    :param ns_per_class: (int) number of samples to generate per class   

    outputs samples saved to directory 
    """

    bfr = int((window_size-1)/2) # buffer  

    # pad bands and rows so the centre pixel labels 
    # can be taken from the edge of the unpadded data 
    # shaped (rows, cols, bands)  
    X = np.pad(X, ( (bfr,bfr), # start,end pad (rows) 
                    (bfr,bfr), # start,end pad (col) 
                    (0, 0)), #start, end pad (band)
                mode='constant', 
                constant_values=0
            )

    labels, counts = np.unique(y, return_counts=True)
   
    # assumes first class == 0 (no data/label)
    # samples per class
    # ns_per_class = np.min(counts[1:])

    for label in labels: 
        if label !=0: 
            print(f'generating training data for label ID {label}')
            row_idx, col_idx = np.where(y==label)

            # shuffle row, col idx 
            np.random.seed(161)
            np.random.shuffle(row_idx)

            np.random.seed(161)
            np.random.shuffle(col_idx)

            # get number up to ns per class 
            for i, j in enumerate(zip(row_idx, col_idx)): 
                if i < ns_per_class: 

                # grab centre pixel as label from y 
                # buffer and crop area around pixel - assign as X sample 
                    y_train = y[j[0],j[1]]
                        
                    # corrected centre pixel coord
                    # accounts for padding added to X 
                    crw, ccl = (j[0]+bfr, j[1]+bfr)
    
                    # grab pixel box around centre coord 
                    X_train = X[crw-bfr:crw+bfr+1,
                                ccl-bfr:ccl+bfr+1,:]

                    ### Visualize samples ###     
                    # plt.title(f'Class ID {str(y_train)}')
                    # plt.plot(bfr, bfr,'b+', markersize=20)
                    # plt.imshow((X_train[:,:,[50,30,17]]/8000)*1.5, vmin=0, vmax=1)
                    # plt.show()

            ## split into train, validation, and testing chips 
            ## save chips into respective directories 
            

                # exit()

    return 


X_train, y_train, X_test, y_test = loadData(r'C:\Users\agraham1\Documents\PythonScripts\Hyperspectral_AI_Benchmarks\datasets\Pavia\images\PaviaC.mat', 
                r'C:\Users\agraham1\Documents\PythonScripts\Hyperspectral_AI_Benchmarks\datasets\Pavia\ground_truth\PaviaC_gt.mat', 
                r'C:\Users\agraham1\Documents\PythonScripts\Hyperspectral_AI_Benchmarks\datasets\Pavia\images\PaviaU.mat', 
                r'C:\Users\agraham1\Documents\PythonScripts\Hyperspectral_AI_Benchmarks\datasets\Pavia\ground_truth\PaviaU_gt.mat', 

)

gen_random_pix_samples(X_train, y_train, window_size=25, ns_per_class=2)


