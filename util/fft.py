import numpy as np
import matplotlib.pyplot as plt
from util.coil import *
import time

def mosaic(images):
    fig = plt.figure(figsize=(16, 12), dpi=80)
    l = images.shape[-1]
    w = (l//5)+1
    for idx in range(l):
        ax = fig.add_subplot(w, int(np.floor(l/w))+1, idx+1)
        ax.imshow(np.abs(images[...,idx]), cmap = "gray")
    plt.show()
        
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import clear_output
def slider(images):
    images = np.abs(images)
    if len(images.shape) > 3:
            images = rsos(images)
    def show(idx):
        _, ax = plt.subplots(1,1)
        plt.imshow(images[...,int(idx)], cmap="gray")
    interact(show, idx = widgets.FloatSlider(value=0,
                                                   min=0,
                                                   max=images.shape[-1]-1,
                                                   step=1))
    
def plot1d(traj):
    val = 0. # this is the value where you want the data to appear on the y-axis.
    ar = traj # just as an example array
    plt.plot(ar, np.zeros_like(ar) + val, '.')
    plt.show()
def show(image):
    if len(image.shape) > 2: 
        mosaic(image)
    else: 
        plt.imshow(np.abs(image), cmap="gray")    
        plt.show()
    
def showrsos(image):
    if len(image.shape) > 3: 
        mosaic(rsos(image))
    else: 
        plt.imshow(np.abs(rsos(image)), cmap="gray")
    
    

def conjugate(data):
    return fft2c(np.conj(ifft2c(data)))

def showc(image):
    plt.figure()
    tf = plt.imshow(np.abs(image),cmap='jet')
    plt.colorbar(tf, fraction=0.046, pad=0.04)
    plt.show()
    
def ifft(F, axis = (0)):
    x = (axis)
    tmp0 = np.fft.ifftshift(F, axes=(x,))
    tmp1 = np.fft.ifft(tmp0, axis = x)
    f = np.fft.fftshift(tmp1, axes=(x,))
    return f * F.shape[x]

def fft(f, axis = (0)):
    x = (axis)
    tmp0 = np.fft.fftshift(f, axes=(x,))
    tmp1 = np.fft.fft(tmp0, axis = x)
    F = np.fft.ifftshift(tmp1, axes=(x,))
    return F / f.shape[x]
def fft1c(f, axis = (0)):
    x = (axis)
    tmp0 = np.fft.fftshift(f, axes=(x,))
    tmp1 = np.fft.fft(tmp0, axis = x)
    F = np.fft.ifftshift(tmp1, axes=(x,))
    return F / f.shape[x]
def ifft2c(F, axis = (0,1)):
    x,y = (axis)
    tmp0 = np.fft.ifftshift(np.fft.ifftshift(F, axes=(x,)), axes=(y,))
    tmp1 = np.fft.ifft(np.fft.ifft(tmp0, axis = x), axis = y)
    f = np.fft.fftshift(np.fft.fftshift(tmp1, axes=(x,)), axes=(y,))
    return f * F.shape[x]* F.shape[y] 

def fft2c(f, axis = (0,1)):
    x,y = (axis)
    tmp0 = np.fft.fftshift(np.fft.fftshift(f, axes=(x,)), axes=(y,))
    tmp1 = np.fft.fft(np.fft.fft(tmp0, axis = x), axis = y)
    F = np.fft.ifftshift(np.fft.ifftshift(tmp1, axes=(x,)), axes=(y,))
    return F / f.shape[x]/ f.shape[y]

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def siemens_readout_trim(F):
    [ny,nx,] = F.shape[:2]
    f = ifft(F,1)
    f = f[:,nx//4:nx-nx//4,...]
    return fft(f, 1)

def srsos(images,coilaxis = 2):
    images = fft2c(images)
    images = siemens_readout_trim(images)
    images = ifft2c(images)
    return np.sqrt(np.sum(np.square(np.abs(images)),axis = coilaxis))

def process_reference(acs):
    l = len(acs.shape)
    tmp = acs
    for i in range(l-1):
        tmp = np.sum(tmp, -1)
    index = np.where(np.abs(tmp)!=0)[0]
    acs = acs[index,:,:]
    return acs