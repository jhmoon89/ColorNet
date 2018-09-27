import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import imgaug as ia
from imgaug import augmenters as iaa


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(image, noise_typ):
    if noise_typ == "gaussian":
        # np.array([103.939, 116.779, 123.68])
        mean = 0
        var = 0.01 * 255.0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "salt&pepper":
        s_vs_p = 0.5
        amount = 0.05
        out = image.copy()

        # print image.size
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, np.max((i - 1, 1)), int(num_salt)) for i in image.shape]
        out[coords] = 255.0

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, np.max((i - 1, 1)), int(num_pepper)) for i in image.shape]
        out[coords] = 0.0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(np.abs(image) * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        gauss = np.random.randn(*image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + image * 0.1 * gauss
        return noisy
    else:
        return image.copy()


def imageAugmentation(inputImages):
    noiseTypeList = ['gaussian', 'salt&pepper', 'poisson', 'speckle']
    random.shuffle(noiseTypeList)
    select = np.random.randint(0, 2, len(noiseTypeList))
    for i in range(len(noiseTypeList)):
        if select[i] == 1:
            inputImages = noisy(image=inputImages, noise_typ=noiseTypeList[i])
    return inputImages


def cvt_image_to_z(img, gamut): # input image : Lab channel (batch_num, frame_num, H, W, 3)
    gamut_range = np.argwhere(gamut != 0)
    color_bin_num = len(gamut_range)
    # print color_bin_num

    # cv2.imshow('img', cv2.resize(cv2.cvtColor(img, cv2.COLOR_Lab2BGR), img_size))

    # ab channel to bin
    img = (img[..., 1:] / 8).astype('uint8')
    img_size = np.shape(img)[-3:-1]

    tic = time.time()
    img_tile = np.tile(np.expand_dims(img, -2), [1, 1, 1, 1, color_bin_num, 1]).astype('float32')
    gamut_range_tile = (np.tile(
        np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(gamut_range, 0), 0), 0), 0),
        [np.shape(img)[0], np.shape(img)[1], img_size[0], img_size[1], 1, 1])).astype('float32')
    toc = time.time()
    print ("tile gamut: "), toc-tic

    tic = time.time()
    sigma = 0.5
    dist = np.exp(-1 * np.sum(np.square(img_tile - gamut_range_tile), -1) / (2 * sigma * sigma)) / \
           (np.sqrt(2 * np.pi * sigma * sigma))
    dist_sorted = np.sort(dist, -1)
    toc = time.time()
    print ("gaussian: "), toc-tic

    # smoothing by Gaussian Filter
    tic = time.time()
    mask = (dist >= np.tile(np.expand_dims(dist_sorted[..., -5], -1), [1, 1, color_bin_num])).astype('float32')
    toc = time.time()
    print ("mask: "), toc-tic
    tic = time.time()
    dist_truncated = dist * mask
    toc = time.time()
    print ("matrix mul: "), toc-tic
    tic = time.time()
    value_sum = np.tile(np.expand_dims(np.sum(dist_truncated, -1), -1), [1, 1, color_bin_num])
    dist_truncated = dist_truncated / value_sum
    toc = time.time()
    print ("final: "), toc-tic

    return dist_truncated

    # print np.sort(dist_truncated, -1)[0][0][-5:]
    # print np.sort(dist_truncated2, -1)[0][0][-5:]

    # ind = np.argsort(dist, -1)[4]
    # th = np.take(dist, ind)


    # cv2.imshow('class', ind)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ind = np.argsort(dist, -1)[..., 5:]
    # ind2 = np.unravel_index(ind, np.shape(dist))
    # dist[ind2] = 0
    # # print dist
    # dist[dist < 1e-9] = 0
    # print np.argwhere(dist[100, 100] != 0)

def cvt_z_to_image(Z, gamut, T=0.38): # H function
    Interporlated_Z = np.exp(np.log(Z.astype('float32') + 1e-7) / (T + 1e-7)).astype('uint8').astype('float32')
    # Interporlated_Z = Z - np.exp(T)
    # print np.shape(Interporlated_Z)
    a_list = np.expand_dims(np.expand_dims(np.argwhere(gamut != 0)[:, 0] * 16, 0), 0)
    b_list = np.expand_dims(np.expand_dims(np.argwhere(gamut != 0)[:, 1] * 16, 0), 0)

    # print np.shape(a_list)
    Height = np.shape(Interporlated_Z)[0]
    Width = np.shape(Interporlated_Z)[1]

    output_a = np.sum(Interporlated_Z * np.tile(a_list, [Height, Width, 1]), -1)
    output_b = np.sum(Interporlated_Z * np.tile(b_list, [Height, Width, 1]), -1)

    return (np.stack([output_a, output_b], -1)).astype('uint8')



'''https://github.com/aleju/imgaug'''


def imgAug(inputImage, crop=True, flip=True, gaussianBlur=True, channelInvert=True, brightness=True, hueSat=True):
    augList = []
    if crop:
        augList += [iaa.Crop(px=(0, 16))]  # crop images from each side by 0 to 16px (randomly chosen)
    if flip:
        augList += [iaa.Fliplr(0.5)]  # horizontally flip 50% of the images
    if gaussianBlur:
        augList += [iaa.GaussianBlur(sigma=(0, 3.0))]  # blur images with a sigma of 0 to 3.0
    if channelInvert:
        augList += [iaa.Invert(0.05, per_channel=True)]  # invert color channels
    if brightness:
        augList += [iaa.Add((-10, 10), per_channel=0.5)]  # change brightness of images (by -10 to 10 of original value)
    if hueSat:
        augList += [iaa.AddToHueAndSaturation((-20, 20))]  # change hue and saturation
    seq = iaa.Sequential(augList)
    # seq = iaa.Sequential([
    #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    #     iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    #     iaa.Invert(0.05, per_channel=True),  # invert color channels
    #     iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
    #     iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    # ])

    image_aug = seq.augment_image(inputImage)
    return image_aug

########### Rebalance Factaor ############
# gamut = np.load("/ssdubuntu/color/gamut.npy")
# gamut_normalize = gamut / np.sum(gamut)
# ind_mask = np.where(gamut_normalize != 0)
# Q = np.shape(ind_mask)[1]
# gamut_blur = cv2.GaussianBlur(gamut_normalize, (5, 5), 5)
# gamut_smooth = np.zeros_like(gamut)
# gamut_smooth[ind_mask] = 1.0 / (0.5 * (gamut_blur[ind_mask] + 1 / float(Q)))
# gamut_smooth = gamut_smooth / np.sum(gamut_smooth)
# weight_list = gamut_smooth[ind_mask]
# np.save("/ssdubuntu/color/rebalance_weight.npy", weight_list)
###########################################


# gamut = np.load("/ssdubuntu/color/gamut.npy")
# # print len(np.argwhere(gamut != 0))
# # weight_list = np.load("/ssdubuntu/color/rebalance_weight.npy")
# # print np.shape(weight_list)
#
# img_name = "/ssdubuntu/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00001003/000000.JPEG"
# img = cv2.cvtColor(cv2.resize(cv2.imread(img_name, cv2.IMREAD_COLOR), (256, 256)), cv2.COLOR_BGR2Lab)
# data = cvt_image_to_z(img, gamut)
#
# # max_cls = (np.argmax(data, -1) / np.max(data) * 255).astype('uint8')
# z_test = cvt_z_to_image(data, gamut, 0.99)
# final_img = np.stack([img[..., 0], z_test[..., 0], z_test[..., 1]], -1)
#
# # print np.shape(final_img)
# # print np.shape(z_test)
# # print img - final_img
#
# cv2.imshow('org', cv2.cvtColor(img, cv2.COLOR_Lab2BGR))
# cv2.imshow('image', cv2.cvtColor(final_img, cv2.COLOR_Lab2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
