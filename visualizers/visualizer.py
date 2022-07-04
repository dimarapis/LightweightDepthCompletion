import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral
cmap3 = plt.cm.turbo

def validcrop(img):
    ratio = 256/1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h-int(ratio*w):, :]

def depth_colorize_np(depth):
    depth = np.squeeze(depth.data.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap3(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    #print(np.min(depth),np.max(depth))
    depth = 255 * cmap3(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def rgb_visualizer(image):
    rgb = np.squeeze(image[0, ...].data.cpu().numpy())
    #print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    return rgb.astype('uint8')
    #return rgb

def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = 255 * cmap2(feature)[:, :, :3]
    return feature.astype('uint8')

def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = 255 * mask
    return mask.astype('uint8')

def merge_into_row(ele, pred, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    elif 'g' in ele:
        g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert('RGB'))
        img_list.append(g)
    if 'd' in ele:
        img_list.append(preprocess_depth(ele['d'][0, ...]))
        img_list.append(preprocess_depth(pred[0, ...]))
    if extrargb is not None:
        img_list.append(preprocess_depth(extrargb[0, ...]))
    if predrgb is not None:
        predrgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        predrgb = np.transpose(predrgb, (1, 2, 0))
        #predrgb = predrgb.astype('uint8')
        img_list.append(predrgb)
    if predg is not None:
        predg = np.squeeze(predg[0, ...].data.cpu().numpy())
        predg = mask_vis(predg)
        predg = np.array(Image.fromarray(predg).convert('RGB'))
        #predg = predg.astype('uint8')
        img_list.append(predg)
    if extra is not None:
        extra = np.squeeze(extra[0, ...].data.cpu().numpy())
        extra = mask_vis(extra)
        extra = np.array(Image.fromarray(extra).convert('RGB'))
        img_list.append(extra)
    if extra2 is not None:
        extra2 = np.squeeze(extra2[0, ...].data.cpu().numpy())
        extra2 = mask_vis(extra2)
        extra2 = np.array(Image.fromarray(extra2).convert('RGB'))
        img_list.append(extra2)
    if 'gt' in ele:
        img_list.append(preprocess_depth(ele['gt'][0, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_image_torch(rgb, filename):
    #torch2numpy
    #rgb = validcrop(rgb)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    #print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype('uint8')
    image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_depth_as_uint16png(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)

def save_depth_as_uint16png_upload(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def save_depth_as_uint8colored(img, filename):
    #from tensor
    #img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = depth_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
    
def wandb_image_prep(image, pred):

    depth = np.squeeze(pred.cpu().detach().numpy())
    depth = depth_colorize(depth)
    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
    
    rgb = np.squeeze(image.cpu().detach().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))
    #rgb = rgb.astype('uint8')
    
    return rgb, depth
    #cv2.imwrite(filename, img)
    #wandb_image_prep(image,pred) 
    
def wandb_image_prep_refined(image, pred, refined_pred):

    depth = np.squeeze(pred.cpu().detach().numpy())
    depth = depth_colorize(depth)
    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
    
    refined_depth = np.squeeze(refined_pred.cpu().detach().numpy())
    refined_depth = depth_colorize(refined_depth)
    refined_depth = cv2.cvtColor(refined_depth, cv2.COLOR_RGB2BGR)
    

    rgb = np.squeeze(image.cpu().detach().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))
    #rgb = rgb.astype('uint8')
    
    return rgb, depth, refined_depth
    #cv2.imwrite(filename, img)
    #wandb_image_prep(image,pred)

def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    if(normalized==False):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if(colored==True):
        img = 255 * cmap(img)[:, :, :3]
    else:
        img = 255 * img
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def depth_histogram(depth_prediction):
    # Depth prediction
    img = np.squeeze(depth_prediction.data.cpu().numpy())
    
    # find frequency of pixels in range 0-255
    histr = cv2.calcHist([img],[0],None,[100],[0,100])
    
    return histr

def save_depth_prediction(pred,image):
    save_depth_as_uint16png_upload(pred, 'test_result.png')
    save_depth_as_uint8colored(pred, 'test_results_colorized.png') 
    save_image_torch(image,'test_rgb_image.png')

                                                                                        
def plotter(pred_d,sparse_depth,depth_gt,pred,image):
    fig, axs = plt.subplots(3,2)
    fig.suptitle('Distribution of depth values and corresponding images')
    axs[0,0].title.set_text('Depth prediction')
    axs[0,0].plot(depth_histogram(pred_d))
    axs[1,0].title.set_text('Sparse depth data')
    axs[1,0].plot(depth_histogram(sparse_depth))
    axs[1,0].set_ylim(axs[0,0].get_ylim())
    axs[2,0].title.set_text('Ground truth depth data')
    axs[2,0].plot(depth_histogram(depth_gt))
    axs[2,0].set_ylim(axs[0,0].get_ylim())
    axs[0,1].imshow(depth_colorize_np(pred))
    axs[1,1].imshow(rgb_visualizer(image))
    axs[2,1].imshow(depth_colorize_np(sparse_depth))        
    plt.show()