import os
import glob
  
def make_dataset_list_nn_or_kitti(file_name,root_folder,rgb_path,depth_path,gt_path):
    rgb_files = glob.glob(os.path.join(root_folder,rgb_path,'*'))
    print(rgb_files)
    nn_val = open(os.path.join(root_folder,file_name), "w")
    for file in rgb_files:
        rgb_name = file
        sparse_name = rgb_name.replace(rgb_path,depth_path,2)
        gt_name = rgb_name.replace(rgb_path,gt_path,2)
        nn_val.write(rgb_name + ' ' + sparse_name + ' ' + gt_name + '\n')
    print(rgb_name,sparse_name,gt_name)
    nn_val.close()
    
    
#Datalist maker for nn data
make_dataset_list_nn_or_kitti(file_name='datalist_train_nn.list',
                              root_folder='data/nn_dataset/train',
                              rgb_path='rgb_cropped',
                              depth_path='depth_cm_cropped',
                              gt_path='pcl_cm_cropped')


#Datalist maker for kitti data
'''
make_dataset_list_nn_or_kitti(file_name='datalist_val_kitti.list',
                              root_folder='data/kitti_dataset/val_selection_cropped',
                              rgb_path='image',
                              depth_path='velodyne_raw',
                              gt_path='groundtruth_depth')

'''
'''NEED TO FIX THIS
def make_dataset_list_sim_warehouse(file_name,root_folder,rgb_path,depth_path,gt_path):
    rgb_files_1 = glob.glob(os.path.join(root_folder,rgb_path,'*'))
    print(rgb_files)
    nn_val = open(os.path.join(root_folder,file_name), "w")
    for file in rgb_files:
        rgb_name = file
        sparse_name = rgb_name.replace(rgb_path,depth_path,2)
        gt_name = rgb_name.replace(rgb_path,gt_path,2)
        nn_val.write(rgb_name + ' ' + sparse_name + ' ' + gt_name + '\n')
    print(rgb_name,sparse_name,gt_name)
    nn_val.close()



#Datalist maker for simulated warehouse data 
make_dataset_list_sim_warehouse(file_name='datalist_sim_warehouse.list',
                                root_folder='data/sim_warehouse/test',
                                rgb_path='rgb_cropped',
                                depth_path='depth_cm_cropped',
                                gt_path='pcl_cm_cropped')
'''


