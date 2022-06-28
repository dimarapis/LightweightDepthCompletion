import argparse

def decnet_args_parser():
    #parser adapted from Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image (https://github.com/fangchangma/sparse-to-dense)
    parser = argparse.ArgumentParser(description='Decnet-argument-parser')
    #parser.add_argument('--task',
    #                    type=str,
    #                    default="depth-estimation",
    #                    choices=["decnet-completion", "decnet-estimation"],
    #                    help='choose a task: decnet-completion or decnet-estimation'
    #                    )
    parser.add_argument('--network-model',
                        type=str,
                        default="GuideDepth",
                        choices=["GuideDepth", "SparseGuidedDepth", "SparseAndRGBGuidedDepth", "ENET2021"],
                        help='choose a model'
                        )
    parser.add_argument('--pretrained',
                        type=bool,
                        default=False,
                        help='Choose between loading a pretrained model or training from scratch'
                        )
    parser.add_argument('-m','--message',
                        default="",
                        help='optional message for details on training or eval'
                        )
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        choices=["train", "eval", "batchpass"],
                        help='choose a mode: training,evaluation,batchpass'
                        )
    parser.add_argument('--dataset',
                        type=str,
                        default="nn",
                        choices=["nn", "nyuv2", "kitti", "isaacsim"],
                        help='choose a dataset: nn, nyuv2, kitti, isaacsim'
                        )
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--criterion',
                        metavar='LOSS',
                        default='l2')
                        #choices=criteria.loss_names,
                        #help='loss function: | '.join(criteria.loss_names) +
                        #' (default: l2)')
    parser.add_argument('--batch-size',
                        default=8,
                        type=int,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--learning-rate',
                        default=1e-06,
                        type=float,
                        metavar='LR',
                        help='initial learning rate (default 1e-05 in PENET 1e-04 in guided)')
    parser.add_argument('--weight-decay',
                        default=1e-05,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--print-freq',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-folder',
                        default='/datasets',
                        type=str,
                        metavar='PATH',
                        help='data folder (default: none)')
    #geometric encoding
    parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                        choices=["std", "z", "uv", "xyz"],
                        help='information concatenated in encoder convolutional layers')
    parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                        choices=[1, 2, 4],
                        help='CSPN++ dilation rate')

    #my args
    parser.add_argument('--val_h',
                        default=352,
                        type=int,
                        #metavar='N',
                        help='height of validation image - centercropped (default: 352)')
    parser.add_argument('--val_w',
                        default=608,
                        type=int,
                        #metavar='N',
                        help='width of validation image - centercropped (default: 608)')     
    parser.add_argument('--train_height',
                        default=352,
                        type=int,
                        #metavar='N',
                        help='height of training image')
    parser.add_argument('--train_width',
                        default=608,
                        type=int,
                        #metavar='N',
                        help='width of training image')     
    parser.add_argument('--min_depth_eval',
                        default=0.1,
                        type=float,
                        #metavar='N',
                        help='minimum depth to count for when validation (default: 0.1 meters)')         
    parser.add_argument('--max_depth_eval',
                        default=80.0,
                        type=float,
                        #metavar='N',
                        help='maximum depth to count for when validation (default: 70.0 meters)')
    
    #base args
    parser.add_argument('--kitti_crop', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='eigen_crop',
                        type=str,
                        #metavar='N',
                        help='which crop to follow for validationmetrics')
    parser.add_argument('--train_datalist', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        #default='datalist_train_nn.list',
                        default='train_dim_kitti.list',
                        #default='8batch_dim_kitti.list',
                        type=str,
                        #required=True,
                        help='list file to use to load dataset')
    parser.add_argument('--val_datalist', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        #default='datalist_test_nn.list',
                        default='val_dim_kitti.list',
                        #default='8batch_dim_kitti.list',
                        type=str,
                        #required=True,
                        help='list file to use to load dataset')
    parser.add_argument('--root_folder', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        #default='data/nn_dataset/',
                        default='data/kitti_dataset/val_selection_cropped/',
                        type=str,
                        #required=True,
                        help='Root folder where the list and data is located')
    parser.add_argument('--torch_mode', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='pytorch',
                        choices=["pytorch", "tensorrt"],
                        type=str,
                        #required=True,
                        help='Decide if you run with pytorch or tensorrt')
    
    #WANDB args
    parser.add_argument('--wandblogger', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default=False,
                        type=bool,
                        help='Parameter to decide if we store wandb or not')
    
    parser.add_argument('--project', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='depth',
                        type=str,
                        #metavar='N',
                        help='Project name when saving in wandb')
    parser.add_argument('--entity', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='wandbdimar',
                        type=str,
                        #required=True,
                        help='Entity name when saving in wandb')
    
    
    #args parser
    args = parser.parse_args()
    
    
    return args