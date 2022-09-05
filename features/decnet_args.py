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
    parser.add_argument('--networkmodel',
                        type=str,
                        default="DecnetNLSPN",
                        choices=["DecnetModuleSmall","DecnetLateBase","DecnetEarlyBase","DecnetNLSPNSmall","DecnetNLSPN_decoshared","GuideDepth", "SparseGuidedDepth", "SparseAndRGBGuidedDepth", "AuxGuideDepth", "ENET2021","AuxSparseGuidedDepth", "DecnetModule", "DecnetNLSPN"],
                        help='choose a model'
                        )
    parser.add_argument('--resolution',
                        type=str,
                        default="half",
                        choices=["full", "half", "mini"],
                        help='choose a resolution for input images'
                        )
    parser.add_argument('--pretrained',
                        type=bool,
                        default=False,
                        help='Choose between loading a pretrained model or training from scratch'
                        )
    parser.add_argument('--sparsities',
                        type=bool,
                        default=False,
                        help='Choose between varying sparsities or original (500 pts)'
                        )
    parser.add_argument('--augment',
                        type=bool,
                        default=False,
                        help='Choose if you want to augment the data during trianing or no'
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
                        default="nyuv2",
                        choices=["nn", "nyuv2", "kitti", "isaacsim"],
                        help='choose a dataset: nn, nyuv2, kitti, isaacsim'
                        )
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=30,
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
                        default=1e-5,
                        type=float,
                        metavar='lr',
                        help='initial learning rate (default 1e-05 in PENET 1e-04 in guided)')
    parser.add_argument('--weight-decay',
                        default=0,
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
    parser.add_argument('--error_vis_min',
                        default=-5,
                        type=float,
                        #metavar='W',
                        help='error visualize minimum value (default: -5 meters)')
    parser.add_argument('--error_vis_max',
                        default=5,
                        type=float,
                        #metavar='W',
                        help='error visualize maximum value (default: 5 meters)')
    parser.add_argument('--training_subset',
                        default=0,
                        type=int,
                        #metavar='N',
                        help='How many minibatches to use for training (depends on batch size)')
    parser.add_argument('--show_sensor_error', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default=False,
                        type=bool,
                        help='Choose if showing sensor error or present sparse error')
    
    #base args
    parser.add_argument('--kitti_crop', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='eigen_crop',
                        type=str,
                        #metavar='N',
                        help='which crop to follow for validationmetrics')
    parser.add_argument('--train_datalist', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        #default='datalist_train_nn.list',
                        #default='train_dim_kitti.list',
                        #default='4batch_dim_kitti.list',
                        #default='8batch_overfit_nn.list',
                        #default='4batch_overfit_nn_test.list',
                        #default='4batch_overfit_nn_train.list',
                        default='nyu_train_random_pts_half_reso.list',
                        #default='nyu_2000_train.list',
                        #default='nyu_test.list',
                        #default='nyu_4_overfit.list',
                        #default='single_image_4batch_overfit.list',
                        type=str,
                        #required=True,
                        help='list file to use to load dataset')
    parser.add_argument('--val_datalist', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        #default='datalist_test_nn.list',
                        #default='val_dim_kitti.list',
                        #default='4batch_dim_kitti.list',
                        #default='8batch_overfit_nn.list',
                        #default='4batch_overfit_nn_test.list',
                        #default='nyu_4_overfit.list',
                        default='nyu_test_random_pts_half_reso.list',
                        #default='nyu_test.list',
                        #default='single_image_4batch_overfit.list',
                        type=str,
                        #required=True,
                        help='list file to use to load dataset')
    parser.add_argument('--root_folder', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        #default='data/nn_dataset/',
                        default='data/nyuv2_dataset/',
                        #default='data/kitti_dataset/val_selection_cropped/',
                        type=str,
                        #required=True,
                        help='Root folder where the list and data is located')
    parser.add_argument('--torch_mode', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='pytorch',
                        choices=["pytorch", "tensorrt"],
                        type=str,
                        #required=True,
                        help='Decide if you run with pytorch or tensorrt')
    parser.add_argument('--saved_weights',
                        default='weights/KITTI_Full_GuideDepth.pth',
                        type=str,
                        help='Location of saved weights')
    #WANDB args
    parser.add_argument('--wandblogger', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default=False,
                        type=bool,
                        help='Parameter to decide if we store wandb or not')
    
    parser.add_argument('--project', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='depthcompletionpaper',
                        type=str,
                        #metavar='N',
                        help='Project name when saving in wandb')
    parser.add_argument('--entity', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='wandbdimar',
                        type=str,
                        #required=True,
                        help='Entity name when saving in wandb')
    parser.add_argument('--wandbrunname', #SHOULD ALSO INCLUDE THE OTHER OPTINS HERE
                        default='',
                        type=str,
                        #required=True,
                        help='Run name when saving in wandb')
    
    #NLSPN
    
    # Network
    parser.add_argument('--model_name',
                        type=str,
                        default='NLSPN',
                        choices=('NLSPN',),
                        help='model name')
    parser.add_argument('--networknlspn',
                        type=str,
                        default='resnet34',
                        choices=('resnet18', 'resnet34'),
                        help='network name')
    parser.add_argument('--from_scratch',
                        action='store_true',
                        default=False,
                        help='train from scratch')
    parser.add_argument('--prop_time',
                        type=int,
                        default=18,
                        help='number of propagation')
    parser.add_argument('--prop_kernel',
                        type=int,
                        default=3,
                        help='propagation kernel size')
    parser.add_argument('--preserve_input',
                        action='store_true',
                        default=False,
                        help='preserve input points by replacement')
    parser.add_argument('--affinity',
                        type=str,
                        default='TGASS',
                        choices=('AS', 'ASS', 'TC', 'TGASS'),
                        help='affinity type (dynamic pos-neg, dynamic pos, '
                            'static pos-neg, static pos, none')
    parser.add_argument('--affinity_gamma',
                        type=float,
                        default=0.5,
                        help='affinity gamma initial multiplier '
                            '(gamma = affinity_gamma * number of neighbors')
    parser.add_argument('--conf_prop',
                        action='store_true',
                        default=True,
                        help='confidence for propagation')
    parser.add_argument('--no_conf',
                        action='store_false',
                        dest='conf_prop',
                        help='no confidence for propagation')
    parser.add_argument('--legacy',
                        action='store_true',
                        default=False,
                        help='legacy code support for pre-trained models')

    
    
    #args parser
    args = parser.parse_args()
    
    
    return args