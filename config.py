import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', '--msg', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='real')
    parser.add_argument('--train_path', type=str, default="/workspace/datasets/cleansing_datasets/baseline_followup_pair_4class")
    parser.add_argument('--test_path', type=str, default="/workspace/datasets/cleansing_datasets/baseline_followup_pair_4class")
    parser.add_argument('--random_seed', type=int, default=100)

    parser.add_argument('--start_epoch', type=int, default = 0)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
    parser.add_argument('--gpu_idx', type=str, default='0')

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w', type=int, default=16)
    
    parser.add_argument('--log_dir', type=str, default="runs")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")

    parser.add_argument('--print_freq', type=int, default=1000)
    
    parser.add_argument('--backbone', type=str, default="resnet152")
    parser.add_argument('--img_folder', type=str, default='imgs') 
    
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained', type=str, default=None)

    parser.add_argument('--disease_weight', type=float, default=2.)
    parser.add_argument('--change_weight', type=float, default=1.)
    parser.add_argument('--matching_weight', type=float, default=0.01)

    return parser.parse_args()
