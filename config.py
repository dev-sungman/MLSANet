import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', '--msg', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='real')
    parser.add_argument('--train_path', type=str, default="/workspace/datasets/cleansing_datasets/baseline_followup_pair_4class")
    parser.add_argument('--test_path', type=str, default="/workspace/datasets/cleansing_datasets/baseline_followup_pair")

    parser.add_argument('--epochs', type=int, default=300, help="number of epochs")
    parser.add_argument('--gpu_idx', type=str, default='0')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--w', type=int, default=16)
    
    parser.add_argument('--log_dir', type=str, default="runs")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")

    parser.add_argument('--print_freq', type=int, default=100)
    
    parser.add_argument('--backbone', type=str, default="resnet")
    parser.add_argument('--img_folder', type=str, default='imgs') 

    return parser.parse_args()
