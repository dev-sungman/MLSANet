import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', '--msg', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='real')
    parser.add_argument('--train_path', type=str, default="/workspace/datasets/cleansing_datasets/baseline_followup_pair_4class")
    parser.add_argument('--test_path', type=str, default="/workspace/datasets/cleansing_datasets/baseline_followup_pair_4class")

    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--gpu_idx', type=str, default='0')

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w', type=int, default=16)
    
    parser.add_argument('--log_dir', type=str, default="runs")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")

    parser.add_argument('--print_freq', type=int, default=300)
    
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--img_folder', type=str, default='imgs') 
    
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default=None)
    return parser.parse_args()
