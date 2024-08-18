import argparse
import test
from name import *
from utils import log_print

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='AIDS', help='Dataset used')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=256, help='Training batch size')
parser.add_argument('--nepoch', type=int, default=20, help='Number of training epochs')
parser.add_argument('--hdim', type=int, default=128, help='Hidden feature dim')
parser.add_argument('--width', type=int, default=4, help='Width of GCN')
parser.add_argument('--depth', type=int, default=6, help='Depth of GCN')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--normalize', type=int, default=1, help='Whether batch normalize')
parser.add_argument('--beta', type=float, default=0.999, help='CB loss beta')
parser.add_argument('--gamma', type=float, default=1.5, help='CB loss gamma')
parser.add_argument('--decay', type=float, default=0, help='Weight decay')
parser.add_argument('--seed', type=int, default=10, help='Random seed')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--intergraph', default='none', help="mean or max or attention or none")
parser.add_argument('--alltests', type=int, default=0, help='Run all tests for the data and hyperparameter')
parser.add_argument('--datagroup', type=int, default=1, help="select dataset group")
args = parser.parse_args()


#checking all hyper parameters with intergraph max and mean with integraph 0 (disabled)
if args.alltests == 1:
    if args.datagroup == 1:
        completed_index = 17
        datasets = group1
    elif args.datagroup == 2:
        datasets = group2
        completed_index = 999
    elif args.datagroup == 3:
        completed_index = 999
        datasets = group3
    elif args.datagroup == 4:
        datasets = group4
        completed_index = 999
    elif args.datagroup == 5:
        completed_index = 15
        datasets = group5
        end_index = 18
    elif args.datagroup == 6:
        completed_index = 14
        datasets = group6
        end_index = 18
    intergraph_options = ['none', 'sort', 'set2set', "sage", 'mean', 'max']
    total_tests = (
        len(datasets) *
        len(intergraph_options)
    )

    log_print(f"There will be {total_tests} total tests below")

    index = 1
    for dataset in datasets:
        log_print(f"Group by {dataset}")
        for pooling_type in intergraph_options:
            if index > completed_index or index < end_index:
                args.data = dataset
                args.intergraph = pooling_type
                args.lr = 1e-3
                args.batchsize = 256
                args.hdim = 128
                args.width = 4
                args.depth = 6
                args.dropout = 0.4
                args.decay = 0  # Set decay value
                log_print(f"Test number: {index}/{total_tests}")
                test.execute(args)
            else:
                log_print(f"Test number: {index}/{total_tests} skipped")
            index += 1                                  
        log_print(f"End group by {dataset}")
elif args.alltests == 2:
    completed_index = 15
    end_index = 18
    dataset = 'NCI-H23'
    learning_rates = [5e-3, 1e-3]
    batch_sizes = [256]
    hidden_dims = [64, 128]
    widths = [4]
    depths = [6]
    dropouts = [0.4, 0.5]
    decay_values = [0, 1e-4, 1e-5]
    total_tests = (
        len(learning_rates) *
        len(batch_sizes) *
        len(hidden_dims) *
        len(widths) *
        len(depths) *
        len(dropouts) *
        len(decay_values)
    )
    log_print(f"There will be {total_tests} total tests below")

    index = 1
    for width in widths:
        for depth in depths:
            for lr in learning_rates:
                for batchsize in batch_sizes:
                    for hdim in hidden_dims:
                        for decay in decay_values:  # Loop through decay values
                            for dropout in dropouts:
                                for pooling_type in ['mean']:
                                    if index > completed_index and index < end_index:
                                        args.data = dataset
                                        args.intergraph = pooling_type
                                        args.lr = lr
                                        args.batchsize = batchsize
                                        args.hdim = hdim
                                        args.width = width
                                        args.depth = depth
                                        args.dropout = dropout
                                        args.decay = decay
                                        log_print(f"Test number: {index}/{total_tests} {args}")
                                        test.execute(args)
                                    else:
                                        log_print(f"Test number: {index}/{total_tests} skipped")
                                    index += 1     
elif args.alltests == 3:
    completed_index = 14
    dataset = 'SN12C'
    end_index = 18
    learning_rates = [5e-3, 1e-3]
    batch_sizes = [256]
    hidden_dims = [64, 128]
    widths = [4]
    depths = [6]
    dropouts = [0.4, 0.5]
    decay_values = [0, 1e-4, 1e-5]
    total_tests = (
        len(learning_rates) *
        len(batch_sizes) *
        len(hidden_dims) *
        len(widths) *
        len(depths) *
        len(dropouts) *
        len(decay_values)
    )
    log_print(f"There will be {total_tests} total tests below")

    index = 1
    for width in widths:
        for depth in depths:
            for lr in learning_rates:
                for batchsize in batch_sizes:
                    for hdim in hidden_dims:
                        for decay in decay_values:  # Loop through decay values
                            for dropout in dropouts:
                                for pooling_type in ['mean']:
                                    if index > completed_index and index < end_index:
                                        args.data = dataset
                                        args.intergraph = pooling_type
                                        args.lr = lr
                                        args.batchsize = batchsize
                                        args.hdim = hdim
                                        args.width = width
                                        args.depth = depth
                                        args.dropout = dropout
                                        args.decay = decay
                                        log_print(f"Test number: {index}/{total_tests} {args}")
                                        test.execute(args)
                                    else:
                                        log_print(f"Test number: {index}/{total_tests} skipped")
                                    index += 1                                  
elif args.alltests == 4:
    completed_index = -1
    dataset = 'Mutagenicity'
    learning_rates = [5e-3, 1e-3]
    batch_sizes = [256]
    hidden_dims = [64, 128]
    widths = [4]
    depths = [6]
    dropouts = [0.4, 0.5]
    decay_values = [0, 1e-4, 1e-5]
    total_tests = (
        len(learning_rates) *
        len(batch_sizes) *
        len(hidden_dims) *
        len(widths) *
        len(depths) *
        len(dropouts) *
        len(decay_values)
    )
    log_print(f"There will be {total_tests} total tests below")

    index = 1
    for width in widths:
        for depth in depths:
            for lr in learning_rates:
                for batchsize in batch_sizes:
                    for hdim in hidden_dims:
                        for decay in decay_values:  # Loop through decay values
                            for dropout in dropouts:
                                for pooling_type in ['mean']:
                                    if index > completed_index or index < end_index:
                                        args.data = dataset
                                        args.intergraph = pooling_type
                                        args.lr = lr
                                        args.batchsize = batchsize
                                        args.hdim = hdim
                                        args.width = width
                                        args.depth = depth
                                        args.dropout = dropout
                                        args.decay = decay
                                        log_print(f"Test number: {index}/{total_tests} {args}")
                                        test.execute(args)
                                    else:
                                        log_print(f"Test number: {index}/{total_tests} skipped")
                                    index += 1                                                                  
else:
    test.execute(args)
