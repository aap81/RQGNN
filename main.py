import argparse
import test
from utils import log_print
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='AIDS', help='Dataset used')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=512, help='Training batch size')
parser.add_argument('--nepoch', type=int, default=20, help='Number of training epochs')
parser.add_argument('--hdim', type=int, default=64, help='Hidden feature dim')
parser.add_argument('--width', type=int, default=4, help='Width of GCN')
parser.add_argument('--depth', type=int, default=6, help='Depth of GCN')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--normalize', type=int, default=1, help='Whether batch normalize')
parser.add_argument('--beta', type=float, default=0.999, help='CB loss beta')
parser.add_argument('--gamma', type=float, default=1.5, help='CB loss gamma')
parser.add_argument('--decay', type=float, default=0, help='Weight decay')
parser.add_argument('--seed', type=int, default=10, help='Random seed')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--intergraph', type=int, default=0, help="Combine existing intra graph analysis with inter graph analysis")
parser.add_argument('--intergraphpooling', default='mean', help="mean or max")
parser.add_argument('--alltests', type=int, default=0, help='Run all tests for the data and hyperparameter')
args = parser.parse_args()


if args.alltests == 1:
    # Define your range of hyperparameters
    learning_rates = [5e-3, 1e-3]
    batch_sizes = [256, 512, 1024]
    hidden_dims = [64, 128]
    widths = [4, 8]
    depths = [6, 10]
    dropouts = [0.4, 0.5]
    decay_values = [0, 1e-4, 1e-5]  # Add decay parameter values

    intergraph_options = 3
    total_tests = (
        len(learning_rates) *
        len(batch_sizes) *
        len(hidden_dims) *
        len(widths) *
        len(depths) *
        len(dropouts) *
        len(decay_values) *
        intergraph_options
    )

    log_print(f"There will be {total_tests} total tests below")

    index = 1
    # Automate testing
    for lr in learning_rates:
        for batchsize in batch_sizes:
            for hdim in hidden_dims:
                for width in widths:
                    for depth in depths:
                        for dropout in dropouts:
                            for decay in decay_values:  # Loop through decay values
                                for pooling_type in ['mean', 'max']:
                                    # Run intergraph analysis with mean/max pooling
                                    args.intergraph = 1
                                    args.intergraphpooling = pooling_type
                                    args.lr = lr
                                    args.batchsize = batchsize
                                    args.hdim = hdim
                                    args.width = width
                                    args.depth = depth
                                    args.dropout = dropout
                                    args.decay = decay  # Set decay value
                                    log_print(f"Test number: {index}/{total_tests}")
                                    index += 1
                                    test.execute(args)  # Assuming you move your training loop logic to a function

                                # Run without intergraph analysis
                                args.intergraph = 0
                                log_print(f"Test number: {index}/{total_tests}")
                                index += 1
                                test.execute(args)
elif args.alltests == 2:
    # Define your range of hyperparameters
    learning_rates = [5e-3, 1e-3]
    # batch_sizes = [256, 512, 1024]
    batch_sizes = [512]
    # hidden_dims = [64, 128]
    hidden_dims = [64]
    # widths = [4, 8]
    widths = [4]
    # depths = [6, 10]
    depths = [6]
    dropouts = [0.4, 0.5]
    # decay_values = [0, 1e-4, 1e-5]  # Add decay parameter values
    decay_values = [0]  # Add decay parameter values

    intergraph_options = 3
    total_tests = (
        len(learning_rates) *
        len(batch_sizes) *
        len(hidden_dims) *
        len(widths) *
        len(depths) *
        len(dropouts) *
        len(decay_values) *
        intergraph_options
    )

    log_print(f"There will be {total_tests} total tests below")

    index = 1
    # Automate testing
    for lr in learning_rates:
        for batchsize in batch_sizes:
            for hdim in hidden_dims:
                for width in widths:
                    for depth in depths:
                        for dropout in dropouts:
                            for decay in decay_values:  # Loop through decay values
                                for pooling_type in ['mean', 'max']:
                                    # Run intergraph analysis with mean/max pooling
                                    args.intergraph = 1
                                    args.intergraphpooling = pooling_type
                                    args.lr = lr
                                    args.batchsize = batchsize
                                    args.hdim = hdim
                                    args.width = width
                                    args.depth = depth
                                    args.dropout = dropout
                                    args.decay = decay  # Set decay value
                                    log_print(f"Test number: {index}/{total_tests}")
                                    index += 1
                                    test.execute(args)  # Assuming you move your training loop logic to a function

                                # Run without intergraph analysis
                                args.intergraph = 0
                                log_print(f"Test number: {index}/{total_tests}")
                                index += 1
                                test.execute(args)
else:
    test.execute(args)