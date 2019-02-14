import sys
sys.path.append('./')
import json
import argparse
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
torch.backends.cudnn.enabled = True

from nep.utils import *
from nep.logger import myLogger
from nep.dataset import EvaDataset, Dataset
from nep.model import MLPClassifier, LabelEncoder, ModuleNet


def evaluate_embedding(args, dataset, embedding, repeat_times=5):
    print('=' * 150)
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []

    X_train = embedding[dataset.train_nodes]
    y_train = np.array([dataset.node_to_label[i] for i in dataset.train_nodes]).reshape(-1, 1)
    train = np.concatenate((X_train, y_train), axis=1)
    X_test = embedding[dataset.test_nodes]
    y_test = np.array([dataset.node_to_label[i] for i in dataset.test_nodes]).reshape(-1, 1)
    test = np.concatenate((X_test, y_test), axis=1)

    X_train, y_train = torch.FloatTensor(train[:, :-1]), torch.LongTensor(train[:, -1])
    X_test, y_test = torch.FloatTensor(test[:, :-1]), torch.LongTensor(test[:, -1])
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    y_train = y_train.cuda()
    y_test = y_test.cuda()
    dataloader = DataLoader(EvaDataset(X_train, y_train), batch_size=args.batch_size_eval, shuffle=True)

    kwargs = {
        'input_dim': args.embedding_size,
        'hidden_dim': args.embedding_size // 2,  # args.hidden_eval,
        'output_dim': args.num_class
    }

    for i in range(repeat_times):
        model = MLPClassifier(**kwargs).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_eval)
        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        count = 0
        for epoch in range(args.num_epoch_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch, label)
                loss.backward()
                optimizer.step()

            test_acc = model.predict(X_test, y_test)
            test_acc *= 100
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch + 1
                count = 0
            else:
                count += 1
                if count >= args.patience_eval:
                    break

            train_acc = model.predict(X_train, y_train)
            train_acc *= 100
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1

            print('\repoch {}/{} train acc={:.4f}, test acc={:.4f}, best train acc={:.4f} @epoch:{:d}, best test acc={:.4f} @epoch:{:d}'.
                  format(epoch + 1, args.num_epoch_eval, train_acc, test_acc, best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch), end='')
            sys.stdout.flush()

        print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs)
    std = np.std(best_test_accs)

    print('=' * 150)
    return best_train_acc, best_test_acc, std, int(best_train_acc_epoch), int(best_test_acc_epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    # general options
    parser.add_argument('--dataset', type=str, default='dblp-sub')
    parser.add_argument('--pattern_path', type=str, default='', help="path to load/save pattern")
    parser.add_argument("--prefix", type=str, default='', help="prefix use as addition directory")
    parser.add_argument('--suffix', default='', type=str, help='suffix append to log dir')
    parser.add_argument('--log_level', default=20, help='logger level.')
    parser.add_argument('--log_every', type=int, default=100, help='log results every epoch.')
    parser.add_argument('--save_every', type=int, default=500, help='save learned embedding every epoch.')

    # data options
    parser.add_argument('--target_node_type', type=str, default='a')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--superv_ratio', type=float, default=1.0)
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--num_pattern', type=int, default=None)
    parser.add_argument('--num_walkers_for_pattern', type=int, default=100)
    parser.add_argument('--path_max_length', type=int, default=7)

    # module options
    parser.add_argument('--embedding_size', type=int, default=64)

    # Optimization options
    parser.add_argument('--num_epoch', type=int, default=100000)
    parser.add_argument('--num_data_per_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--early_stop', type=int, default=1)
    parser.add_argument('--patience', type=int, default=2000)

    # evluation options
    parser.add_argument('--batch_size_eval', type=int, default=32)
    parser.add_argument('--learning_rate_eval', type=float, default=5e-4)
    parser.add_argument('--num_epoch_eval', type=int, default=500)
    parser.add_argument('--patience_eval', type=int, default=50, help='used for early stop in evaluation')

    # Output options
    parser.add_argument('--output_path', type=str, default='')
    return parser.parse_args()


def main(args):
    start_time = time.time()

    dataset = Dataset(data_dir=osp.join('data', args.dataset), num_data_per_epoch=args.num_data_per_epoch,
                      threshold=args.threshold, superv_ratio=args.superv_ratio, train_ratio=args.train_ratio)
    args.num_class = dataset.num_class
    args.num_module = dataset.num_link_type
    args.num_link = dataset.num_link
    args.node_type = dataset.id_to_type
    args.num_node = dataset.num_node
    args.num_target_node = len(dataset.type_to_node[args.target_node_type])
    args.num_labeled_node = len(dataset.train_nodes)+len(dataset.test_nodes)

    # initialize logger
    if args.prefix:
        base = os.path.join('log', args.prefix)
        log_dir = os.path.join(base, args.suffix)
    else:
        comment = f'_{args.dataset}_{args.suffix}' if args.suffix else f'_{args.dataset}'
        current_time = datetime.now().strftime('%b_%d_%H-%M-%S')
        log_dir = os.path.join('log', current_time + comment)
    args.log_dir = log_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = myLogger(name='exp', log_path=os.path.join(log_dir, 'log.txt'))
    logger.setLevel(args.log_level)
    print_config(args, logger)

    if not args.output_path:
        args.output_path = osp.join(log_dir, 'embedding.bin')

    # encode label
    logger.info('=' * 100)
    logger.info('Start encoding label to seed nodes...')
    label_encoder = LabelEncoder(dataset, args)
    target_embedding = label_encoder.train()
    del label_encoder

    # generate pattern
    logger.info('Check pattern file...')
    if not args.pattern_path:
        args.pattern_path = f'data/tmp/{args.dataset}_pattern.dat'
        logger.info(f'No input pattern file, so we generate pattern and save it into {args.pattern_path}')
        dataset.init_pattern(dataset.train_nodes, args.num_pattern, args.num_walkers_for_pattern,
                             args.path_max_length, args.target_node_type, reverse_path=False, verbose=True)
        dataset.save_pattern(args.pattern_path)
    else:
        logger.info(f'Load pattern from {args.pattern_path}')
        dataset.load_pattern(args.pattern_path)
    dataset.free_memory()

    def next_batch(X, batch_size):
        num = len(X)
        for i in np.arange(0, num, batch_size):
            yield X[i:i + batch_size]

    # initialize model
    kwargs = {
        'target_embedding': target_embedding,
        'num_node': args.num_target_node,
        'embedding_size': args.embedding_size,
        'num_module': args.num_module,
    }
    model = ModuleNet(**kwargs)
    model.cuda()
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.learning_rate)

    # train model
    train_start_time = time.time()
    train_time, sample_time, sample_count = 0, 0, 0
    acc_loss, count_loss, count = 0, 0, 0
    best_test_acc = -1
    trace = {'loss': [], 'test acc': []}
    logger.info('=' * 100)
    logger.info("Starting training model...")
    for epoch in range(1, args.num_epoch+1):
        start_sample_time = time.time()
        ret = dataset.collect_data(args.num_data_per_epoch)
        epoch_data, path = ret
        train_data = torch.LongTensor(epoch_data).cuda()
        sample_time += time.time() - start_sample_time
        sample_count += 1

        start_train_time = time.time()
        for batch in next_batch(train_data, batch_size=args.batch_size):
            optimizer.zero_grad()
            loss = model(path, batch)
            loss.backward()
            acc_loss += loss.item()
            count_loss += 1
            optimizer.step()
        model.copy_embedding()
        train_time += time.time() - start_train_time

        avr_sample_time = timedelta(seconds=(sample_time/sample_count))
        avr_train_time = timedelta(seconds=(train_time/(epoch + 1)))

        if epoch % args.log_every == 0:
            duration = time.time() - train_start_time
            avr_loss = acc_loss / count_loss
            acc_loss, count_loss = 0, 0
            logger.info(f'Epoch: {epoch:04d} loss: {avr_loss:.4f} duration: {duration:.4f} avr train time: {avr_train_time} avr sample time: {avr_sample_time}')
            trace['loss'].append((epoch, avr_loss))

        if epoch % args.save_every == 0:
            train_acc, test_acc, std, train_acc_epoch, test_acc_epoch = evaluate_embedding(args, dataset, model.return_embedding())
            trace['test acc'].append((epoch, test_acc))
            logger.info('best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f} @epoch:{:d}'.
                        format(train_acc, train_acc_epoch, test_acc, std, test_acc_epoch))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_std = std
                best_epoch = epoch
                best_model = model.state_dict()
                best_opt = optimizer.state_dict()
                model.save_embedding(dataset.id_to_name, args.output_path, True)
                count = 0
            else:
                if args.early_stop:
                    count += args.save_every
                if count >= args.patience:
                    logger.info('early stopped!')
                    break

    print('')

    # save results
    json.dump(trace, open(osp.join(args.log_dir, 'trace.json'), 'w'), indent=4)
    save_checkpoint({
        'args': args,
        'model': best_model,
        'optimizer': best_opt,
    }, args.log_dir, f'epoch{best_epoch}_acc{best_test_acc}.pth.tar', logger, True)
    total_cost_time = "total cost time: {} ".format(timedelta(seconds=(time.time() - start_time)))

    logger.info('best test acc={:.2f} += {:.2f} @epoch:{:d}'.
                format(best_test_acc, best_std, best_epoch))
    logger.info(total_cost_time)


if __name__ == '__main__':
    args = parse_args()
    main(args)

