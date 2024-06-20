from device_control import device_control
import experiment
device_control = device_control()

class Logger:
    def __init__(self,
                total_epochs, 
                dataset,
                dest):
        self.display_train_time_log = True
        self.display_train_acc_log = True
        self.display_final_results_log = True
        self.display_final = True
        self.display_memory_usage = False

        self.total_epochs = total_epochs 
        self.dataset = dataset

        self.datalog = dict()
        self.datalog['dataset'] = dataset
        self.datalog['dest'] = dest
        self.datalog['epoch_times'] = []
        self.datalog['epoch_tests'] = [] # Recording test results in epochs
        self.datalog['final_results'] = None # Recording test results in final model

        self.log = '''| Train_acc: {:.4f}  | Test_acc: {:.4f}  |
| Train_sens: {:.4f} | Test_sens: {:.4f} |
| Train_spec: {:.4f} | Test_spec: {:.4f} |
| Train_f1: {:.4f}   | Test_f1: {:.4f}   |
| Train_auc: {:.4f}  | Test_auc: {:.4f}  |'''

    def load_model_log(self, accs):
        print(f"Loaded model with accuracy: Test: {accs['test_mask']}")

    def train_time_log(self, epoch, time_per_epoch, accs, senss, specs, f1s, aucs):
        self.datalog['epoch_times'].append(time_per_epoch)
        self.datalog['epoch_tests'].append([accs['test_mask'], senss['test_mask'], specs['test_mask'], f1s['test_mask'], aucs['test_mask']])

        if self.display_train_time_log:
            print(f"Training {self.dataset}, Epoch: {epoch}, Time Per Epoch: {time_per_epoch}")

            if self.display_train_acc_log:
                print(self.log.format( 
                    accs['train_mask'], accs['test_mask'],
                    senss['train_mask'], senss['test_mask'],
                    specs['train_mask'], specs['test_mask'],
                    f1s['train_mask'], f1s['test_mask'],
                    aucs['train_mask'], aucs['test_mask']))

    def final_results_log(self, accs, senss, specs, f1s, aucs):
        if self.display_final:
            print("Final Results:")
            print(self.log.format( 
                accs['train_mask'], accs['test_mask'],
                senss['train_mask'], senss['test_mask'],
                specs['train_mask'], specs['test_mask'],
                f1s['train_mask'], f1s['test_mask'],
                aucs['train_mask'], aucs['test_mask']))

    def save_experiment_results(self):
        experiment.save_experiment_results(self.datalog, self.total_epochs)

class CPSGCNLogger(Logger):
    def __init__(self, 
                times, 
                total_epochs, 
                dataset, 
                prune_ratio, 
                centrality, 
                preserve_rate, 
                preserve_duration,
                dest):
        super().__init__(total_epochs, dataset, dest)

        self.times = times
        self.prune_ratio = prune_ratio
        self.centrality = centrality
        self.preserve_rate = preserve_rate
        self.preserve_duration = preserve_duration

    def train_time_log(self, j, epoch, time_per_epoch, accs, senss, specs, f1s, aucs):
        self.datalog['epoch_times'].append(time_per_epoch)
        self.datalog['epoch_tests'].append([accs['test_mask'], senss['test_mask'], specs['test_mask'], f1s['test_mask'], aucs['test_mask']])

        if self.display_train_time_log:
            print(f"Training {self.dataset}, Epoch: {j * self.times + epoch}, Time Per Epoch: {time_per_epoch}")

            if self.display_train_acc_log:
                print(self.log.format( 
                    accs['train_mask'], accs['test_mask'],
                    senss['train_mask'], senss['test_mask'],
                    specs['train_mask'], specs['test_mask'],
                    f1s['train_mask'], f1s['test_mask'],
                    aucs['train_mask'], aucs['test_mask']))

    def after_final_wprune_log(self, accs, senss, specs, f1s, aucs):
        self.datalog['final_results'] = accs['test_mask'], senss['test_mask'], specs['test_mask'], f1s['test_mask'], aucs['test_mask']

        if self.after_final_wprune_log:
            print("After weight pruning:")
            print(self.log.format(
                accs['train_mask'], accs['test_mask'],
                senss['train_mask'], senss['test_mask'],
                specs['train_mask'], specs['test_mask'],
                f1s['train_mask'], f1s['test_mask'],
                aucs['train_mask'], aucs['test_mask']))

            print(f"Graph Ratio: {self.prune_ratio}, Weight Ratio: {self.prune_ratio}")

class SGCNLogger(Logger):
    def __init__(self, 
                times, 
                total_epochs, 
                dataset, 
                prune_ratio, 
                dest):
        super().__init__(total_epochs, dataset, dest)

        self.times = times
        self.prune_ratio = prune_ratio
        
    def train_time_log(self, j, epoch, time_per_epoch, accs, senss, specs, f1s, aucs):
        self.datalog['epoch_times'].append(time_per_epoch)
        self.datalog['epoch_tests'].append([accs['test_mask'], senss['test_mask'], specs['test_mask'], f1s['test_mask'], aucs['test_mask']])

        if self.display_train_time_log:
            print(f"Training {self.dataset}, Epoch: {j * self.times + epoch}, Time Per Epoch: {time_per_epoch}")

            if self.display_train_acc_log:
                print(self.log.format(
                    accs['train_mask'], accs['test_mask'],
                    senss['train_mask'], senss['test_mask'],
                    specs['train_mask'], specs['test_mask'],
                    f1s['train_mask'], f1s['test_mask'],
                    aucs['train_mask'], aucs['test_mask']))

    def after_final_wprune_log(self, accs, senss, specs, f1s, aucs):
        self.datalog['final_results'] = accs['test_mask'], senss['test_mask'], specs['test_mask'], f1s['test_mask'], aucs['test_mask']

        if self.after_final_wprune_log:
            print("After weight pruning:")
            print(self.log.format(
                accs['train_mask'], accs['test_mask'],
                senss['train_mask'], senss['test_mask'],
                specs['train_mask'], specs['test_mask'],
                f1s['train_mask'], f1s['test_mask'],
                aucs['train_mask'], aucs['test_mask']))

            print(f"Graph Ratio: {self.prune_ratio}, Weight Ratio: {self.prune_ratio}")
