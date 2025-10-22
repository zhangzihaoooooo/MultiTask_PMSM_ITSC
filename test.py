import torch
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,mean_absolute_error

class Test:
    def __init__(self, configs):
        self.t_time = 0.0
        self.t_sec = 0.0
        self.net = configs['netname']('_')

        self.test = configs['dataset']['test']
        self.val_dataloader = torch.utils.data.DataLoader(self.test,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=4)
        self.device = torch.device( "cpu")

        self.pth = configs['pth_repo']
        self.sava_path = configs['test_path']
        self.print_staistaic_text = self.sava_path + 'print_staistaic_text.txt'


    def start(self):
        print("Loading .......   path:{}".format(self.pth))

        state = torch.load(self.pth, map_location="cpu")

        new_state_dict = {}
        for key, value in state['model'].items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value

        self.net.load_state_dict(new_state_dict)

        self.net.to(self.device)
        test_normstatic=1
        accuracy,mae = self.val_step(test_normstatic,self.pth[-5],self.val_dataloader)

        return accuracy,mae

    def val_step(self,test_normstatic, epoch, dataset):
        print('-----------------start test--------------------')


        self.csv_onlylable = []

        self.net = self.net.eval()
        star_time = time.time()

        for i, data in enumerate(dataset):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)
            with torch.no_grad():

                prediction = self.net(images)
                p1 = prediction.to(self.device)

                l1 = labels.to(self.device)


                temp_onlylable = torch.cat([l1.squeeze(0), p1.squeeze(0)], dim=-1)
                self.csv_onlylable.append(temp_onlylable.cpu().detach().numpy().squeeze())


        duration = time.time() - star_time
        speed = 1 / (duration / len(dataset))
        print('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle = open(self.print_staistaic_text, mode='a')

        file_handle.write('-----------------start test--------------------')
        file_handle.write('\n')
        file_handle.write('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle.write('\n')
        file_handle.close()

        self.net = self.net.train()
        accuracy, mae = self.tocsv_onlylable(epoch)

        print('-----------------test over--------------------')

        return accuracy, mae





    def tocsv_onlylable(self, epoch):

        np_data = np.array(self.csv_onlylable)

        label = np_data[:,:2]
        pred = np_data[:,2:]
        label[:,1]  = label[:,1]*16
        pred[:, 1] = np.clip(pred[:, 1] * 16, a_min=0, a_max=None)

        pred[:, 0] = (pred[:, 0] > 0.5).astype(np.int32)
        pred[:, 1] = np.round(pred[:, 1])

        label_class = label[:,0].astype(int)
        label_reg = label[:,1]

        pred_class = pred[:, 0].astype(int)
        pred_reg = pred[:,1]


        precision = precision_score(label_class, pred_class, average='macro')
        recall = recall_score(label_class, pred_class, average='macro')
        f1 = f1_score(label_class, pred_class, average='macro')
        accuracy = accuracy_score(label_class, pred_class)
        mae = mean_absolute_error(label_reg, pred_reg)

        #

        print(
            'epoch:{} 测试accuracy:{}'.format(epoch,
                        accuracy)
        )
        print(
            'epoch:{} 测试precision:{}'.format(epoch,
                        precision)
        )
        print(
            'epoch:{} 测试recall:{}'.format(epoch,
                        recall)
        )
        print(
            'epoch:{} 测试f1:{}'.format(epoch,
                        f1)
        )

        print(
            'epoch:{} 测试mae:{}'.format(epoch,
                        mae)
        )


        file_handle = open(self.print_staistaic_text, mode='a')



        file_handle.write('epoch:{}测试accuracy:{}'.format(epoch,
                                                    accuracy ))
        file_handle.write('\n')

        file_handle.write('epoch:{}测试precision:{}'.format(epoch,
                                                    precision ))
        file_handle.write('\n')
        file_handle.write('epoch:{}测试recall:{}'.format(epoch,
                                                    recall))
        file_handle.write('\n')
        #
        file_handle.write('epoch:{}测试f1:{}'.format(epoch,f1))
        file_handle.write('\n')

        #
        file_handle.write('epoch:{}测试mae:{}'.format(epoch,mae))
        file_handle.write('\n')

        file_handle.write('-----------------测试结束--------------------')
        file_handle.write('\n')
        file_handle.close()

        np.savetxt(self.sava_path + str(epoch)+'_pred_onlylable.csv', np_data, delimiter=',')

        return accuracy, mae


