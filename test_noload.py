from dataset import get_pathdata
import model_repo
import test
import os
import numpy as np

if __name__ == '__main__':
    testdata_path = r'./wval_data/used/test_data.npy'
    os.makedirs('./Inference/', exist_ok=True)
    save_path = r'./Inference/'

    pth = r'./LearningRate_Experiment1_proposed_two_branch_Adam_0.01/epoch92.pth'
    netname = model_repo.proposed_two_branch

    config = {
        'netname': netname.Net,
        'dataset': {'test': get_pathdata(testdata_path),},
        'pth_repo': pth,
        'test_path': save_path,
    }
    tester = test.Test(config)

    accuracys = []
    maes = []


    for i in range(1):
        print(f"Running test iteration {i + 1}...")
        accuracy,mae = tester.start()
        accuracy = accuracy*100
        accuracys.append(accuracy)
        maes.append(mae)

        print(f"Test iteration {i + 1} completed.")

    accuracys = np.array(accuracys)
    print("\t".join(map(str, accuracys)))
    maes = np.array(maes)
    print("\t".join(map(str, maes)))
