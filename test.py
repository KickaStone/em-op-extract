import os
import numpy as np
from models import CNN_1d
from torchsummary import summary
import torch
from batch import get_batch
from train import split_batch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def test_datasets(dataset_names, with_bounds=False, argument=False):
    model = CNN_1d(input_size=131072, num_classes=9)

    datasets_root = "./datasets"

    # laod model
    model.load_state_dict(torch.load('./models/cnn1d.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (1, 131072))

    batch_c = []
    confustion_matrix = np.zeros((9, 9)).astype(int)

    print('testing...')

    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataste_files = list(os.listdir(dataset_path))

        for i, dataset_file in enumerate(dataste_files):
            if '_traces.npy' in dataset_file:
                for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=20):
                    input, target, label = split_batch(batch)

                    if np.isnan(input).any():
                        print("nan in batch")
                        continue

                    xs = torch.tensor(input).float().to(device)
                    ys = torch.tensor(label).float().to(device)

                    y_pred = model(xs)
                    assert(len(y_pred) == len(ys))
                    
                    pred_labels = torch.argmax(y_pred, dim=1)
                    orig_labels = torch.argmax(ys, dim=1)

                    for i in range(len(ys)):
                        confustion_matrix[orig_labels[i]][pred_labels[i]] += 1


    print(confustion_matrix)

    labels = ["aes", "sha1prf", "hmacsha1", "des_openssl", "aes_openssl", "aes_tiny", "sha1", "sha1transform", "noise"]
    ConfusionMatrixDisplay(confustion_matrix, display_labels=labels).plot()
    plt.show()

    

if __name__ == '__main__':
    test_datasets(["nodemcu-random-test2"], with_bounds=False, argument=False)

