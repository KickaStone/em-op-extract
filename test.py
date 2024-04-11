import os
import numpy as np
from models import CNN_1d
from torchsummary import summary
import torch
from batch import get_batch
from train import split_batch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

def test_datasets(dataset_names, with_bounds=False, argument=False):
    model = CNN_1d(input_size=131072, num_classes=9)

    datasets_root = "./datasets"

    # laod model
    model_filename = input("Enter model path: ")
    model.load_state_dict(torch.load(model_filename))

    # model.load_state_dict(torch.load('models/best_cnn-2024-04-09_09-02-49--epoch-65-batch214000.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (1, 131072))

    batch_c = []

    print('testing...')

    # y_true = np.array([], dtype=int)
    # y_pred = np.array([], dtype=int)

    y_pred = []
    y_true = []


    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for i, dataset_file in enumerate(dataset_files):
            if '_traces.npy' in dataset_file:
                for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=20):
                    data, target, label = split_batch(batch)

                    if np.isnan(data).any():
                        print("nan in batch")
                        continue

                    xs = torch.tensor(data).float().to(device)
                    ys = torch.tensor(label).float().to(device)

                    y_out = model(xs).detach().cpu().numpy()

                    y_true.extend(np.argmax(label, axis=1).tolist())
                    y_pred.extend(np.argmax(y_out, axis=1).tolist())
                    # y_true.append(np.argmax(label))
                    # y_pred.append(np.argmax(y_out.cpu().detach().numpy()))

                    # pred_labels = torch.argmax(y_pred, dim=1)
                    # orig_labels = torch.argmax(ys, dim=1)
    print('y_true[0:100]', y_true[0:100])
    print('y_pred[0:100]', y_pred[0:100])
    assert len(y_true) == len(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    print(f'Accuracy: {accuracy:}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    conf_matrix = confusion_matrix(y_true, y_pred)
    labels = ["aes", "sha1prf", "hmacsha1", "des_openssl", "aes_openssl", "aes_tiny", "sha1", "sha1transform", "noise"]
    cmp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    _, ax = plt.subplots(figsize=(12, 12))
    cmp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.show()

if __name__ == '__main__':
    test_datasets(["nodemcu-random-test2"], with_bounds=False, argument=False)

