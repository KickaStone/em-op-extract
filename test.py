import os
import numpy as np
from models import CNN_1d
from torchsummary import summary
import torch
from batch import get_batch
from train import split_batch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, classification_report

def test_datasets(dataset_names, model_path=None, with_bounds=False, argument=False):
    model = CNN_1d(input_size=131072, num_classes=9)
    datasets_root = "./datasets"
    # laod model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    else:
        model_filename = input("Enter model path: ")
        model.load_state_dict(torch.load(model_filename))

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
                    y_out = model(xs).detach().cpu().numpy()
                    y_true.extend(np.argmax(label, axis=1).tolist())
                    y_pred.extend(np.argmax(y_out, axis=1).tolist())
    # print('y_true[0:100]', y_true[0:100])
    # print('y_pred[0:100]', y_pred[0:100])
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
    fig, ax = plt.subplots(figsize=(10, 10))
    cmp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical', colorbar=False)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(cmp.im_, cax=cax)
    plt.show()

def test_model(datasets, model_path, model):
    print('test model: ' + model_path)
    datasets_root = "./datasets"

    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_c = []
    y_pred = []
    y_true = []

    for dataset_name in datasets:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for i, dataset_file in enumerate(dataset_files):
            if '_traces.npy' in dataset_file:
                for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=20):
                    data, target, label = split_batch(batch)

                    if np.isnan(data).any():
                        print("nan in batch")
                        continue

                    x = torch.tensor(data).float().to(device)
                    y_out = model(x).detach().cpu().numpy()

                    y_true.extend(np.argmax(label, axis=1).tolist())
                    y_pred.extend(np.argmax(y_out, axis=1).tolist())

    assert len(y_true) == len(y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return report


if __name__ == '__main__':
    model_cnn = CNN_1d(input_size=131072, num_classes=9)
    labels = ["aes", "sha1prf", "hmacsha1", "des_openssl", "aes_openssl", "aes_tiny", "sha1", "sha1transform", "noise"]
    models_root = "./models"
    model_list = os.listdir(models_root)
    print(model_list)
    result = []
    accuracy = []
    reco_rates = []
    for i in range(9):
        reco_rates.append([])

    for i in range(5):
        model_path = f'./models/best_cnn-epoch-{i}.pt'
        result = test_model(datasets=['nodemcu-random-test2'], model_path=model_path, model=model_cnn)
        for l, d in result.items():
            if l.isdigit() == False:
                continue
            idx = int(l)
            reco_rates[idx].append(float(d['precision']))
        accuracy.append(result['accuracy'])

    # plot

    for i in range(9):
        plt.plot(reco_rates[i], label=labels[i])

    print(reco_rates)
    plt.plot(accuracy, label='avg accuracy')
    plt.legend()
    plt.show()