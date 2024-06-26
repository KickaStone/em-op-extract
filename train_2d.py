import sys
import os
from batch import *
import numpy as np
import torch
import torch.nn as nn
from models import CNN_2d
import datetime

# 2d 模型输入为512*512
def train_datasets(dataset_names, with_bounds=False, epochs=1):
    try:
        validation_batch = get_validation_batch(dataset_names[0].replace("train", "val"), with_bounds=with_bounds)
    except Exception:
        print("Failed to get validation set")
        validation_batch = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_2d().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_c = []  # Batch container
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))
        cnt = 0
        for epoch in range(epochs):
            for i, dataset_file in enumerate(dataset_files):
                if '_traces.npy' in dataset_file:
                    for batch in get_batch_2d(dataset_path, dataset_file, batch_c, batch_size=20):
                        cnt += 1
                        # print(f"\n\nBatch {cnt}")

                        input, target, label = split_batch(batch)
                        # check if any nan in the input
                        if np.isnan(input).any():
                            # print("nan in batch")
                            continue
                        # print('training...')

                        xs = torch.tensor(input).float().to(device)
                        ys = torch.tensor(label).float().to(device)

                        y_pred = model(xs)

                        loss = criterion(y_pred, ys)
                        if cnt % 100 == 0:
                            print(f'Epoch {epoch}, Batch {cnt}, Loss: {loss.item()}')
                        # print('loss', loss.item())

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()

                        # save checkpoint
                        if cnt % 1000 == 0:
                            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            torch.save(model.state_dict(), f"models/{model_to_use}-{time}.pt")
                            print(f"Model saved as {model_to_use}-{time}.pt")

        # saving model
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(model.state_dict(), f"models/cnn1d.pt")
        print(f"Model saved")


def split_batch(batch):
    inputs = np.stack(batch[:, 0], axis=0)[:, :, None]
    targets = np.stack(batch[:, 1], axis=0)
    labels = np.stack(batch[:, 2], axis=0)
    inputs = np.squeeze(inputs, axis=2)
    return inputs, targets, labels


# Args
filter_method = 'abs_nofit'
datasets_root = './datasets/'
batch_size = 20
model_to_use = 'best_cnn'
if model_to_use == "wavenet":
    batch_size = 1

# #noise_snippets = snippetize(np.load("./datasets/noise.npy"))
# noise_snippets = snippetize(np.load("./datasets/nodemcu-fullconnect/2020-02-19_11-52-45_598201_traces.npy")[0], snippet_length=128)
# noise_patch = filter_trace(noise_snippets, filter_method)
# use_newaugment = args.use_newaugment
if '__main__' == __name__:
    train_datasets(['nodemcu-random-train2'], with_bounds=False, epochs=1)
