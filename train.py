import sys
import os
from batch import get_batch

def train_datasets(dataset_names, load=False, with_bounds=False, epochs=1):
    try:
        validation_batch = get_validation_batch(dataset_names[0].replace("train", "val"), with_bounds=with_bounds)
    except Exception:
        print("Failed to get validation set")
        validation_batch = None

    # define the model
    if model_to_use == "wavenet":
        model = ClassificationModel(input_size, num_classes, num_layers=16, load=load)
    else:
        if with_bounds:
            model = BestCNNBB(input_size, num_classes, load=load, valbatch=validation_batch)
        else:
            model = BestCNN(input_size, num_classes, load=load, valbatch=validation_batch)

    print("Receptive field: %d" % model.calculate_receptive_field())
    # assert (input_size == model.calculate_receptive_field() + 1)  # If we want receptive size to match size of trace (not really required here)

    batch_c = []  # Batch container
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for epoch in range(epochs):
            for i, dataset_file in enumerate(dataset_files):
                if '_traces.npy' in dataset_file:
                    for batch in get_batch(dataset_path, dataset_file, batch_c):
                        model.train_batch(batch)


# Args
filter_method = 'abs_nofit'
datasets_root = './datasets/'
batch_size = 20
model_to_use = 'best_cnn'
if model_to_use == "wavenet":
    batch_size = 1


#noise_snippets = snippetize(np.load("./datasets/noise.npy"))
noise_snippets = snippetize(np.load("./datasets/nodemcu-fullconnect/2020-02-19_11-52-45_598201_traces.npy")[0], snippet_length=128)
noise_patch = filter_trace(noise_snippets, filter_method)
use_newaugment = args.use_newaugment
