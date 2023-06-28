# %%
global QUIET
QUIET = True
from finetuning_utils import *
import wandb


# %%
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
weights_dir = config['data']['model_weights']


# %%
def calculate_accuracy_sweep(model, test_loader, num_classes):
    """
    Calculate the pixel-wise accuracy of a model on the test set, 
    along with per-class accuracy, mean accuracy, and average accuracy.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        num_classes: The number of target classes.

    Returns:
        DataFrame containing per-class accuracies, mean accuracy, and average accuracy.
    """
    #print("\nEvaluating Accuracy on Test Set...")

    correct_pixels = 0
    num_pixels = 0

    # Maintain a list of counters for each class
    correct_pixels_per_class = [0] * num_classes
    num_pixels_per_class = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            logits = model(data)['out']
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            correct_pixels += (pred.cpu() == label).sum().item()
            num_pixels += np.prod(label.shape)

            # Calculate per class accuracy
            for i in range(num_classes):
                if i in [0,4]:
                    continue
                else:
                    correct_pixels_per_class[i] += ((pred.cpu() == label) & (label == i)).sum().item()
                    num_pixels_per_class[i] += (label == i).sum().item()

    # Calculate and store per class accuracies
    accuracy_dict = {}
    per_class_acc = []
    for i in range(num_classes):
        if i in [0,4]:
                continue
        else:
            if num_pixels_per_class[i] > 0:
                accuracy = np.round((correct_pixels_per_class[i]/num_pixels_per_class[i])*100, 2)
                per_class_acc.append(accuracy)
                accuracy_dict[Label_classes[i]] = accuracy
            else:
                accuracy_dict[Label_classes[i]] = 'No instances in the test set'

    # Calculate and store mean accuracy
    if per_class_acc:
        mean_acc = np.mean(per_class_acc)
        accuracy_dict['Mean Accuracy'] = mean_acc

    # Calculate and store average accuracy
    avg_acc = correct_pixels / num_pixels * 100
    accuracy_dict['Average Accuracy'] = avg_acc

    # Convert dictionary to pandas DataFrame and return
    df = pd.DataFrame(list(accuracy_dict.items()), columns=['Metric', 'Accuracy (%)'])
    
    return accuracy_dict['Mean Accuracy'], accuracy_dict['Average Accuracy']

# %%
def train_epoch_sweep(model, train_loader, epoch):
    """
    Trains the model for one epoch using the provided training data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader object that provides the training data.
        epoch (int): The current epoch number.

    Returns:
        None
    """
    model.train()
    loss_sum = 0
    for batch_id, (data, label) in enumerate(train_loader):
        # Send data and label to DEVICE
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.cuda.amp.autocast(): 
            # Forward Pass:
            output = model(data)['out']

            # Caluclate Loss
            loss = LOSS_FUNC(output, label.long())

        # Backward Pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Comupte loss sum and count batches
        loss_sum += loss.item()
        batch_id += 1

        #if batch_id%5==0:
        #    progress = f'Epoch: {epoch} | Batch {batch_id} / {len(train_loader)} | Loss: {np.round(loss_sum/(batch_id),4)}'
        #    print(progress)
        return loss_sum/len(train_loader)
# %%
def train_model_sweep(config=None):

    # Initialize a new wandb run
    with wandb.init(config=config):
        
        # Set wandb config
        config = wandb.config

        #NUM_EPOCHS=2
        BATCH_SIZE = config.batchSize
        NUM_EPOCHS = config.num_epochs

        # Initialize Model
        FROZEN = True
        model_name = 'loveda_DeepLabV3_r101_epochs15_lr0-07487'
        model = torch.load(weights_dir + '/own_training/' + model_name + '/' + model_name + '_epoch14.pth.tar', 
                        map_location=torch.device('cpu'))

        # freezing backbone and training only the classifier
        if FROZEN: # freeze all weights except last layer, i.e. clasification head
            for layer_name, param in model.named_parameters():
                if 'classifier' not in layer_name:
                    param.requires_grad = False

        model = model.to(DEVICE)

        # Initialize Dataloaders 
        train_loader, test_loader = get_munich_tuning_loaders(batch_size=BATCH_SIZE)

        # Optimization setup
        global optimizer
        global LOSS_FUNC
        global scaler

        # Scaler
        scaler = torch.cuda.amp.GradScaler()

        # Optimizer
        if config.optimizer == 'sgd':
            optimizer = optim.SGD(params=model.parameters(), lr=config.learningRate)
        elif config.optimizer == 'adam':
            optimizer = optim.Adam(params=model.parameters(), lr=config.learningRate)

        # Loss function
        LOSS_FUNC = nn.CrossEntropyLoss()


        for epoch in range(NUM_EPOCHS):
            loss = train_epoch_sweep(model, train_loader, epoch)
            mean_acc, avg_acc = calculate_accuracy_sweep(model, test_loader, 6)

            wandb.log({'Loss': loss, 'MeanAccuracy': mean_acc, 'AvgAccuracy':avg_acc, 'Epoch': epoch+1})

# %%
# # Specifying the search strategy
sweep_config = {
    'method': 'random'
    }

# Define metric 
metric = {
    'name': 'MeanAccuracy',
    'goal': 'maximize'
    }

sweep_config['metric'] = metric

# Define search space
parameters_dict = {
    'learningRate': {
        'distribution': 'uniform',
        'min': 1e-5,
        'max': 1e-2
        },
    'batchSize': {
        'values': list(range(2, 10, 2))
        },
    'num_epochs':{
        'values':[1,2,3]
    },
    'optimizer': {
        'values': [
            'sgd' ,
            'adam'#,
            #'RMSprop',
            #'adagrad'
        ]
    }
    }

sweep_config['parameters'] = parameters_dict       

# %%
key = 'c8f11a2349926152113dd98ada229184fe660459'
wandb.login(key = key)
# %%
# Assign device
assign_device()
sweep_id = wandb.sweep(sweep_config, project="DS_Project")

wandb.agent(sweep_id, train_model_sweep, count=100)
# %%