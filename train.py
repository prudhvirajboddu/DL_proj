import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS , OUT_DIR
from model import create_model
from utils import Averager
from datasets import train_loader, valid_loader


def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for _, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        a_images = list(image.to(DEVICE) for image in images)
        a_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        p_images = list(image.to(DEVICE) for image in images)
        p_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        n_images = list(image.to(DEVICE) for image in images)
        n_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        anchor = model(a_images, a_targets)
        positive = model(p_images, p_targets)
        negative = model(n_images, n_targets)

        anchor_loss = sum(loss for loss in anchor.values())
        positive_loss = sum(loss for loss in positive.values())
        negative_loss = sum(loss for loss in negative.values())

        losses = criterion(anchor_loss, positive_loss, negative_loss)
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for _, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            anchor_valid = model(images, targets)
            positive_valid = model(images, targets)
            negative_valid = model(images, targets)


        anchor_valid_loss = sum(loss for loss in anchor_valid.values())
        positive_valid_loss = sum(loss for loss in positive_valid.values())
        negative_valid_loss = sum(loss for loss in negative_valid.values())

        losses = criterion(anchor_valid_loss, positive_valid_loss, negative_valid_loss)
        loss_value = losses.item()
        val_loss_list.append(loss_value)

        val_loss_hist.send(loss_value)

        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


if __name__ == '__main__':
    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    #criterion
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), 
                                                margin=0.01 , swap = True, reduction='none')

    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=1e-9, momentum=0.02, weight_decay=0.000005)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

    # name to save the trained model with
    MODEL_NAME = 'model'

    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

    training = 'done'

    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()

    #save the model and plots after training

    if training == 'done': 
                train_ax.plot(train_loss, color='blue')
                train_ax.set_xlabel('Epochs')
                train_ax.set_ylabel('train loss')
                valid_ax.plot(val_loss, color='red')
                valid_ax.set_xlabel('Epochs')
                valid_ax.set_ylabel('validation loss')
                figure_1.savefig(f"{OUT_DIR}/train_loss.png")
                figure_2.savefig(f"{OUT_DIR}/valid_loss.png")

                torch.save(model.state_dict(), f"{OUT_DIR}/model.pth")