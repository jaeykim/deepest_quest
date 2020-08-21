import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# check if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("gpu is available")

def train(n_epochs, start_epoch, loaders, model, optimizer, criterion, use_cuda, filename, ensemble, debug):

    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    if use_cuda:
        # model = nn.DataParallel(model)
        model.cuda()

    for epoch in range(start_epoch, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            ## record the average training loss, using something like
            train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data - train_loss)

        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            if not debug:
                state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, 'model_{}.pt'.format(filename))

        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        ######################    
        # test the model #
        ######################
        # model.eval()
        # if use_cuda:
        #     model.cuda()
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            # max = output.data.max(1, keepdim=True)
            # print(max)
            # values = output.data.max(1, keepdim=True)[0].cpu().numpy().flatten()
            # preds = output.data.max(1, keepdim=True)[1].cpu().numpy().flatten()
            # print(values)
            # print(preds)
            # preds[values < 1] = -1
            # print(preds)
            
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

        print('-' * 100)
        print('Test Loss: {:.6f}'.format(test_loss))
        test_accuracy = 100. * correct / total
        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            test_accuracy, correct, total))
        print('-' * 100)

        if not ensemble and not debug:
            # state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            # torch.save(state, 'model_{}.pt'.format(filename))

            with open('./results_{}.txt'.format(filename), 'a') as file:
                file.write('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(
                epoch, 
                train_loss,
                valid_loss,
                test_loss,
                test_accuracy
                ))

    # return trained model
    return model

def inference(loaders, model, criterion, ssl_threshold, use_cuda):

    """returns the inferred labels"""
    labels = np.array([], dtype="i")

    model.eval()
    if use_cuda:
      model.cuda()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # print(data)
        # print(target)
        # print(target.values)
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # print(data)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # convert output probabilities to predicted class
        # max = output.data.max(1, keepdim=True)
        # print(max)
        values = output.data.max(1, keepdim=True)[0].cpu().numpy().flatten()
        preds = output.data.max(1, keepdim=True)[1].cpu().numpy().flatten()
        # print(values)
        # print(labels)
        preds[values < np.mean(values)] = -1
        # print(labels)

        # preds = output.data.max(1, keepdim=False)[1].cpu().numpy()
        # print(batch_idx, preds)
        labels = np.concatenate((labels, preds))
    return labels

def ensemble_train(n_epochs, start_epoch, en_loaders, en_models, en_optimizers, n_ensemble, criterion, use_cuda, filename, debug):

    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_mins = [np.Inf, np.Inf, np.Inf]
    valid_losses = np.zeros(3)
    train_losses = np.zeros(3)

    if use_cuda:
        # model = nn.DataParallel(model)
        for model in en_models:
            model.cuda()

    for epoch in range(start_epoch, n_epochs+1):
        print('=' * 100)
        for i in range(n_ensemble):
            print('[model {}]'.format(i))
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            ###################
            # train the model #
            ###################
            en_models[i].train()
            for batch_idx, (data, target) in enumerate(en_loaders[i]['train']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                en_optimizers[i].zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = en_models[i](data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                en_optimizers[i].step()
                # update training loss
                ## record the average training loss, using something like
                train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data - train_loss)

            ######################    
            # validate the model #
            ######################
            en_models[i].eval()
            for batch_idx, (data, target) in enumerate(en_loaders[i]['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                # forward pass: compute predicted outputs by passing inputs to the model
                output = en_models[i](data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)
                
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))
            
            valid_losses[i] = valid_loss
            train_losses[i] = train_loss

            ## TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_mins[i]:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_mins[i],
                valid_loss))
                valid_loss_mins[i] = valid_loss
                if not debug:
                    state = {'epoch': epoch + 1, 'state_dict': en_models[i].state_dict(), 'optimizer': en_optimizers[i].state_dict()}
                    torch.save(state, 'model_{}_{}.pt'.format(filename, i))

        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        ######################    
        # test the model #
        ######################
        # model.eval()
        # if use_cuda:
        #     model.cuda()
        for batch_idx, (data, target) in enumerate(en_loaders[0]['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = []
            for i in range(n_ensemble):
                outputs.append(en_models[i](data))
            # print(outputs)
            output = (outputs[0] + outputs[1] + outputs[2]) / 3
            # print(output)
            # calculate the loss
            loss = criterion(output.cuda(), target)
            # update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

        print('-' * 100)
        print('Test Loss: {:.6f}'.format(test_loss))
        test_accuracy = 100. * correct / total
        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            test_accuracy, correct, total))

        if not debug:
            # state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            # torch.save(state, 'model_{}.pt'.format(filename))

            with open('./results_{}.txt'.format(filename), 'a') as file:
                file.write('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(
                epoch, 
                np.mean(train_losses),
                np.mean(valid_losses),
                test_loss,
                test_accuracy
                ))

    # return trained model
    return en_models

def ensemble_inference(loaders, en_models, n_ensemble, criterion, ssl_threshold, use_cuda):

    """returns the inferred labels"""
    labels = np.array([], dtype="i")

    for model in en_models:
            model.eval()
    if use_cuda:
        for model in en_models:
            model.cuda()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # print(data)
        # print(target)
        # print(target.values)
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # print(data)
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = []
        for i in range(3):
            outputs.append(en_models[i](data))
        # print(outputs)
        output = (outputs[0] + outputs[1] + outputs[2]) / 3

        # convert output probabilities to predicted class
        # max = output.data.max(1, keepdim=True)
        # print(max)
        values = output.data.max(1, keepdim=True)[0].cpu().numpy().flatten()
        preds = output.data.max(1, keepdim=True)[1].cpu().numpy().flatten()
        # print(values)
        # print(labels)
        preds[values < np.mean(values)] = -1
        # print(labels)

        # preds = output.data.max(1, keepdim=False)[1].cpu().numpy()
        # print(batch_idx, preds)
        labels = np.concatenate((labels, preds))
    return labels