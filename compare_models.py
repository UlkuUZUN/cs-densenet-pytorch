import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


import densenet_cycle_spinning
import densenet

def main():
    batch_size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print("device: ", device)

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(root='./../data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # model = densenet.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,bottleneck=args.bottleneck, dropRate=args.droprate)
    model = densenet.DenseNet3(100, 10, 12, reduction=0.5, bottleneck=True, dropRate=0)
    # model = VGG_13_bn()

    #checkpoint = torch.load('C:/Users/yigit/PycharmProjects/densenet-pytorch-master/runs/original/checkpoint.pth.tar')
    checkpoint = torch.load(F"/content/gdrive/My Drive/densenet-pytorch-master/runs/original/checkpoint.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    model_cycle = densenet_cycle_spinning.DenseNet3(100, 10, 12, reduction=0.5, bottleneck=True, dropRate=0)
    # model_cycle = VGG_13_bn_cycle()
   # checkpoint_cycle = torch.load('C:/Users/yigit/PycharmProjects/densenet-pytorch-master/runs/roll1shift_withAug/checkpoint.pth.tar')
    checkpoint_cycle = torch.load(F"/content/gdrive/My Drive/densenet-pytorch-master/runs/DenseNet_BC_100_12/checkpoint.pth.tar")
    model_cycle.load_state_dict(checkpoint_cycle['state_dict'])
    model_cycle.to(device)
    model_cycle.eval()


    def get_accuracy_and_consistency_constantShift(network, originalImage, prediction, target, shift):
        pad = (shift, 0, 0, 0)
        data_padded = nn.functional.pad(originalImage, pad, mode='constant')
        shifted_data = data_padded[:, :, :, 0:-shift]

        output_shifted_data = network(shifted_data)
        prediction_shifted_data = output_shifted_data.argmax(dim=1, keepdim=True)

        accuracy = target.eq(prediction_shifted_data.view_as(target)).sum().item()
        consistency = prediction.eq(prediction_shifted_data.view_as(prediction)).sum().item()

        return accuracy, consistency

    def get_accuracy_and_consistency_circularShift(network, originalImage, prediction, target, shift):
        pad = (shift, 0, 0, 0)
        data_padded = nn.functional.pad(originalImage, pad, mode='circular')
        shifted_data = data_padded[:, :, 0:-shift, :]

        output_shifted_data = network(shifted_data)
        prediction_shifted_data = output_shifted_data.argmax(dim=1, keepdim=True)

        accuracy = target.eq(prediction_shifted_data.view_as(target)).sum().item()
        consistency = prediction.eq(prediction_shifted_data.view_as(prediction)).sum().item()

        return accuracy, consistency

    def get_accuracy_and_consistency_reflectShift(network, originalImage, prediction, target, shift):
        pad = (shift, 0, 0, 0)
        data_padded = nn.functional.pad(originalImage, pad, mode='reflect')
        shifted_data = data_padded[:, :, 0:-shift, :]

        output_shifted_data = network(shifted_data)
        prediction_shifted_data = output_shifted_data.argmax(dim=1, keepdim=True)

        accuracy = target.eq(prediction_shifted_data.view_as(target)).sum().item()
        consistency = prediction.eq(prediction_shifted_data.view_as(prediction)).sum().item()

        return accuracy, consistency

    def get_accuracy_and_consistency_mirrorShift(network, originalImage, prediction, target, shift):

        pad = (shift, 0, 0, 0)
        data_padded = nn.functional.pad(originalImage, pad, mode='replicate')

        shifted_data = data_padded[:, :, :, 0:-shift]


        output_shifted_data = network(shifted_data)
        prediction_shifted_data = output_shifted_data.argmax(dim=1, keepdim=True)

        accuracy = target.eq(prediction_shifted_data.view_as(target)).sum().item()
        consistency = prediction.eq(prediction_shifted_data.view_as(prediction)).sum().item()

        return accuracy, consistency

    def get_accuracy_and_consistency_rollShift(network, originalImage, prediction, target, shift):
        shifted_data = torch.roll(originalImage, shifts=(shift, shift), dims=(2, 3))

        output_shifted_data = network(shifted_data)
        prediction_shifted_data = output_shifted_data.argmax(dim=1, keepdim=True)

        accuracy = target.eq(prediction_shifted_data.view_as(target)).sum().item()
        consistency = prediction.eq(prediction_shifted_data.view_as(prediction)).sum().item()

        return accuracy, consistency
        
    def get_accuracy_and_consistency_replicateShift(network, originalImage, prediction, target, shift):
        pad = (shift, 0, 0, 0)
        data_padded = nn.functional.pad(originalImage, (pad[0], pad[1], pad[2], pad[3]), mode='replicate')
        shifted_data = data_padded[:, :, 0: 32, 0: 32]
        output_shifted_data = network(shifted_data)
        prediction_shifted_data = output_shifted_data.argmax(dim=1, keepdim=True)

        accuracy = target.eq(prediction_shifted_data.view_as(target)).sum().item()
        consistency = prediction.eq(prediction_shifted_data.view_as(prediction)).sum().item()

        return accuracy, consistency

    total_shift = 33

    accuracies_model = np.zeros(total_shift + 1)
    consistencies_model = np.zeros(total_shift + 1)

    accuracies_model_cycle = np.zeros(total_shift + 1)
    consistencies_model_cycle = np.zeros(total_shift + 1)

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)

            output_data = model(data)
            prediction_data = output_data.argmax(dim=1, keepdim=True)
            accuracies_model[0] += target.eq(prediction_data.view_as(target)).sum().item()
            consistencies_model[0] += len(data)

            for i in range(1, total_shift + 1):
                accuracy, consistency = get_accuracy_and_consistency_replicateShift(model, data, prediction_data, target,
                                                                               i)

                # print(' {0} \t'
                #       ' {1} \t'
                #       ' {2} \t'
                #       ' {3} '.
                #
                #     format(
                #     i, len(testloader),
                #     accuracy,
                #     consistency
                # ))
                accuracies_model[i] += accuracy
                consistencies_model[i] += consistency

            output_data = model_cycle(data)
            prediction_data = output_data.argmax(dim=1, keepdim=True)
            accuracies_model_cycle[0] += target.eq(prediction_data.view_as(target)).sum().item()
            consistencies_model_cycle[0] += len(data)

            for i in range(1, total_shift + 1):
                accuracy, consistency = get_accuracy_and_consistency_replicateShift(model_cycle, data, prediction_data,
                                                                                 target,
                                                                                 i)
                # print(' {0} \t'
                #       ' {1} \t'
                #       ' {2} \t'
                #       ' {3} '.
                #
                #     format(
                #     i, len(testloader),
                #     accuracy,
                #     consistency
                # ))



                accuracies_model_cycle[i] += accuracy
                consistencies_model_cycle[i] += consistency

            if batch_id % 10 == 9:
                print("id: ", batch_id * batch_size + 1)

    accuracies_model = accuracies_model / len(testloader.dataset)
    consistencies_model = consistencies_model / len(testloader.dataset)

    accuracies_model_cycle = accuracies_model_cycle / len(testloader.dataset)
    consistencies_model_cycle = consistencies_model_cycle / len(testloader.dataset)

    print((accuracies_model_cycle[1] - accuracies_model[1]) * 100)
    print((consistencies_model_cycle[1] - consistencies_model[1]) * 100)

    print((accuracies_model_cycle[2] - accuracies_model[2]) * 100)
    print((consistencies_model_cycle[2] - consistencies_model[2]) * 100)

    print((accuracies_model_cycle[3] - accuracies_model[3]) * 100)
    print((consistencies_model_cycle[3] - consistencies_model[3]) * 100)

    print((accuracies_model_cycle[4] - accuracies_model[4]) * 100)
    print((consistencies_model_cycle[4] - consistencies_model[4]) * 100)

    print((accuracies_model_cycle[5] - accuracies_model[5]) * 100)
    print((consistencies_model_cycle[5] - consistencies_model[5]) * 100)

    print((accuracies_model_cycle[6] - accuracies_model[6]) * 100)
    print((consistencies_model_cycle[6] - consistencies_model[6]) * 100)

    print((accuracies_model_cycle[7] - accuracies_model[7]) * 100)
    print((consistencies_model_cycle[7]-consistencies_model[7])*100)

    print((accuracies_model_cycle[8] - accuracies_model[8]) * 100)
    print((consistencies_model_cycle[8] - consistencies_model[8]) * 100)

    print((accuracies_model_cycle[9] - accuracies_model[9]) * 100)
    print((consistencies_model_cycle[9] - consistencies_model[9]) * 100)

    print((accuracies_model_cycle[10] - accuracies_model[10]) * 100)
    print((consistencies_model_cycle[10] - consistencies_model[10]) * 100)

    print((accuracies_model_cycle[11] - accuracies_model[11]) * 100)
    print((consistencies_model_cycle[11] - consistencies_model[11]) * 100)

    print((accuracies_model_cycle[12] - accuracies_model[12]) * 100)
    print((consistencies_model_cycle[12] - consistencies_model[12]) * 100)

    print((accuracies_model_cycle[13] - accuracies_model[13]) * 100)
    print((consistencies_model_cycle[13] - consistencies_model[13]) * 100)

    print((accuracies_model_cycle[14] - accuracies_model[14]) * 100)
    print((consistencies_model_cycle[14] - consistencies_model[14]) * 100)

    print((accuracies_model_cycle[15] - accuracies_model[15]) * 100)
    print((consistencies_model_cycle[15] - consistencies_model[15]) * 100)

    print((accuracies_model_cycle[16] - accuracies_model[16]) * 100)
    print((consistencies_model_cycle[16] - consistencies_model[16]) * 100)

    print((accuracies_model_cycle[17] - accuracies_model[17]) * 100)
    print((consistencies_model_cycle[17]-consistencies_model[17])*100)

    print((accuracies_model_cycle[18] - accuracies_model[18]) * 100)
    print((consistencies_model_cycle[18] - consistencies_model[18]) * 100)

    print((accuracies_model_cycle[19] - accuracies_model[19]) * 100)
    print((consistencies_model_cycle[19] - consistencies_model[19]) * 100)
    
    print((accuracies_model_cycle[20] - accuracies_model[20]) * 100)
    print((consistencies_model_cycle[20] - consistencies_model[20]) * 100)

    print((accuracies_model_cycle[21] - accuracies_model[21]) * 100)
    print((consistencies_model_cycle[21] - consistencies_model[21]) * 100)

    print((accuracies_model_cycle[22] - accuracies_model[22]) * 100)
    print((consistencies_model_cycle[22] - consistencies_model[22]) * 100)

    print((accuracies_model_cycle[23] - accuracies_model[23]) * 100)
    print((consistencies_model_cycle[23] - consistencies_model[23]) * 100)

    print((accuracies_model_cycle[24] - accuracies_model[24]) * 100)
    print((consistencies_model_cycle[24] - consistencies_model[24]) * 100)

    print((accuracies_model_cycle[25] - accuracies_model[25]) * 100)
    print((consistencies_model_cycle[25] - consistencies_model[25]) * 100)

    print((accuracies_model_cycle[26] - accuracies_model[26]) * 100)
    print((consistencies_model_cycle[26] - consistencies_model[26]) * 100)

    print((accuracies_model_cycle[27] - accuracies_model[27]) * 100)
    print((consistencies_model_cycle[27]-consistencies_model[27])*100)

    print((accuracies_model_cycle[28] - accuracies_model[28]) * 100)
    print((consistencies_model_cycle[28] - consistencies_model[28]) * 100)

    print((accuracies_model_cycle[29] - accuracies_model[29]) * 100)
    print((consistencies_model_cycle[29] - consistencies_model[29]) * 100)
    
    print((accuracies_model_cycle[30] - accuracies_model[30]) * 100)
    print((consistencies_model_cycle[30]-consistencies_model[30])*100)

    print((accuracies_model_cycle[31] - accuracies_model[31]) * 100)
    print((consistencies_model_cycle[31] - consistencies_model[31]) * 100)

    print((accuracies_model_cycle[32] - accuracies_model[32]) * 100)
    print((consistencies_model_cycle[32] - consistencies_model[32]) * 100)


    # %%

    mean_accuracy_model = np.mean(accuracies_model[1:]) * 100
    mean_accuracy_model_cycle = np.mean(accuracies_model_cycle[1:]) * 100

    mean_consistencies_model = np.mean(consistencies_model[1:]) * 100
    mean_consistencies_model_cycle = np.mean(consistencies_model_cycle[1:]) * 100

    print("mean standard accuracy: {:3.1f}".format(mean_accuracy_model) + " | mean cycle accuracy: {:3.1f}".format(
        mean_accuracy_model_cycle))
    print("mean standard consistency: {:3.1f}".format(
        mean_consistencies_model) + " | mean cycle consistency: {:3.1f}".format(mean_consistencies_model_cycle))

    # %%
    plt.figure(figsize=(8, 8))

    plt.subplot(211)
    plt.grid()
    plt.plot(accuracies_model * 100, 'b', label='Standard Model')
    plt.plot(accuracies_model_cycle * 100, 'r', label='Cycle Model')
    plt.xlabel("Shift")
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(212)
    plt.grid()
    plt.plot(consistencies_model * 100, 'b', label='Standard Model')
    plt.plot(consistencies_model_cycle * 100, 'r', label='Cycle Model')
    plt.xlabel("Shift")
    plt.ylabel('Consistency')
    plt.legend()

    plt.suptitle('shift', y=1, fontsize=14)
    from google.colab import files
    plt.savefig("abc.png")
    files.download("abc.png")   


if __name__ == '__main__':
    main()