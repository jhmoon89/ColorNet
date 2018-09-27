import numpy as np
import time, sys
import tensorflow as tf
import dataset_utils.dataset_loader.ImagenetVid_CN_dataset as ImagenetVid_CN_dataset
import src.module.colorizor as colorizor
import math
import datetime

CNConfig = {
    'classDim': 31,
    'consecutiveFrame': 2
}


def trainCNColorizor(
        CNConfig, batchSize=8, training_epoch=10,
        max_iter=10,
        learningRate=0.001,
        savePath=None, restorePath=None
):
    datasetPath = '/ssdubuntu/data/ILSVRC'
    dataset = ImagenetVid_CN_dataset.imagenetVidDataset(datasetPath,
                                                        consecutiveLength=CNConfig['consecutiveFrame'],
                                                        classNum=CNConfig['classDim'])
    dataset.setImageSize((224, 224))
    model = colorizor.colornet_colorizor(classNum=CNConfig['classDim'], consecutiveFrame=CNConfig['consecutiveFrame'])

    if restorePath != None:
        print 'restore weights...'
        model.restoreNetworks(restorePath)
        # model.restoreSecondCore(restorePath)

    loss = 0.0
    acc = 0.0
    epoch = 0
    iteration = 0
    run_time = 0.0
    if learningRate == None:
        learningRate = 0.001

    veryStart = time.time()
    print 'start training...'

    # while epoch < training_epoch:
    #     iteration = 0
    #     for cursor in range(max_iter):
    #         start = time.time()
    #         iteration = cursor
    #         batchData = dataset.getNextBatch(batchSize=batchSize)
    #         batchData['LearningRate'] = learningRate
    #         epochCurr = dataset._epoch
    #         dataStart = dataset._dataStart
    #         dataLength = dataset._dataLength
    #         if epochCurr != epoch:
    #             epoch = epochCurr
    #             break
    #
    #         lossTemp, accTemp = model.fit(batchData)
    #
    #         end = time.time()
    #         loss = float(loss * iteration + lossTemp) / float(iteration + 1.0)
    #         acc = float(acc * iteration + accTemp) / float(iteration + 1.0)
    #         run_time = (run_time * iteration + (end - start)) / float(iteration + 1.0)
    #
    #         sys.stdout.write(
    #             "Epoch:{:03d} iter:{:05d} runtime:{:.3f} ".format(int(epoch + 1), int(iteration + 1), run_time))
    #         sys.stdout.write("cur/tot:{:07d}/{:07d} ".format(dataStart, dataLength))
    #         # sys.stdout.write("Current Loss={:.6f} ".format(lossTemp))
    #         sys.stdout.write("Average Loss={:.6f} ".format(loss))
    #         sys.stdout.write("Average Acc={:.6f}% ".format(acc * 100))
    #         sys.stdout.write("\n")
    #         sys.stdout.flush()
    #
    #         if math.isnan(loss):
    #             break
    #
    #         if cursor != 0 and cursor % 2000 == 0:
    #             model.saveNetworks(savePath)
    #
    #     if math.isnan(loss):
    #         break
    #
    #     if savePath != None:
    #         print 'save model...'
    #         model.saveNetworks(savePath)
    #
    #     dataset.newEpoch()
    #     epoch += 1
    ###############################################################################################
    while epoch < training_epoch:

        start = time.time()
        batchData = dataset.getNextBatch(batchSize=batchSize)
        batchData['LearningRate'] = learningRate
        epochCurr = dataset._epoch
        dataStart = dataset._dataStart
        dataLength = dataset._dataLength

        if epochCurr != epoch or ((iteration + 1) % 1000 == 0 and (iteration + 1) != 1):
            print ''
            # iteration = 0
            # loss = loss * 0.0
            # run_time = 0.0
            if savePath != None:
                print 'save model...'
                model.saveNetworks(savePath)
        epoch = epochCurr

        lossTemp, accTemp = model.fit(batchData)

        end = time.time()
        loss = float(loss * iteration + lossTemp) / float(iteration + 1.0)
        acc = float(acc * iteration + accTemp) / float(iteration + 1.0)
        run_time = (run_time * iteration + (end - start)) / float(iteration + 1.0)

        sys.stdout.write(
            "Epoch:{:03d} iter:{:05d} runtime:{:.3f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:07d}/{:07d} ".format(dataStart, dataLength))
        # sys.stdout.write("Current Loss={:.6f} ".format(lossTemp))
        sys.stdout.write("Average Loss={:.6f} ".format(loss))
        sys.stdout.write("Average Acc={:.6f}% ".format(acc * 100))
        sys.stdout.write("\n")
        sys.stdout.flush()

        iteration = iteration + 1.0

    veryEnd = time.time()
    sys.stdout.write("total training time:" + str(datetime.timedelta(seconds=veryEnd - veryStart)))


if __name__ == "__main__":
    sys.exit(trainCNColorizor(
        CNConfig=CNConfig
        , batchSize=64
        , training_epoch=2
        , max_iter=20
        , learningRate=3e-5
        , savePath='/ssdubuntu/ColorNet_Weights/2frames/180926/1.lr1e-4epoch2_cell16'
        # , restorePath='/ssdubuntu/ColorNet_Weights/2frames/180926/1.lr1e-4epoch2_cell16'
    ))

