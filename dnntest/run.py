import argparse
import cv2
from datetime import datetime
from keras.datasets import mnist,cifar10
from keras.models import model_from_json, load_model, save_model
from keras.datasets import cifar10
import scipy
#from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import save_layerwise_relevances, load_layerwise_relevances
from utils import save_layer_outs, load_layer_outs, get_layer_outs_new
from utils import save_data, load_data, save_quantization, load_quantization
from utils import generate_adversarial, filter_correct_classifications
from coverages.idc import ImportanceDrivenCoverage
from coverages.neuron_cov import NeuronCoverage
from coverages.tkn import DeepGaugeLayerLevelCoverage
from coverages.kmn import DeepGaugePercentCoverage
from coverages.ss import SSCover
from coverages.sa import SurpriseAdequacy
from lrp_toolbox.model_io import write, read
import tensorflow as tf
import numpy as np
from keras.applications.resnet50 import ResNet50
__version__ = 0.9
from keras.utils import np_utils

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def by_indices(outs, indices):
    return [[outs[i][0][indices]] for i in range(len(outs))]


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Coverage Analyzer for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.")#, required=True)
                        # choices=['lenet1','lenet4', 'lenet5'], required=True)
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        imagenet, or cifar10).", choices=["mnist","cifar10", "imagenet"])#, required=True)
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to measure coverage", choices=['idc','nc','kmnc',
                        'nbc','snac','tknc','ssc', 'lsa', 'dsa'])
    parser.add_argument("-C", "--class", help="the selected class", type=int)
    parser.add_argument("-Q", "--quantize", help="quantization granularity for \
                        combinatorial other_coverage_metrics.", type= int)
    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)
    parser.add_argument("-KS", "--k_sections", help="number of sections used in \
                        k multisection other_coverage_metrics", type=int)
    parser.add_argument("-KN", "--k_neurons", help="number of neurons used in \
                        top k neuron other_coverage_metrics", type=int)
    parser.add_argument("-RN", "--rel_neurons", help="number of neurons considered\
                        as relevant in combinatorial other_coverage_metrics", type=int)
    parser.add_argument("-AT", "--act_threshold", help="a threshold value used\
                        to consider if a neuron is activated or not.", type=float)
    parser.add_argument("-R", "--repeat", help="index of the repeating. (for\
                        the cases where you need to run the same experiments \
                        multiple times)", type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-ADV", "--advtype", help="path to log file")
    parser.add_argument("-var", "--variance", help="variance.", type= float)

    # parse command-line arguments


    # parse command-line arguments
    args = parser.parse_args()

    return vars(args)
import keras
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import keras

def train_y(y):
    y_one = np.zeros(10)
    y_one[y] = 1
    return y_one
if __name__ == "__main__":
    args = parse_arguments()
    model_path     = args['model'] if args['model'] else 'neural_networks/cifar10_resnet56'
    dataset        = args['dataset'] if args['dataset'] else 'cifar10'
    approach       = args['approach'] if args['approach'] else 'idc'
    num_rel_neurons= args['rel_neurons'] if args['rel_neurons'] else 2
    act_threshold  = args['act_threshold'] if args['act_threshold'] else 0
    top_k          = args['k_neurons'] if args['k_neurons'] else 3
    k_sect         = args['k_sections'] if args['k_sections'] else 1000
    selected_class = args['class'] if not args['class']==None else -1 #ALL CLASSES
    repeat         = args['repeat'] if args['repeat'] else 1
    logfile_name   = args['logfile'] if args['logfile'] else 'result.log'
    quantization_granularity = args['quantize'] if args['quantize'] else 3
    adv_type      = args['advtype'] if args['advtype'] else 'fgsm'
    var         = args['variance'] if args['variance'] else 0
    logfile = open(logfile_name, 'a')


    ####################
    # 0) Load data
    if dataset == 'mnist':
        '''
        X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
        print(X_train.shape)
        img_rows, img_cols = 28, 28
        if not selected_class == -1:
            X_train, Y_train = filter_val_set(selected_class, X_train, Y_train)  # Get training input for selected_class
            X_test, Y_test = filter_val_set(selected_class, X_test, Y_test)  # Get testing input for selected_class
        '''
        nb_classes = 10

        # the data, shuffled and split between tran and test sets
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train,X_test,Y_train,Y_test = train_test_split(X_test, Y_test,test_size = 0.1)
        print("X_train original shape", X_train.shape)
        print("Y_train original shape", Y_train.shape)

        X_train = X_train.reshape(9000, 28,28,1)
        X_test = X_test.reshape(1000, 28,28,1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print("Training matrix shape", X_train.shape)
        print("Testing matrix shape", X_test.shape)

        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)

    if dataset == 'fashionmnist':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        print(X_train.shape)
        img_rows, img_cols = 28, 28
        X_train = X_train.reshape(-1,28,28,1)
        X_test = X_test.reshape(-1,28,28,1)
    if dataset == 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train,X_test,Y_train,Y_test = train_test_split(X_test, Y_test,test_size = 0.1)
        X_train,X_test,Y_train,Y_test = train_test_split(X_test, Y_test,test_size = 0.1)
        num_classes = 10
        # Input image dimensions.
        input_shape = X_train.shape[1:]
        img_rows, img_cols = 32, 32
        # Normalize data.
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        subtract_pixel_mean = True
        if subtract_pixel_mean:
            X_train_mean = np.mean(X_train, axis=0)
            X_train -= X_train_mean
            X_test -= X_train_mean

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        print('Y_train shape:', Y_train.shape)

        # Convert class vectors to binary class matrices.
        Y_train = keras.utils.to_categorical(Y_train, num_classes)
        Y_test = keras.utils.to_categorical(Y_test, num_classes)
    # else:
    #     X_train, Y_train, X_test, Y_test = load_CIFAR()
    #     img_rows, img_cols = 32, 32

    # Prepare imagenet dataset
    if dataset == "imagenet":
        train_datagen = ImageDataGenerator(
        rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
                '../imagenet/train',  # this is the target directory
                target_size=(224, 224),  # all images will be resized to 150x150
                batch_size=100,
                class_mode='categorical')
        X_train,Y_train = train_generator.next()
        X_test,Y_test = train_generator.next()
    ####################
    # 1) Setup the model

    if model_path == "resnet":
        model = ResNet50(weights="imagenet")
    else:
        model_name = model_path.split('/')[-1]
        try:
            json_file = open(model_path + '.json', 'r') #Read Keras model parameters (stored in JSON file)
            file_content = json_file.read()
            json_file.close()

            model = model_from_json(file_content)
            model.load_weights(model_path + '.h5')

            # Compile the model before using
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

        except:
            model = load_model(model_path + '.h5')


    # 2) Load necessary information
    trainable_layers = get_trainable_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'

    # Investigate the penultimate layer
    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]
    # SKIP LAYERS FOR NC, KMNC, NBC etc.
    skip_layers = [0]
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print(skip_layers)
    print(Y_train[0])
    print(Y_test[0])
    ####################
    # 3) Analyze Coverages
    X_test_ = X_test
    if approach == 'nc':
        # SKIP ONLY INPUT AND FLATTEN LAYERS
        if model_path == "resnet":
            X_test = np.array(X_test)
        variance = [0,0.1,1,5,10]
        threshold = [0,0.25,0.5,0.75]
        ACC = []
        Coverage = []
        for j in range(0,4):
            X_test = X_test_
            nc = NeuronCoverage(model, threshold=threshold[j], skip_layers = skip_layers)
            for i in range(0,5):
                X_test = X_test + np.random.normal(loc=0, scale=variance[i], size=X_test.shape)
                coverage, _, _, _, _ = nc.test(X_test)
                print("Your test set's coverage is: ", coverage)
                score=model.evaluate(X_test,Y_test)
                ACC.append(score[1])
                Coverage.append(coverage)
        nc.set_measure_state(nc.get_measure_state())
        print(ACC)
        print(Coverage)
    elif approach == 'idc':
        print(model.summary())
        X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model,
                                                                           X_train,
                                                                           Y_train)
        
        cc = ImportanceDrivenCoverage(model, model_name, num_rel_neurons, selected_class,
                          subject_layer, X_train_corr, Y_train_corr)#,
                          #quantization_granularity)
        variance = [0,0.1,1,5,10]
        ACC = []
        Coverage = []
        for i in range(0,5):
            X_test = X_test + np.random.normal(loc=0.0, scale=variance[i], size=X_test.shape)
            coverage, covered_combinations = cc.test(X_test)
            print("Your test set's coverage is: ", coverage)
            print("Number of covered combinations: ", len(covered_combinations))
            score=model.evaluate(X_test,Y_test)
            ACC.append(score[1])
            Coverage.append(coverage)
        cc.set_measure_state(covered_combinations)
        print(ACC)
        print(Coverage)
    elif approach == 'kmnc' or approach == 'nbc' or approach == 'snac':

        res = {
            'model_name': model_name,
            'num_section': k_sect,
        }

        dg_ = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
        X_test = X_test + np.random.normal(loc=0.0, scale=var, size=X_test.shape)
        score = dg_.test(X_test)
        kmnc_prcnt = score[0]
        nbc_prcnt = score[2]
        snac_prcnt = score[3]
        print("Your test set's kmnc_prcnt coverage is: ", kmnc_prcnt)
        print("Your test set's nbc_prcnt coverage is: ", nbc_prcnt)
        print("Your test set's snac_prcnt coverage is: ", snac_prcnt)
    elif approach == 'tknc':
        ACC = []
        tknccoverage = []
        kmnc_prcntcoverage = []
        nbc_prcntcoverage = []
        snac_prcntcoverage = []
        
        res = {
            'model_name': model_name,
            'num_section': k_sect,
        }
        
        variance = [0,0.1,1,5,10]
        for i in range(0,5):
            dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=skip_layers)
            dg_ = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
            var = variance[i]
            X_test = X_test + np.random.normal(loc=0.0, scale=var, size=X_test.shape)
            score=model.evaluate(X_test,Y_test)
            ACC.append(score[1])
            orig_coverage, _, _, _, _, orig_incrs = dg.test(X_test)
            tknccoverage.append(orig_coverage)
            kmnc = dg_.test(X_test)
            kmnc_prcntcoverage.append(kmnc[0])
            nbc_prcntcoverage.append(kmnc[2])
            snac_prcntcoverage.append(kmnc[3])
        print(ACC)
        print("Your test set's tknccoverage coverage is: ",tknccoverage)
        print("Your test set's kmnc_prcntcoverage coverage is: ",kmnc_prcntcoverage)
        print("Your test set's nbc_prcntcoverage coverage is: ",nbc_prcntcoverage)
        print("Your test set's snac_prcntcoverage coverage is: ",snac_prcntcoverage)
    elif approach == 'ssc':

        ss = SSCover(model, skip_layers=non_trainable_layers)
        score = ss.test(X_test)

        print("Your test set's coverage is: ", score)

    elif approach == 'lsa' or approach == 'dsa':
        upper_bound = 2000

        layer_names = [model.layers[-3].name]

        #for lyr in model.layers:
        #    layer_names.append(lyr.name)

        sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, dataset)
        sa.test(X_test, approach)

    logfile.close()

