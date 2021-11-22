import argparse
import torch
from termcolor import colored
import colorama

colorama.init()

from neuralnet import Simple_CNN, classes
from preprocess_data import predict_preprocessing


def predict(model, image):
    model.load_state_dict(torch.load('./img_classifier.pt'))
    output = model(image)
    _, prediction = torch.max(output, 1)
    print(f'Prediction: {classes[prediction]}')
    if classes[prediction] == 'bad':
        print('Image quality: ' + colored('Low quality', 'red'))
    elif classes[prediction] == 'average':
        print('Image quality: ' + colored('Average quality', 'yellow'))
    elif classes[prediction] == 'good':
        print('Image quality: ' + colored('Good quality', 'green'))


def main():
    model = Simple_CNN()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--image",
                    required=True,
                    help="Provide full path to the image")
    args = vars(ap.parse_args())
    predict(model, predict_preprocessing(args['image']))


main()
