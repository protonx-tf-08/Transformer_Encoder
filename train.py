# TODO 3
import tensorflow as tf
import os
from keras.datasets import imdb
from argparse import ArgumentParser
from models import TransformerClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import BinaryAccuracy
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--max_length", default=200,type=int)
    parser.add_argument("--vocab_size", default=10000, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--num_encoder_layers",default = 2, type=int)
    parser.add_argument("--learning_rate",default=0.001,type=float)
    parser.add_argument("--dropout-rate", default=0.1, type = float)

    home_dir = os.getcwd()
    args = parser.parse_args()

    num_encoder_layers = args.num_encoder_layers
    d_model = args.d_model
 
    # FIXME
    # Project Description

    print('---------------------Welcome to ProtonX Transformer Encoder-------------------')
    print('Github: Dungfx15018')
    print('Email: dungtrandinh513@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Transformer Classifier model with hyper-params:')
    print('===========================')

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")


    # Process data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=args.vocab_size)
    x_train = pad_sequences(x_train, maxlen = args.max_length)
    x_test = pad_sequences(x_test, maxlen= args.max_length)

    # Instantiate the model
    model = TransformerClassifier(
        num_encoder_layers = num_encoder_layers,
        d_model = d_model,
        num_heads = args.num_heads,
        input_vocab_size = args.vocab_size,
        maximum_position_encoding = args.max_length,
        dff = args.dff,
        rate = args.dropout_rate
    )
    # Compile the model
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer= tf.keras.optimizers.Adam(), metrics = ['BinaryAccuracy'] )

    # Train the model
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_test,y_test))
    # Evaluate the model
    test_loss,test_acc = model.evaluate(x_test, y_test)

    print(f'Test Loss: {test_loss}, Test_Accuracy: {test_acc}')
   