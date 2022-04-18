def runProgram(inputTxt):
    import numpy as np
    import pickle
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from keras.models import Sequential, load_model
    from keras.layers import Dense, LSTM, Embedding, LeakyReLU, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten, Reshape
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import pandas as pd
    import random
    import tkinter as tk
    import tensorflow as tf




    ENGLISH_MODEL_PATH =  '/Users/nitin/Desktop/pythonFlaskAPP/data/english-20-model.h5'
    ENGLISH_TEXT_PATH = '/Users/nitin/Desktop/pythonFlaskAPP/data/AdventrueOfSherlockHolmes.txt'
    ENGLISH_TOKENIZER_PATH = '/Users/nitin/Desktop/pythonFlaskAPP/data/english-20-tokenizer.pkl'


    description = 'english-20'




    # load english
    in_filename = ENGLISH_TEXT_PATH
    with open(in_filename) as f:
        doc = f.read()
    lines = doc.split('\n')
    lines = [' '.join(l.split(' ')[:20]) for l in lines[:4000]]




    # load the tokenizer
    tokenizer = pickle.load(open(ENGLISH_TOKENIZER_PATH, 'rb'))



    def get_sequence_of_tokens(corpus, tokenizer):
        ## convert data to sequence of tokens 
        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        total_words = len(tokenizer.word_index) + 1
        return input_sequences, total_words




    X, vocab_size = get_sequence_of_tokens(lines, tokenizer)




    def generate_padded_sequences(input_sequences, total_words):
        max_sequence_len = max([len(x) for x in input_sequences])
        print(max_sequence_len)
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = to_categorical(label, num_classes=total_words)
        return predictors, label, max_sequence_len

    X, y, seq_length = generate_padded_sequences(X, vocab_size)




    model = load_model(ENGLISH_MODEL_PATH)


    result = list()
    in_text = inputTxt
    # generate a fixed number of words
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # truncate sequences to a fixed length
    #encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    encoded = pad_sequences([encoded], maxlen=19, truncating='pre')
    # predict probabilities for each word
    # yhat = self.model.predict_classes(encoded, verbose=0)
    predicted_l = list(tuple(enumerate(model.predict(encoded)[0])))
    top_3 = sorted(predicted_l, key=lambda x: x[1], reverse=True)[:3]
    print(top_3)
    # map predicted word index to word
    predicted_words = []
    for i, word in enumerate(top_3):
        for w in list(tokenizer.word_index.items()):
            if w[1] == word[0]:
                predicted_words.append({'word': w[0], 'probability': word[1]})
    
    
    return predicted_words



# In[56]:


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # truncate sequences to a fixed length
    #encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    encoded = tensorflow.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # predict probabilities for each word
    # yhat = self.model.predict_classes(encoded, verbose=0)
    predicted_l = list(tuple(enumerate(model.predict(encoded)[0])))
    top_3 = sorted(predicted_l, key=lambda x: x[1], reverse=True)[:3]
    print(top_3)
    # map predicted word index to word
    predicted_words = []
    for i, word in enumerate(top_3):
        for w in list(tokenizer.word_index.items()):
            if w[1] == word[0]:
                predicted_words.append({'word': w[0], 'probability': word[1]})
    return predicted_words

def tempRun(inputTxt):


    model = runProgram(inputTxt)
    makeitastring = ''.join(map(str, model))
    return makeitastring
    #generated = generate_seq(model, tokenizer, 19, "what is this" , 1)

def runAPP():
    
    import tkinter as tk


    model, tokenizer = runProgram()


    # Top level window
    frame = tk.Tk()
    frame.title("TextBox Input")
    frame.geometry('400x200')
    # Function for getting Input
    # from textbox and printing it 
    # at label widget

    def printInput():


        inp = inputtxt.get(1.0, "end-1c")


        generated = generate_seq(model, tokenizer, 19, inp , 1)

        makeitastring = ''.join(map(str, generated))


        lbl.config(text = "Provided Input: "+makeitastring)



    # TextBox Creation
    inputtxt = tk.Text(frame,
                       height = 5,
                       width = 20)

    inputtxt.pack()




    # Button Creation
    printButton = tk.Button(frame, text = "Print", command = printInput)
    printButton.pack()


    # Label Creation
    lbl = tk.Label(frame, text = "")
    lbl.pack()
    frame.mainloop()






from flask import Flask, request, render_template

app = Flask(__name__)

#@app.route("/")
@app.route('/', methods=['GET', 'POST'])
def server():
    if request.method == 'POST':
            # Then get the data from the form
        tag = request.form['fname']

        result = tempRun(tag)
        # Generate just a boring response
        #return "Your name is "+tag

        return render_template('index.html', dataToRender=result)
    return render_template('index.html') 
        # Or you could have a custom template for displaying the info
        # return render_template('asset_information.html',
        #                        username=user, 
        #                        password=password)

# def index():
#     return  render_template('index.html')#tempRun() 


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
