from wordpredictor import WordPredictor

predictor = WordPredictor()
predictor.loadtextfile('TheFestival.txt')
predictor.init()

prompt = input(f'> Enter {predictor.chain_length - 1} words: ')
while True:
    prediction = predictor.predict(prompt)
    print(f'\nPrompt: {prompt}')
    print(f'Prediction: {prediction}')
    prompt += ' ' + prediction
    print(prompt)
    i = input('> q to exit, enter to continue: ')
    if i == 'q':
        break