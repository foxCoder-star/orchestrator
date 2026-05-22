from pipeline import NLPBrain

nlp = NLPBrain()

while True:
    text = input("You: ")
    result = nlp.predict(text)
    print(result)