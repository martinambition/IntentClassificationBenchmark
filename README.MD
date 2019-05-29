# Intent Classification
Intent Classification is the import part of chatbot.
Chatbot must understand user's intention with very hight accuracy. Otherwise, "stupid", "useless" will be tagged to chatbot.

### I compare several approach for intent classification.
1. Simple sum of wordembeding,
2. Facebook InferSent.
3. TF-IDF
4. Semhash, https://arxiv.org/abs/1810.07150
## Accuracy Result

![](https://github.com/martinambition/IntentClassificationBenchmark/blob/master/images/Acc.png?raw=true)
![](https://github.com/martinambition/IntentClassificationBenchmark/blob/master/images/AccChart.png?raw=true)

Simple sum the word embeding achieve the best accuray in intention classification.
