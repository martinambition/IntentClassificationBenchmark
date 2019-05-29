# Intent Classification
Intent Classification is the import part of chatbot.
Chatbot must understand user's intention with very hight accuracy. Otherwise, "stupid", "useless" will be tagged to chatbot.

### I compare several approach for intent classification.
1. Simple sum of wordembeding,
2. Facebook InferSent.
3. TF-IDF
4. Semhash, https://arxiv.org/abs/1810.07150
## Accuracy Result

<img src="https://raw.githubusercontent.com/martinambition/IntentClassificationBenchmark/master/images/Acc.png" alt="alt text" width="150" >
<img src="https://raw.githubusercontent.com/martinambition/IntentClassificationBenchmark/master/images/AccChart.png" alt="alt text" width="150" >

## Conclusion
Simple sum the word embeding achieve the best accuray in intention classification.
