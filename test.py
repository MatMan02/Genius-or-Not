from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("model_ex-048_acc-0.791667.h5")
prediction.setJsonPath("model_class.json")
prediction.loadModel(num_objects=2)

predictions, probabilities = prediction.predictImage("pes.jpg", result_count=2)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
