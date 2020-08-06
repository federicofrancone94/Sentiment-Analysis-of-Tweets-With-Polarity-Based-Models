import java.io.File;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

public class TweetClassification_learn {
	public static void main(String[] args) throws Exception {

		if (args.length != 3) {
			System.err.println("Usage: training_set_path json_file_path output_model_file_path");
			//return;
		}

		//String trainingSetFilePath = args[0];
		//String jsonAlgorithmPath = args[1];
		//String modelFilePath = args[2];

		//String trainingSetFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\train.klp";
		String trainingSetFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\inputs_kelp\\train_v4_final.klp";
		//String jsonAlgorithmPath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\kelp_engine\\json\\ova_rbf_polb.json";

		String jsonAlgorithmPath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\kelp_engine\\json\\ova_invented_4kernel.json";
		String modelFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\models_learnt\\invented_4kernel_final.model";

		//polarity model
		// String modelFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\polarity.model" ;

		// Read the training and test dataset
		SimpleDataset trainingSet = new SimpleDataset();
		trainingSet.populate(trainingSetFilePath);

		// Loading the classifier from the JSON file
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		OneVsAllLearning ovaLearner = serializer.readValue(new File(jsonAlgorithmPath), OneVsAllLearning.class);
		ovaLearner.setLabels(trainingSet.getClassificationLabels());

		// Learn and get the prediction function
		ovaLearner.learn(trainingSet);
		Classifier classifier = ovaLearner.getPredictionFunction();
		
		// Save the model on file
		JacksonSerializerWrapper modelSerializer = new JacksonSerializerWrapper();
		modelSerializer.writeValueOnFile(classifier, modelFilePath);

	}

}
