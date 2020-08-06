import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.standard.RbfKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.util.ArrayList;
import java.util.List;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Objects;

public class ModelSelectionLast {

	public static void main(String[] args) throws Exception {

		// set the train and dev file
		String trainingSetFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\inputs_kelp\\train_v4_final.klp";
		String devsetFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\inputs_kelp\\dev_v4_final.klp";
		//String trainingSetFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\inputs_kelp\\train.klp";
		//String devsetFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\inputs_kelp\\dev.klp";


		// kernel type - Insert the correct string to choose the kernel (simplelin, norm, poly, lin-comb, rbf)
		//String kernelType = "lin-comb";
		String kernelType = "bubu";  //non serve a nulla nel codice...si usa la lista dei kernels non questa

		//List<String> kernels = Arrays.asList("norm", "lin-comb", "poly", "rbf");
		//List<Float> cps = Arrays.asList(0.01f, 0.1f, 1.0f, 3.0f);
		//List<Float> cns = Arrays.asList(0.01f, 0.1f, 1.0f, 3.0f);

		//List<String> kernels = Arrays.asList("norm","lin-comb","poly", 'simplelin');
		List<String> kernels = Arrays.asList("invented_kernel");
		//List<String> kernels = Arrays.asList("lin-comb", "simplelin", "norm");  //non la uso per questo file, non cicla sui kernels...

		//List<Float> cps = Arrays.asList(0.001f, 0.005f, 0.01f, 0.05f, 0.1f, 0.5f, 1.0f, 5.0f, 10.0f);
		//List<Float> cns = Arrays.asList(0.001f, 0.005f, 0.01f, 0.05f, 0.1f, 0.5f, 1.0f, 5.0f, 10.0f);
		List<Float> cps = Arrays.asList(1.4f);
		List<Float> cns = Arrays.asList(1.15f);
		//List<Float> cns= cps;

		List<Object> results = new ArrayList<Object>();


		Float cp = 1.0f;
		Float cn = 1.0f;

		// degree of polinomial kernel


		int polDegree = 2;

		// parameter for Tree kernels
		float lambda = 0.4f;

		// radial basis kernel parameter
		float gamma = 1f;

		// set model and prediction files based on specific models and parameters. Questi servono senza Grid, se voglio fare modello secco
		//String modelFilePath = "/Users/giando/Google Drive/Courses/Master BIG DATA/Social Media-Basili/Final Project/BigData_2018-2019_TM_SMA_project/kelp_engine/model-" + kernelType + "_cp" + cp + "_cn" + cn + ".model";
		//String modelFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\kelp_engine\\lin_comb_model-" + kernelType + "_cp" + cp + "_cn" + cn + ".model";
		//String devOutputFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\predDev-"+ kernelType + "_cp" + cp + "_cn" + cn + ".out";
		//String trainOutputFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\predTrain-"+ kernelType + "_cp" + cp + "_cn" + cn + ".out";

		//String devOutputFilePath = "/Users/giando/Google Drive/Courses/Master BIG DATA/Social Media-Basili/Final Project/BigData_2018-2019_TM_SMA_project/resource/pred-Dev-" + kernelType + "_cp" + cp + "_cn" + cn + ".out";
		//String trainOutputFilePath = "/Users/giando/Google Drive/Courses/Master BIG DATA/Social Media-Basili/Final Project/BigData_2018-2019_TM_SMA_project/resource/pred-Train-" + kernelType + "_cp" + cp + "_cn" + cn + ".out";
		String resultFilePath = "C:\\Users\\feder\\Desktop\\Master\\Project_NLP\\resource\\outputs\\final_invented.out";

		// read dataset
		SimpleDataset trainingSet = new SimpleDataset();
		trainingSet.populate(trainingSetFilePath);
		System.out.println("The training set is made of " + trainingSet.getNumberOfExamples() + " examples.");

		SimpleDataset devSet = new SimpleDataset();
		devSet.populate(devsetFilePath);
		System.out.println("The test set is made of " + devSet.getNumberOfExamples() + " examples.");

		System.out.println("classification labels " + trainingSet.getClassificationLabels().toString());

		int count = 1;

		for (String x : kernels) {
			for (Float y : cps) {
				for (Float z : cns) {

					kernelType = x;
					cp = y;
					cn = z;
					//float z= cn;

					System.out.println("Kernel:" + x + " cp:" + y + " cn:" + z + " cicle:" + count);

					// calculating the size of the gram matrix to store all the examples
					int cacheSize = trainingSet.getNumberOfExamples() + devSet.getNumberOfExamples();
					System.out.println("Number of Examples is:" + cacheSize);

					// Initialize the proper kernel function
					Kernel usedKernel = null;

					// linear kernel
					if (kernelType.equalsIgnoreCase("simplelin")) {
						//String vectorRepresentationName = "bow";
						//String vectorRepresentationName = "polb";
						String vectorRepresentationName = "bigrams";
						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						usedKernel = linearKernel;

						// normalized kernel
					} else if (kernelType.equalsIgnoreCase("norm")) {
						//String vectorRepresentationName = "bow";
						//String vectorRepresentationName = "trigrams";
						String vectorRepresentationName = "bigrams";
						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
						usedKernel = normalizedLinearKernel;

						// polinomial kernel of degree exponent
					} else if (kernelType.equalsIgnoreCase("poly")) {
						//String vectorRepresentationName = "trigrams";
						String vectorRepresentationName = "bigrams";
						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
						Kernel polynomialKernel = new PolynomialKernel(polDegree, normalizedLinearKernel);
						usedKernel = polynomialKernel;

						// tree kernel
					} else if (kernelType.equalsIgnoreCase("tk")) {
						String treeRepresentationName = "grct";
						Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);
						usedKernel = tkgrct;

						//  kernel linear combination
					} else if (kernelType.equalsIgnoreCase("lin-comb")) {
						//String vectorRepresentationName = "bow";
						String vectorRepresentationName = "polb";
						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);

						String vectorRepresentationName2 = "bow_nvj";
						Kernel linearKernel2 = new LinearKernel(vectorRepresentationName2);
						Kernel normalizedLinearKernel2 = new NormalizationKernel(linearKernel2);

						LinearKernelCombination combination = new LinearKernelCombination();
						combination.addKernel(0.5f, normalizedLinearKernel);
						combination.addKernel(1.0f, normalizedLinearKernel2);

						usedKernel = combination;

					} else if (kernelType.equalsIgnoreCase("invented_kernel")) {
						//String vectorRepresentationName = "bow";
						String vectorRepresentationName = "bow_nvj";
						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);

						String vectorRepresentationName2 = "polb";
						Kernel linearKernel2 = new LinearKernel(vectorRepresentationName2);
						Kernel normalizedLinearKernel2 = new NormalizationKernel(linearKernel2);

						String vectorRepresentationName3 = "pol_smiles";
						Kernel linearKernel3 = new LinearKernel(vectorRepresentationName3);
						Kernel normalizedLinearKernel3 = new NormalizationKernel(linearKernel3);

						String vectorRepresentationName4 = "bigrams";
						Kernel linearKernel4 = new LinearKernel(vectorRepresentationName4);
						Kernel normalizedLinearKernel4 = new NormalizationKernel(linearKernel4);

						LinearKernelCombination combination = new LinearKernelCombination();
						combination.addKernel(0.9f, normalizedLinearKernel);
						combination.addKernel(1.0f, normalizedLinearKernel2);
						combination.addKernel(0.7f, normalizedLinearKernel3);
						combination.addKernel(0.4f, normalizedLinearKernel4);

						usedKernel = combination;
						//  kernel RBF


						//  kernel RBF
					} else if (kernelType.equalsIgnoreCase("rbf")) {
						String vectorRepresentationName = "trigrams";
						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
						Kernel rbfKernel = new RbfKernel(gamma, normalizedLinearKernel);
						usedKernel = rbfKernel;

						// combination kernel (Bag of Words and Trees)
					} else if (kernelType.equalsIgnoreCase("comb")) {
						String vectorRepresentationName = "bow";
						String treeRepresentationName = "grct";

						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);

						LinearKernelCombination combination = new LinearKernelCombination();
						combination.addKernel(1, linearKernel);
						combination.addKernel(1, tkgrct);
						usedKernel = combination;

						// normalized combination kernel (Bag of Words and Trees)
					} else if (kernelType.equalsIgnoreCase("comb-norm")) {
						String vectorRepresentationName = "bow";
						String treeRepresentationName = "grct";

						Kernel linearKernel = new LinearKernel(vectorRepresentationName);
						Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
						Kernel treeKernel = new SubSetTreeKernel(lambda, treeRepresentationName);
						Kernel normalizedTreeKernel = new NormalizationKernel(treeKernel);

						LinearKernelCombination combination = new LinearKernelCombination();
						combination.addKernel(1, normalizedLinearKernel);
						combination.addKernel(1, normalizedTreeKernel);
						usedKernel = combination;
					} else {
						System.err.println("The specified kernel (" + kernelType + ") is not valid.");
					}

					// Setting the cache to speed up the computations
					KernelCache cache = new FixIndexKernelCache(cacheSize);
					usedKernel.setKernelCache(cache);

					// Instantiate the SVM learning Algorithm.
					BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
					//Set the kernel
					svmSolver.setKernel(usedKernel);
					//Set the C parameter
					svmSolver.setCp(cp);
					//svmSolver.setCp(c);
					svmSolver.setCn(cn);

					// Instantiate the multi-class classifier that apply a One-vs-All schema
					OneVsAllLearning ovaLearner = new OneVsAllLearning();
					ovaLearner.setBaseAlgorithm(svmSolver);
					ovaLearner.setLabels(trainingSet.getClassificationLabels());

					// Writing the learning algorithm and the kernel to file
					//JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
					//serializer.writeValueOnFile(ovaLearner, "/Users/giando/Google Drive/Courses/Master BIG DATA/Social Media-Basili/Final Project/BigData_2018-2019_TM_SMA_project/resource/ova_learning_algorithm-" + kernelType + "_cp" + cp + "_cn" + cn + ".klp");

					//Learn and get the prediction function
					ovaLearner.learn(trainingSet);
					//Selecting the prediction function
					Classifier classifier = ovaLearner.getPredictionFunction();

					// Write the model (aka the Classifier for further use)
					//serializer.writeValueOnFile(classifier, "/Users/giando/Google Drive/Courses/Master BIG DATA/Social Media-Basili/Final Project/BigData_2018-2019_TM_SMA_project/resource/model_kernel-" + kernelType + "_cp" + cp + "_cn" + cn + ".klp");

					// set Prediction file
					//PrintStream ps_train = new PrintStream(trainOutputFilePath);
					//PrintStream ps_dev = new PrintStream(devOutputFilePath);

					//Building the evaluation function
					MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
							trainingSet.getClassificationLabels());

					// Classify train examples and compute the accuracy
					//for (Example e : trainingSet.getExamples()) {
					// Predict the class
					//	ClassificationOutput p = classifier.predict(e);
					//evaluator.addCount(e, p);
					//System.out.println("Question:\t" + e.getRepresentation("quest"));
					//System.out.println("Original class:\t" + e.getClassificationLabels());
					//System.out.println("Predicted class:\t" + p.getPredictedClasses());
					//System.out.println();

					// write the classification
					//ps_train.println(p.getPredictedClasses().get(0));
					//}

					// Classify dev examples and compute the accuracy
					for (Example e : devSet.getExamples()) {
						// Predict the class
						ClassificationOutput p = classifier.predict(e);
						evaluator.addCount(e, p);

						// write the classification
						//ps_dev.println(p.getPredictedClasses().get(0));
					}

					System.out.println("Accuracy: " + evaluator.getAccuracy());

					results.add(x);
					results.add(y);
					results.add(z);
					results.add(evaluator.getAccuracy());

					count = count + 1;

				}
			}
		}
		PrintStream resultFile = new PrintStream(resultFilePath);
		resultFile.println(results);
	}
}