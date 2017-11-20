package com.noveria.examples.forex.v2.predict;

import com.noveria.examples.forex.v2.model.ForexDataSetIterator;
import com.noveria.examples.forex.v2.model.PriceCategory;
import com.noveria.examples.forex.v2.network.NeuralNetwork;
import com.noveria.examples.forex.v2.util.PlotUtil;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;
import java.util.NoSuchElementException;


public class ForexPricePredictionToo {

    private static final Logger log = LoggerFactory.getLogger(ForexPricePredictionToo.class);

    public static void main(String[] args) throws IOException {
        String file = new ClassPathResource("GBP_USD_testData.csv").getFile().getAbsolutePath();

        String instrument = "GBP_USD"; // stock name
        int batchSize = 128; // mini-batch size
        int exampleLength = 22; // time series length, assume 22 working days per month
        double splitRatio = 0.95; // 90% for training, 10% for testing
        int epochs = 20; // training epochs

        log.info("Create dataSet iterator...");
        ForexDataSetIterator iterator = new ForexDataSetIterator(file, instrument,
                batchSize, exampleLength, splitRatio);

        log.info("Iterator Labels:{}",iterator.getLabels());

        //log.info("Load test dataset...");
        //List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        MultiLayerNetwork net = NeuralNetwork.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

        //net.init();

        log.info("Fit network...");
        net.fit(iterator);

        log.info("Evaluate network...");
        Evaluation evaluation = net.evaluate(iterator);
        System.out.println(evaluation.stats());

//        log.info("Training...");
//        for (int i = 0; i < epochs; i++) {
//
//            DataSet dataSet = null;
//
//            while (iterator.hasNext()) {
//                dataSet = iterator.next();
//                net.fit(dataSet);
//            }
//
//            iterator.reset(); // reset iterator
//            net.rnnClearPreviousState(); // clear previous state
//        }

        //log.info("Testing...");
        //runTestForCloseCategory(exampleLength, iterator, test, net);
    }

    private static void runTestForCloseCategory(int exampleLength, ForexDataSetIterator iterator, List<Pair<INDArray, INDArray>> test, MultiLayerNetwork net) {
        //double max = iterator.getMaxNum()[1];
        //log.info("Max Close Price:{}",max);

        //double min = iterator.getMinNum()[1];
        //log.info("Min Close Price:{}",min);

        double[] predicts = new double[test.size()];
        double[] actuals = new double[test.size()];

        //log.info("Example Length:{}",exampleLength);

        for (int i = 0; i < test.size(); i++) {
            double actualClosePrice = test.get(i).getValue().getDouble(0);
            //log.info("actualClosePrice:{}",actualClosePrice);

            INDArray testPairKey = test.get(i).getKey();
            //log.info("testPairKey:{}",testPairKey);

            double timeStepResult = net.rnnTimeStep(testPairKey).getDouble(exampleLength - 1);
            //log.info("timeStepResult:{}",timeStepResult);

            predicts[i] = timeStepResult; //* (max - min) + min;
            //log.info("timeStepResult maz/min:{}",predicts[i]);

            actuals[i] = actualClosePrice;
        }

        log.info("Print out Predictions and Actual Values...");
        log.info("Predict\t\t\t\t\t\t\tActual");

        for (int i = 0; i < predicts.length; i++) {
            double pipDifference = predicts[i] - actuals[i];

            log.info(predicts[i] + "\t\t" + actuals[i] + "\t\t" + (
                    new DecimalFormat("#0.0000").format(pipDifference))
                    + "\t\t" + (pipDifference / 0.0001));
        }

        //Evaluation evaluation = new Evaluation();
        //INDArray output = net.output(dataSet.getFeatureMatrix());
        //INDArray labels = dataSet.getLabels();
        //evaluation.eval(Nd4j.create(actuals), Nd4j.create(predicts));
        //System.out.println(evaluation.stats());

        //log.info("Plot...");
        //PlotUtil.plot(predicts, actuals, String.valueOf(PriceCategory.CLOSE));
    }
}
