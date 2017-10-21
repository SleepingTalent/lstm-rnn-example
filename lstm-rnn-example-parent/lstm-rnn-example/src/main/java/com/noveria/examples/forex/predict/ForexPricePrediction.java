package com.noveria.examples.forex.predict;

import com.noveria.examples.forex.representation.PriceCategory;
import com.noveria.examples.forex.representation.ForexDataSetIterator;
import com.noveria.examples.forex.utils.PlotUtil;
import com.noveria.examples.stock.model.RecurrentNets;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;


public class ForexPricePrediction {

    private static final Logger log = LoggerFactory.getLogger(ForexPricePrediction.class);

    public static void main (String[] args) throws IOException {
        String file = new ClassPathResource("GBP_USD_testData.csv").getFile().getAbsolutePath();

        String instrument = "GBP_USD"; // stock name
        int batchSize = 64; // mini-batch size
        int exampleLength = 22; // time series length, assume 22 working days per month
        double splitRatio = 0.9; // 90% for training, 10% for testing
        int epochs = 100; // training epochs

        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
        //PriceCategory category = PriceCategory.ALL; // ALL: predict all features
        ForexDataSetIterator iterator = new ForexDataSetIterator(file, instrument, batchSize, exampleLength, splitRatio, category);

        log.info("Load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

        log.info("Training...");
        for (int i = 0; i < epochs; i++) {

            DataSet dataSet;

            while (iterator.hasNext()) {
                dataSet = iterator.next();
                net.fit(dataSet);
            }

            iterator.reset(); // reset iterator
            net.rnnClearPreviousState(); // clear previous state
        }

        log.info("Saving model...");
        String outputLocation = new ClassPathResource("forexPrediction").getFile().getAbsolutePath();
        File locationToSave = new File(outputLocation+"/ForexPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));

        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc.
        // Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        log.info("load model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("Testing...");
        if (category.equals(PriceCategory.ALL)) {

            INDArray max = Nd4j.create(iterator.getMaxNum());
            INDArray min = Nd4j.create(iterator.getMinNum());
            INDArray[] predicts = new INDArray[test.size()];
            INDArray[] actuals = new INDArray[test.size()];

            for (int i = 0; i < test.size(); i++) {

                predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
                actuals[i] = test.get(i).getValue();

            }

            log.info("Print out Predictions and Actual Values...");
            log.info("Predict\t\t\t\t\t\t\t\t\t\t\t\t\tActual");

            for (int i = 0; i < predicts.length; i++) {
                log.info(predicts[i] + "\t\t" + actuals[i]);
            }

            log.info("Plot...");
            plotAll(predicts, actuals);
        } else {

            double max = iterator.getMaxNum()[1];
            double min = iterator.getMinNum()[1];
            double[] predicts = new double[test.size()];
            double[] actuals = new double[test.size()];

            for (int i = 0; i < test.size(); i++) {
                predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
                actuals[i] = test.get(i).getValue().getDouble(0);
            }

            log.info("Print out Predictions and Actual Values...");
            log.info("Predict\t\t\t\t\t\t\tActual");

            for (int i = 0; i < predicts.length; i++){
                log.info(predicts[i] + "\t\t" + actuals[i]);
            }

            //log.info("Plot...");
            //PlotUtil.plot(predicts, actuals, String.valueOf(category));
        }
        log.info("Done...");
    }

    // plot all predictions
    private static void plotAll(INDArray[] predicts, INDArray[] actuals) {

        for (int n = 0; n < 5; n++) {

            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];

            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }

            String name;

            switch (n) {
                case 0: name = "Stock OPEN Price"; break;
                case 1: name = "Stock CLOSE Price"; break;
                case 2: name = "Stock LOW Price"; break;
                case 3: name = "Stock HIGH Price"; break;
                case 4: name = "Stock VOLUME Amount"; break;
                default: throw new NoSuchElementException();
            }

            PlotUtil.plot(pred, actu, name);
        }
    }
}
