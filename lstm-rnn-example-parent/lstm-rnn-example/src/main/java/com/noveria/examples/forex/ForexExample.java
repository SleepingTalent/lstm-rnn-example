package com.noveria.examples.forex;


public class ForexExample {

    public static void main(String[] args) throws Exception {
        setUpForexData();

        loadTrainingData();
        loadTestData();

        configureTheNeuralNet();

        //TODO: train the network and evaluate test set...
    }

    private static void configureTheNeuralNet() {
        //TODO: configure the neural net...
    }

    private static void loadTestData() {
        //TODO: load and normalise test data...
    }

    private static void loadTrainingData() {
        //TODO: load and normalise training data...
    }

    private static void setUpForexData() {
        //TODO: setUp Forex tick data to use...
    }
}
