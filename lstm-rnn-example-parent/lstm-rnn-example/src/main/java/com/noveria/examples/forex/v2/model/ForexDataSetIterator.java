package com.noveria.examples.forex.v2.model;

import com.google.common.collect.Lists;
import com.noveria.examples.forex.v2.util.CSVUtil;
import com.opencsv.CSVReader;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;


public class ForexDataSetIterator implements DataSetIterator {

    //private final int VECTOR_SIZE = 5;
    private int miniBatchSize;
    private int exampleLength;

    //private double[] minNum = new double[VECTOR_SIZE];
    //private double[] maxNum = new double[VECTOR_SIZE];

    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    private List<ForexData> train;
    private List<Pair<INDArray, INDArray>> test;

    public ForexDataSetIterator(String filename, String instrument, int miniBatchSize,
                                int exampleLength, double splitRatio) {

        List<ForexData> forexDataList = CSVUtil.readStockDataFromFile(filename, instrument);

        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;

        int split = (int) Math.round(forexDataList.size() * splitRatio);

        train = forexDataList.subList(0, split);
        test = generateTestDataSet(forexDataList.subList(split, forexDataList.size()));

        initializeOffsets();
    }

    private void initializeOffsets() {
        exampleStartOffsets.clear();
        int window = exampleLength + 1;
        for (int i = 0; i < train.size() - window; i++) {
            exampleStartOffsets.add(i);
        }
    }

    public List<Pair<INDArray, INDArray>> getTestDataSet() {
        return test;
    }

//    public double[] getMaxNum() {
//        return maxNum;
//    }
//
//    public double[] getMinNum() {
//        return minNum;
//    }

    @Override
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();

        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[]{actualMiniBatchSize, 1, exampleLength}, 'f');
        INDArray label = Nd4j.create(new int[]{actualMiniBatchSize, 1, exampleLength}, 'f');

        for (int index = 0; index < actualMiniBatchSize; index++) {

            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;

            ForexData curData = train.get(startIdx);
            ForexData nextData;

            for (int i = startIdx; i < endIdx; i++) {

                nextData = train.get(i + 1);
                int c = i - startIdx;

                input.putScalar(new int[]{index, 0, c}, (curData.getClose()));
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    private double feedLabel(ForexData data) {
        return (data.getClose());
    }

    @Override
    public int totalExamples() {
        return train.size() - exampleLength - 1;
    }

    @Override
    public int inputColumns() {
        return 1;
    }

    @Override
    public int totalOutcomes() {
        return 1;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        initializeOffsets();
    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    @Override
    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public List<String> getLabels() {
        List<String> labels = Lists.newArrayList();
        labels.add("GBP_USD");
        return labels;
    }

    @Override
    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    private List<Pair<INDArray, INDArray>> generateTestDataSet(List<ForexData> forexDataList) {

        int window = exampleLength + 1;

        List<Pair<INDArray, INDArray>> test = new ArrayList<>();

        for (int i = 0; i < forexDataList.size() - window; i++) {

            INDArray input = Nd4j.create(new int[]{exampleLength, 1}, 'f');

            for (int j = i; j < i + exampleLength; j++) {

                ForexData forexData = forexDataList.get(j);
                input.putScalar(new int[]{j - i, 0}, (forexData.getClose()));
            }

            ForexData forexData = forexDataList.get(i + exampleLength);

            INDArray label = Nd4j.create(new int[]{1}, 'f');
            label.putScalar(new int[]{0}, forexData.getClose());

            test.add(Pair.of(input, label));
        }

        return test;
    }

}
