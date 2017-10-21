package com.noveria.examples.forex.representation;

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

    private final int VECTOR_SIZE = 5;
    private int miniBatchSize;
    private int exampleLength;

    private double[] minNum = new double[VECTOR_SIZE];
    private double[] maxNum = new double[VECTOR_SIZE];

    private PriceCategory category;

    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    private List<ForexData> train;
    private List<Pair<INDArray, INDArray>> test;

    public ForexDataSetIterator(String filename, String instrument, int miniBatchSize, int exampleLength, double splitRatio,
                                PriceCategory category) {
        List<ForexData> forexDataList = readStockDataFromFile(filename, instrument);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(forexDataList.size() * splitRatio);
        train = forexDataList.subList(0, split);
        test = generateTestDataSet(forexDataList.subList(split, forexDataList.size()));
        initializeOffsets();
    }

    private void initializeOffsets () {
        exampleStartOffsets.clear();
        int window = exampleLength + 1;
        for (int i = 0; i < train.size() - window; i++) {
            exampleStartOffsets.add(i);
        }
    }

    public List<Pair<INDArray, INDArray>> getTestDataSet() { return test; }

    public double[] getMaxNum() { return maxNum; }

    public double[] getMinNum() { return minNum; }

    @Override
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label;
        if (category.equals(PriceCategory.ALL))
            label = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        else
            label = Nd4j.create(new int[] {actualMiniBatchSize, 1, exampleLength}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            ForexData curData = train.get(startIdx);
            ForexData nextData;
            for (int i = startIdx; i < endIdx; i++) {
                nextData = train.get(i + 1);
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, (curData.getOpen() - minNum[0]) / (maxNum[0] - minNum[0]));
                input.putScalar(new int[] {index, 1, c}, (curData.getClose() - minNum[1]) / (maxNum[1] - minNum[1]));
                input.putScalar(new int[] {index, 2, c}, (curData.getLow() - minNum[2]) / (maxNum[2] - minNum[2]));
                input.putScalar(new int[] {index, 3, c}, (curData.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]));
                if (category.equals(PriceCategory.ALL)) {
                    label.putScalar(new int[] {index, 0, c}, (nextData.getOpen() - minNum[1]) / (maxNum[1] - minNum[1]));
                    label.putScalar(new int[] {index, 1, c}, (nextData.getClose() - minNum[1]) / (maxNum[1] - minNum[1]));
                    label.putScalar(new int[] {index, 2, c}, (nextData.getLow() - minNum[2]) / (maxNum[2] - minNum[2]));
                    label.putScalar(new int[] {index, 3, c}, (nextData.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]));
                } else {
                    label.putScalar(new int[]{index, 0, c}, feedLabel(nextData));
                }
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    private double feedLabel(ForexData data) {
        double value;
        switch (category) {
            case OPEN: value = (data.getOpen() - minNum[0]) / (maxNum[0] - minNum[0]); break;
            case CLOSE: value = (data.getClose() - minNum[1]) / (maxNum[1] - minNum[1]); break;
            case LOW: value = (data.getLow() - minNum[2]) / (maxNum[2] - minNum[2]); break;
            case HIGH: value = (data.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]); break;
            default: throw new NoSuchElementException();
        }
        return value;
    }

    @Override
    public int totalExamples() { return train.size() - exampleLength - 1; }

    @Override
    public int inputColumns() { return VECTOR_SIZE; }

    @Override
    public int totalOutcomes() {
        if (this.category.equals(PriceCategory.ALL)) return VECTOR_SIZE;
        else return 1;
    }

    @Override
    public boolean resetSupported() { return false; }

    @Override
    public boolean asyncSupported() { return false; }

    @Override
    public void reset() { initializeOffsets(); }

    @Override
    public int batch() { return miniBatchSize; }

    @Override
    public int cursor() { return totalExamples() - exampleStartOffsets.size(); }

    @Override
    public int numExamples() { return totalExamples(); }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override
    public List<String> getLabels() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override
    public boolean hasNext() { return exampleStartOffsets.size() > 0; }

    @Override
    public DataSet next() { return next(miniBatchSize); }
    
    private List<Pair<INDArray, INDArray>> generateTestDataSet (List<ForexData> forexDataList) {

        int window = exampleLength + 1;

    	List<Pair<INDArray, INDArray>> test = new ArrayList<>();

    	for (int i = 0; i < forexDataList.size() - window; i++) {

    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');

    		for (int j = i; j < i + exampleLength; j++) {

                ForexData forexData = forexDataList.get(j);

    			input.putScalar(new int[] {j - i, 0}, (forexData.getOpen() - minNum[0]) / (maxNum[0] - minNum[0]));
    			input.putScalar(new int[] {j - i, 1}, (forexData.getClose() - minNum[1]) / (maxNum[1] - minNum[1]));
    			input.putScalar(new int[] {j - i, 2}, (forexData.getLow() - minNum[2]) / (maxNum[2] - minNum[2]));
    			input.putScalar(new int[] {j - i, 3}, (forexData.getHigh() - minNum[3]) / (maxNum[3] - minNum[3]));
    		}

            ForexData stock = forexDataList.get(i + exampleLength);
            INDArray label;

            if (category.equals(PriceCategory.ALL)) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f');
                label.putScalar(new int[] {0}, stock.getOpen());
                label.putScalar(new int[] {1}, stock.getClose());
                label.putScalar(new int[] {2}, stock.getLow());
                label.putScalar(new int[] {3}, stock.getHigh());
            } else {
                label = Nd4j.create(new int[] {1}, 'f');
                switch (category) {
                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
                    default: throw new NoSuchElementException();
                }
            }
    		test.add(Pair.of(input, label));
    	}
    	return test;
    }

	private List<ForexData> readStockDataFromFile (String filename, String symbol) {
        List<ForexData> forexDataList = new ArrayList<>();

        try {
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll();
            for (int i = 0; i < maxNum.length; i++) {
                maxNum[i] = Double.MIN_VALUE;
                minNum[i] = Double.MAX_VALUE;
            }
            for (String[] arr : list) {

                if (!arr[0].equals(symbol)){
                    continue;
                }

                double[] nums = new double[VECTOR_SIZE];

                for (int i = 0; i < arr.length - 2; i++) {

                    nums[i] = Double.valueOf(arr[i + 1]);
                    if (nums[i] > maxNum[i]){
                        maxNum[i] = nums[i];
                    }
                    if (nums[i] < minNum[i]){
                        minNum[i] = nums[i];
                    }
                }

                forexDataList.add(new ForexData(arr[0], arr[4], nums[0], nums[1], nums[2], nums[3]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return forexDataList;
    }
}
