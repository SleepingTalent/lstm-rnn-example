package com.noveria.examples.forex.v2.iterator;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by jaybo_000 on 01/11/2017.
 */
public class DataSetPreProcessorImpl implements DataSetPreProcessor {

    private static final Logger log = LoggerFactory.getLogger(DataSetPreProcessorImpl.class);

    @Override
    public void preProcess(DataSet dataSet) {
        log.info("preProcessing dataSet..");
    }
}
