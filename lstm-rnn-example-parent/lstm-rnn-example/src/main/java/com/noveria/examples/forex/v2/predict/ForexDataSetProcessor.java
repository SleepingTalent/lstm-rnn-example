package com.noveria.examples.forex.v2.predict;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.io.ClassPathResource;

import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Date;
import java.util.List;

public class ForexDataSetProcessor {

    private static final Logger log = LoggerFactory.getLogger(ForexDataSetProcessor.class);

    public void processDataSet(String dataSetFileName) throws IOException {
        int linesToSkip = 1;
        String delimiter = ",";

        String dataSetLocation = new ClassPathResource(dataSetFileName).getFile().getAbsolutePath();

        String outputLocation = new ClassPathResource("forexPrediction").getFile().getAbsolutePath();

        String timeStamp = String.valueOf(new Date().getTime());

        String processFileName = "dataSet_processed_"+timeStamp;

        String outputPath = outputLocation+"/"+processFileName;

        /**
         * Data : GBP_USD,1.62800,1.63014,1.62218,1.62305,2012-04-30:01:00:00+0100
         * Fields : instrument,open,high,low,close,date
         */

        Schema inputSchema = new Schema.Builder()
                .addColumnString("instrument")
                .addColumnsDouble("open","high","low","close")
                .addColumnString("date")
                .build();

        /**
         * Define a transform process to extract instrument and close prices
         */

        TransformProcess transformProcess = new TransformProcess.Builder(inputSchema)
                .removeColumns("instrument","open","high","low","date").build();

        int numActions = transformProcess.getActionList().size();

        for(int i = 0; i < numActions; i++) {
            log.info("================================");
            log.info("--- Schema after step ("+transformProcess.getActionList().get(i)+") ---");
            log.info(""+transformProcess.getSchemaAfterStep(i));
        }

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Forex Record Reader Transform");

        try (JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf)) {

            JavaRDD<String> lines = javaSparkContext.textFile(dataSetLocation);

            JavaRDD<List<Writable>> forexReports = lines.map(new StringToWritablesFunction(new CSVRecordReader()));

            JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(forexReports, transformProcess);

            JavaRDD<String> toSave = processed.map(new WritablesToStringFunction(","));

            List<String> processedCollected = toSave.collect();
            List<String> inputDataCollected = lines.collect();

            log.info("---- Original Data ----");
            for (String s : inputDataCollected) System.out.println(s);

            log.info("---- Processed Data ----");
            for (String s : processedCollected) System.out.println(s);

            //log.info("saving as text file to : {}",outputPath);
            //toSave.saveAsTextFile(outputPath);
        }

    }
}
