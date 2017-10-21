package com.noveria.examples.forex.utils;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;
import org.datavec.api.util.ClassPathResource;

import java.io.IOException;

public class DataPreview {
    public static void main (String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().master("local").appName("DataProcess").getOrCreate();
        //String filename = "prices-split-adjusted.csv";
        String filename = "GBP_USD_testData.csv";

        String symbol = "GOOG";
        // load data from csv file

        //date,symbol,open,close,low,high,volume
        //instrument,open,high,low,close,date

        //System.out.println(new ClassPathResource(filename).getFile().getAbsolutePath());
        //2012-04-30:01:00:00+0100


        Dataset<Row> data = spark.read().format("csv").option("header", true)
                //.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ")
                .option("timestampFormat", "yyyy-MM-dd:HH:mm:ss ZZ")
                .load(new ClassPathResource(filename).getFile().getAbsolutePath())
                //.filter(functions.col("symbol").equalTo(symbol))
                //.drop("date").drop("symbol")
                .withColumn("instrument", functions.col("instrument").cast("string"))
                .withColumn("openPrice", functions.col("open").cast("double")).drop("open")
                .withColumn("highPrice", functions.col("high").cast("double")).drop("high")
                .withColumn("lowPrice", functions.col("low").cast("double")).drop("low")
                .withColumn("closePrice", functions.col("close").cast("double")).drop("close")
                .toDF("instrument", "date", "open", "high", "low", "close");

        data.show();

        Dataset<Row> instruments = data.select("date", "instrument").groupBy("instrument").agg(functions.count("date").as("count"));
        System.out.println("Number of Symbols: " + instruments.count());
        instruments.show();

//        VectorAssembler assembler = new VectorAssembler()
//                .setInputCols(new String[] {"open", "low", "high", "volume", "close"})
//                .setOutputCol("features");
//
//        data = assembler.transform(data).drop("open", "low", "high", "volume", "close");
//
//        data = new MinMaxScaler().setMin(0).setMax(1)
//                .setInputCol("features").setOutputCol("normalizedFeatures")
//                .fit(data).transform(data)
//                .drop("features").toDF("features");
    }
}
