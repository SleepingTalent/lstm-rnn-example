package com.noveria.examples.forex.v2.util;

import com.noveria.examples.forex.v2.model.ForexData;
import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jaybo_000 on 01/11/2017.
 */
public class CSVUtil {

    public static List<ForexData> readStockDataFromFile(String filename, String symbol) {
        List<ForexData> forexDataList = new ArrayList<>();

        try {
            List<String[]> csvData = new CSVReader(new FileReader(filename)).readAll();

            for (String[] csvDataRow : csvData) {

                if (!csvDataRow[0].equals(symbol)) {
                    continue;
                }

                String instrument = csvDataRow[0];
                double open = Double.valueOf(csvDataRow[1]);
                double high = Double.valueOf(csvDataRow[2]);
                double low = Double.valueOf(csvDataRow[3]);
                double close = Double.valueOf(csvDataRow[4]);
                String date = csvDataRow[5];

                ForexData forexData = new ForexData(date, instrument, open, close, low, high);
                //System.out.println(forexData.toString());

                forexDataList.add(forexData);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return forexDataList;
    }
}
