package com.noveria.examples.forex.v2.model;

public class ForexData {
    private String date; // date
    private String instrument; // instrument name

    private double open; // open price
    private double close; // close price
    private double low; // low price
    private double high; // high price

    public ForexData() {}

    public ForexData(String date, String instrument, double open, double close, double low, double high) {
        this.date = date;
        this.instrument = instrument;
        this.open = open;
        this.close = close;
        this.low = low;
        this.high = high;
    }

    public String getDate() { return date; }
    public void setDate(String date) { this.date = date; }

    public String getInstrument() { return instrument; }
    public void setInstrument(String instrument) { this.instrument = instrument; }

    public double getOpen() { return open; }
    public void setOpen(double open) { this.open = open; }

    public double getClose() { return close; }
    public void setClose(double close) { this.close = close; }

    public double getLow() { return low; }
    public void setLow(double low) { this.low = low; }

    public double getHigh() { return high; }
    public void setHigh(double high) { this.high = high; }

    @Override
    public String toString() {
        return "ForexData{" +
                "date='" + date + '\'' +
                ", instrument='" + instrument + '\'' +
                ", open=" + open +
                ", close=" + close +
                ", low=" + low +
                ", high=" + high +
                '}';
    }
}
