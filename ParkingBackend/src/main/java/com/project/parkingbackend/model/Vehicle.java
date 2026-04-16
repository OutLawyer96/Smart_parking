package com.project.parkingbackend.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Vehicle {

    @JsonProperty("plate_number")
    private String plateNumber;

    @JsonProperty("make")
    private String make;

    @JsonProperty("model")
    private String model;

    @JsonProperty("length_m")
    private double lengthM;

    @JsonProperty("width_m")
    private double widthM;

    public Vehicle() {}

    public Vehicle(String plateNumber, String make, String model, double lengthM, double widthM) {
        this.plateNumber = plateNumber == null ? null : plateNumber.toUpperCase();
        this.make = make;
        this.model = model;
        this.lengthM = lengthM;
        this.widthM = widthM;
    }

    public String getPlateNumber() { return plateNumber; }
    public void setPlateNumber(String plateNumber) {
        this.plateNumber = plateNumber == null ? null : plateNumber.toUpperCase();
    }

    public String getMake() { return make; }
    public void setMake(String make) { this.make = make; }

    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }

    public double getLengthM() { return lengthM; }
    public void setLengthM(double lengthM) { this.lengthM = lengthM; }

    public double getWidthM() { return widthM; }
    public void setWidthM(double widthM) { this.widthM = widthM; }

    @Override
    public String toString() {
        return "Vehicle{plate='" + plateNumber + "', make='" + make + "', model='" + model +
               "', length=" + lengthM + ", width=" + widthM + "}";
    }
}

