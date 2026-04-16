package com.project.parkingbackend.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class TargetSlot {
    @JsonProperty("x")
    private double x;

    @JsonProperty("y")
    private double y;

    @JsonProperty("width")
    private double width;

    @JsonProperty("length")
    private double length;

    public TargetSlot() {
    }

    public TargetSlot(double x, double y, double width, double length) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.length = length;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getWidth() {
        return width;
    }

    public void setWidth(double width) {
        this.width = width;
    }

    public double getLength() {
        return length;
    }

    public void setLength(double length) {
        this.length = length;
    }

    @Override
    public String toString() {
        return "TargetSlot{" +
                "x=" + x +
                ", y=" + y +
                ", width=" + width +
                ", length=" + length +
                '}';
    }
}

