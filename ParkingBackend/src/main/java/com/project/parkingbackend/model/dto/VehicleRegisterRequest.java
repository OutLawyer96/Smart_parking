package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class VehicleRegisterRequest {

    @JsonProperty("plate_number")
    private String plateNumber;

    public String getPlateNumber() { return plateNumber; }
    public void setPlateNumber(String plateNumber) { this.plateNumber = plateNumber; }
}

