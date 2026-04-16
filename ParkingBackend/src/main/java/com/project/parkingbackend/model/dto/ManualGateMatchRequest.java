package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ManualGateMatchRequest {

    @JsonProperty("plate_number")
    private String plateNumber;

    @JsonProperty("tracking_id")
    private Integer trackingId;

    public String getPlateNumber() {
        return plateNumber;
    }

    public void setPlateNumber(String plateNumber) {
        this.plateNumber = plateNumber;
    }

    public Integer getTrackingId() {
        return trackingId;
    }

    public void setTrackingId(Integer trackingId) {
        this.trackingId = trackingId;
    }
}

