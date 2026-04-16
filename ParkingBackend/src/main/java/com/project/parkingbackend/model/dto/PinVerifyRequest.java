package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class PinVerifyRequest {

    @JsonProperty("pin")
    private String pin;

    public String getPin() { return pin; }
    public void setPin(String pin) { this.pin = pin; }
}

