package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

public class MapResponseDto {

    @JsonProperty("map")
    private JsonNode map;

    public MapResponseDto() {
    }

    public MapResponseDto(JsonNode map) {
        this.map = map;
    }

    public JsonNode getMap() {
        return map;
    }

    public void setMap(JsonNode map) {
        this.map = map;
    }
}

