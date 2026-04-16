package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class RouteWaypointDto {

    @JsonProperty("x")
    private int x;

    @JsonProperty("y")
    private int y;

    @JsonProperty("maneuver")
    private String maneuver;

    public int getX() { return x; }
    public void setX(int x) { this.x = x; }

    public int getY() { return y; }
    public void setY(int y) { this.y = y; }

    public String getManeuver() { return maneuver; }
    public void setManeuver(String maneuver) { this.maneuver = maneuver; }
}

