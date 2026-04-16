package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public class AssignedSlotDto {

    @JsonProperty("slot_id")
    private String slotId;

    @JsonProperty("cx")
    private int cx;

    @JsonProperty("cy")
    private int cy;

    @JsonProperty("polygon")
    private List<List<Integer>> polygon;

    @JsonProperty("zone")
    private String zone;

    @JsonProperty("distance_from_exit_m")
    private double distanceFromExitM;

    public String getSlotId() { return slotId; }
    public void setSlotId(String slotId) { this.slotId = slotId; }

    public int getCx() { return cx; }
    public void setCx(int cx) { this.cx = cx; }

    public int getCy() { return cy; }
    public void setCy(int cy) { this.cy = cy; }

    public List<List<Integer>> getPolygon() { return polygon; }
    public void setPolygon(List<List<Integer>> polygon) { this.polygon = polygon; }

    public String getZone() { return zone; }
    public void setZone(String zone) { this.zone = zone; }

    public double getDistanceFromExitM() { return distanceFromExitM; }
    public void setDistanceFromExitM(double distanceFromExitM) { this.distanceFromExitM = distanceFromExitM; }
}

