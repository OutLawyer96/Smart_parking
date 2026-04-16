package com.project.parkingbackend.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class ParkingResponse {
    @JsonProperty("path")
    private List<Coordinate> path;

    @JsonProperty("obstacles")
    private List<Obstacle> obstacles;

    @JsonProperty("targetSlot")
    private TargetSlot targetSlot;

    public ParkingResponse() {
    }

    public ParkingResponse(List<Coordinate> path, List<Obstacle> obstacles, TargetSlot targetSlot) {
        this.path = path;
        this.obstacles = obstacles;
        this.targetSlot = targetSlot;
    }

    public List<Coordinate> getPath() {
        return path;
    }

    public void setPath(List<Coordinate> path) {
        this.path = path;
    }

    public List<Obstacle> getObstacles() {
        return obstacles;
    }

    public void setObstacles(List<Obstacle> obstacles) {
        this.obstacles = obstacles;
    }

    public TargetSlot getTargetSlot() {
        return targetSlot;
    }

    public void setTargetSlot(TargetSlot targetSlot) {
        this.targetSlot = targetSlot;
    }

    @Override
    public String toString() {
        return "ParkingResponse{" +
                "path=" + path +
                ", obstacles=" + obstacles +
                ", targetSlot=" + targetSlot +
                '}';
    }
}

