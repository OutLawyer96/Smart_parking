package com.project.parkingbackend.model;

import com.project.parkingbackend.model.dto.AssignedSlotDto;
import com.project.parkingbackend.model.dto.RouteWaypointDto;

import java.time.Instant;
import java.util.List;

public class PinSession {

    private String pin;
    private int trackingId;
    private String plate;
    private Vehicle vehicle;
    private AssignedSlotDto assignedSlot;
    private List<RouteWaypointDto> route;
    private Instant createdAt;
    private Instant expiresAt;

    public PinSession() {}

    public PinSession(String pin, int trackingId, String plate, Vehicle vehicle,
                      AssignedSlotDto assignedSlot, List<RouteWaypointDto> route,
                      Instant createdAt, Instant expiresAt) {
        this.pin = pin;
        this.trackingId = trackingId;
        this.plate = plate;
        this.vehicle = vehicle;
        this.assignedSlot = assignedSlot;
        this.route = route;
        this.createdAt = createdAt;
        this.expiresAt = expiresAt;
    }

    public String getPin() { return pin; }
    public void setPin(String pin) { this.pin = pin; }

    public int getTrackingId() { return trackingId; }
    public void setTrackingId(int trackingId) { this.trackingId = trackingId; }

    public String getPlate() { return plate; }
    public void setPlate(String plate) { this.plate = plate; }

    public Vehicle getVehicle() { return vehicle; }
    public void setVehicle(Vehicle vehicle) { this.vehicle = vehicle; }

    public AssignedSlotDto getAssignedSlot() { return assignedSlot; }
    public void setAssignedSlot(AssignedSlotDto assignedSlot) { this.assignedSlot = assignedSlot; }

    public List<RouteWaypointDto> getRoute() { return route; }
    public void setRoute(List<RouteWaypointDto> route) { this.route = route; }

    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }

    public Instant getExpiresAt() { return expiresAt; }
    public void setExpiresAt(Instant expiresAt) { this.expiresAt = expiresAt; }
}

