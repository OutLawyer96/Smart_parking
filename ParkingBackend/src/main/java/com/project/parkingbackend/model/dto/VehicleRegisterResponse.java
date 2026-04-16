package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.project.parkingbackend.model.Vehicle;

import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class VehicleRegisterResponse {

    /**
     * status values:
     *   "assigned"           – slot successfully assigned
     *   "no_slots_available" – parking lot is full (AI returned 409)
     *   "ai_not_ready"       – AI system is still warming up (AI returned 503)
     *   "vehicle_not_found"  – plate number not in the database
     *   "error"              – unexpected error
     */
    @JsonProperty("plate_number")
    private String plateNumber;

    @JsonProperty("tracking_id")
    private Integer trackingId;

    @JsonProperty("status")
    private String status;

    @JsonProperty("message")
    private String message;

    @JsonProperty("vehicle")
    private Vehicle vehicle;

    @JsonProperty("assigned_slot")
    private AssignedSlotDto assignedSlot;

    @JsonProperty("route")
    private List<RouteWaypointDto> route;

    /**
     * 4-digit PIN issued on successful slot assignment.
     * The driver/display device enters this PIN to load the full parking map + real-time updates.
     * Null when status != "assigned".
     */
    @JsonProperty("pin")
    private String pin;

    /** How many minutes the PIN remains valid (always 30 on success). */
    @JsonProperty("pin_expires_in_minutes")
    private Integer pinExpiresInMinutes;

    // ── getters / setters ────────────────────────────────────────────────────

    public String getPlateNumber() { return plateNumber; }
    public void setPlateNumber(String plateNumber) { this.plateNumber = plateNumber; }

    public Integer getTrackingId() { return trackingId; }
    public void setTrackingId(Integer trackingId) { this.trackingId = trackingId; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public Vehicle getVehicle() { return vehicle; }
    public void setVehicle(Vehicle vehicle) { this.vehicle = vehicle; }

    public AssignedSlotDto getAssignedSlot() { return assignedSlot; }
    public void setAssignedSlot(AssignedSlotDto assignedSlot) { this.assignedSlot = assignedSlot; }

    public List<RouteWaypointDto> getRoute() { return route; }
    public void setRoute(List<RouteWaypointDto> route) { this.route = route; }

    public String getPin() { return pin; }
    public void setPin(String pin) { this.pin = pin; }

    public Integer getPinExpiresInMinutes() { return pinExpiresInMinutes; }
    public void setPinExpiresInMinutes(Integer pinExpiresInMinutes) { this.pinExpiresInMinutes = pinExpiresInMinutes; }
}

