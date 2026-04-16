package com.project.parkingbackend.model.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.project.parkingbackend.model.Vehicle;

import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class PinVerifyResponse {

    @JsonProperty("pin")
    private String pin;

    @JsonProperty("tracking_id")
    private Integer trackingId;

    @JsonProperty("plate_number")
    private String plateNumber;

    /** ISO-8601 timestamp when this PIN expires */
    @JsonProperty("expires_at")
    private String expiresAt;

    @JsonProperty("vehicle")
    private Vehicle vehicle;

    @JsonProperty("assigned_slot")
    private AssignedSlotDto assignedSlot;

    @JsonProperty("route")
    private List<RouteWaypointDto> route;

    /** Full parking lot map geometry (same shape as GET /api/parking/map → map) */
    @JsonProperty("map")
    private JsonNode map;

    /** WebSocket connection instructions */
    @JsonProperty("websocket")
    private WebSocketInfo websocket;

    /** Set only on error responses */
    @JsonProperty("error")
    private String error;

    @JsonProperty("message")
    private String message;

    // ── Nested WS info ────────────────────────────────────────────────────────

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class WebSocketInfo {
        @JsonProperty("url")
        private final String url;

        /** Send this exact JSON string over the WebSocket to start receiving updates */
        @JsonProperty("subscribe_message")
        private final String subscribeMessage;

        public WebSocketInfo(String url, String subscribeMessage) {
            this.url = url;
            this.subscribeMessage = subscribeMessage;
        }

        public String getUrl() { return url; }
        public String getSubscribeMessage() { return subscribeMessage; }
    }

    // ── Getters / setters ─────────────────────────────────────────────────────

    public String getPin() { return pin; }
    public void setPin(String pin) { this.pin = pin; }

    public Integer getTrackingId() { return trackingId; }
    public void setTrackingId(Integer trackingId) { this.trackingId = trackingId; }

    public String getPlateNumber() { return plateNumber; }
    public void setPlateNumber(String plateNumber) { this.plateNumber = plateNumber; }

    public String getExpiresAt() { return expiresAt; }
    public void setExpiresAt(String expiresAt) { this.expiresAt = expiresAt; }

    public Vehicle getVehicle() { return vehicle; }
    public void setVehicle(Vehicle vehicle) { this.vehicle = vehicle; }

    public AssignedSlotDto getAssignedSlot() { return assignedSlot; }
    public void setAssignedSlot(AssignedSlotDto assignedSlot) { this.assignedSlot = assignedSlot; }

    public List<RouteWaypointDto> getRoute() { return route; }
    public void setRoute(List<RouteWaypointDto> route) { this.route = route; }

    public JsonNode getMap() { return map; }
    public void setMap(JsonNode map) { this.map = map; }

    public WebSocketInfo getWebsocket() { return websocket; }
    public void setWebsocket(WebSocketInfo websocket) { this.websocket = websocket; }

    public String getError() { return error; }
    public void setError(String error) { this.error = error; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
}

