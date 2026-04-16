package com.project.parkingbackend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.project.parkingbackend.model.dto.ManualGateMatchRequest;
import com.project.parkingbackend.service.AiHttpClient;
import com.project.parkingbackend.service.AiMapCacheService;
import com.project.parkingbackend.service.TrackingSessionStore;
import com.project.parkingbackend.service.VehicleProfileService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class AiParkingController {

    private final AiMapCacheService aiMapCacheService;
    private final AiHttpClient aiHttpClient;
    private final VehicleProfileService vehicleProfileService;
    private final TrackingSessionStore trackingSessionStore;
    private final ObjectMapper objectMapper = new ObjectMapper().findAndRegisterModules();

    public AiParkingController(AiMapCacheService aiMapCacheService,
                               AiHttpClient aiHttpClient,
                               VehicleProfileService vehicleProfileService,
                               TrackingSessionStore trackingSessionStore) {
        this.aiMapCacheService = aiMapCacheService;
        this.aiHttpClient = aiHttpClient;
        this.vehicleProfileService = vehicleProfileService;
        this.trackingSessionStore = trackingSessionStore;
    }

    @GetMapping("/map")
    public ObjectNode getCachedMap() {
        ObjectNode response = objectMapper.createObjectNode();
        JsonNode map = aiMapCacheService.getCachedMap();
        response.set("map", map == null ? objectMapper.nullNode() : map);
        return response;
    }

    @GetMapping("/slots")
    public ResponseEntity<JsonNode> getSlots() {
        AiHttpClient.AiHttpResponse response = aiHttpClient.fetchSlotsResponse();
        return ResponseEntity.status(response.statusCode()).body(response.body());
    }

    @PostMapping("/gate/match")
    public ResponseEntity<JsonNode> manualGateMatch(@RequestBody ManualGateMatchRequest request) {
        String normalizedPlate = normalizePlate(request.getPlateNumber());
        if (normalizedPlate == null) {
            return ResponseEntity.badRequest().body(errorBody("plate_number is required"));
        }

        int trackingId = request.getTrackingId() != null
                ? request.getTrackingId()
                : trackingSessionStore.getOrCreateTrackingId(normalizedPlate);

        trackingSessionStore.bindTrackingId(normalizedPlate, trackingId);

        VehicleProfileService.VehicleDimensions dimensions = vehicleProfileService.getDimensionsByPlate(normalizedPlate);
        AiHttpClient.AiHttpResponse aiResponse = aiHttpClient.gateMatchWithRetry(
                trackingId,
                dimensions.lengthM(),
                dimensions.widthM());

        if (aiResponse.statusCode() == 200) {
            JsonNode slotNode = aiResponse.body().path("assigned_slot").path("slot_id");
            if (!slotNode.isMissingNode() && !slotNode.isNull()) {
                trackingSessionStore.setAssignedSlot(trackingId, slotNode.asText());
            }

            ObjectNode body = aiResponse.body().isObject()
                    ? ((ObjectNode) aiResponse.body()).deepCopy()
                    : objectMapper.createObjectNode();
            if (!aiResponse.body().isObject()) {
                body.set("data", aiResponse.body());
            }
            body.put("plate_number", normalizedPlate);
            body.set("vehicle", vehicleBody(dimensions));
            return ResponseEntity.ok(body);
        }

        if (aiResponse.statusCode() == 409) {
            ObjectNode body = errorBody("no free slots available");
            body.put("plate_number", normalizedPlate);
            body.put("tracking_id", trackingId);
            return ResponseEntity.status(HttpStatus.CONFLICT).body(body);
        }

        if (aiResponse.statusCode() == 503) {
            ObjectNode body = errorBody("AI system not ready");
            body.put("plate_number", normalizedPlate);
            body.put("tracking_id", trackingId);
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(body);
        }

        return ResponseEntity.status(aiResponse.statusCode()).body(aiResponse.body());
    }

    private String normalizePlate(String plateNumber) {
        if (plateNumber == null || plateNumber.isBlank()) {
            return null;
        }
        return plateNumber.trim().toUpperCase();
    }

    private ObjectNode vehicleBody(VehicleProfileService.VehicleDimensions dimensions) {
        ObjectNode vehicle = objectMapper.createObjectNode();
        vehicle.put("length_m", dimensions.lengthM());
        vehicle.put("width_m", dimensions.widthM());
        return vehicle;
    }

    private ObjectNode errorBody(String message) {
        ObjectNode node = objectMapper.createObjectNode();
        node.put("message", message);
        return node;
    }
}





