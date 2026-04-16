package com.project.parkingbackend.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.parkingbackend.model.Vehicle;
import com.project.parkingbackend.model.dto.AssignedSlotDto;
import com.project.parkingbackend.model.dto.RouteWaypointDto;
import com.project.parkingbackend.model.dto.VehicleRegisterResponse;
import com.project.parkingbackend.repository.VehicleRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class VehicleRegistrationService {

    private static final Logger logger = LoggerFactory.getLogger(VehicleRegistrationService.class);

    private final VehicleRepository vehicleRepository;
    private final AiHttpClient aiHttpClient;
    private final TrackingSessionStore trackingSessionStore;
    private final PinSessionStore pinSessionStore;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public VehicleRegistrationService(VehicleRepository vehicleRepository,
                                      AiHttpClient aiHttpClient,
                                      TrackingSessionStore trackingSessionStore,
                                      PinSessionStore pinSessionStore) {
        this.vehicleRepository = vehicleRepository;
        this.aiHttpClient = aiHttpClient;
        this.trackingSessionStore = trackingSessionStore;
        this.pinSessionStore = pinSessionStore;
    }

    /**
     * Full gate-entry flow:
     *  1. Normalise plate
     *  2. Look up vehicle dimensions from DB
     *  3. Get / create a tracking ID for this plate
     *  4. Call ML POST /api/v1/gate/match
     *  5. Return typed response to controller
     */
    public VehicleRegisterResponse register(String rawPlate) {
        String plate = rawPlate.trim().toUpperCase();

        // ── 1. Fetch vehicle from DB ──────────────────────────────────────────
        Vehicle vehicle = vehicleRepository.findByPlateNumber(plate).orElse(null);
        if (vehicle == null) {
            logger.warn("Vehicle not found in DB: {}", plate);
            VehicleRegisterResponse resp = new VehicleRegisterResponse();
            resp.setPlateNumber(plate);
            resp.setStatus("vehicle_not_found");
            resp.setMessage("Vehicle with plate number '" + plate + "' is not registered in the system. " +
                    "Please add it via POST /api/parking/vehicles first.");
            return resp;
        }

        // ── 2. Generate tracking ID ───────────────────────────────────────────
        int trackingId = trackingSessionStore.getOrCreateTrackingId(plate);
        logger.info("Registering vehicle {} (tracking_id={}) | {}m x {}m",
                plate, trackingId, vehicle.getLengthM(), vehicle.getWidthM());

        // ── 3. Call ML gate/match (with 503 retry) ────────────────────────────
        AiHttpClient.AiHttpResponse aiResp = aiHttpClient.gateMatchWithRetry(
                trackingId, vehicle.getLengthM(), vehicle.getWidthM());

        // ── 4. Build response ─────────────────────────────────────────────────
        VehicleRegisterResponse response = new VehicleRegisterResponse();
        response.setPlateNumber(plate);
        response.setTrackingId(trackingId);
        response.setVehicle(vehicle);

        switch (aiResp.statusCode()) {
            case 200 -> {
                JsonNode body = aiResp.body();
                AssignedSlotDto slot = parseSlot(body.path("assigned_slot"));
                List<RouteWaypointDto> route = parseRoute(body.path("route"));

                response.setStatus("assigned");
                response.setAssignedSlot(slot);
                response.setRoute(route);

                if (slot != null && slot.getSlotId() != null) {
                    trackingSessionStore.setAssignedSlot(trackingId, slot.getSlotId());
                    logger.info("Slot {} assigned to {} (tracking_id={})", slot.getSlotId(), plate, trackingId);
                }

                // Generate 4-digit PIN for the driver's display device
                String pin = pinSessionStore.createPin(trackingId, plate, vehicle, slot, route);
                response.setPin(pin);
                response.setPinExpiresInMinutes(pinSessionStore.getPinTtlMinutes());
                logger.info("PIN {} issued for plate {} (tracking_id={})", pin, plate, trackingId);
            }
            case 409 -> {
                response.setStatus("no_slots_available");
                response.setMessage("No free parking slots available at this time.");
                logger.warn("No slots available for {} (tracking_id={})", plate, trackingId);
            }
            case 503 -> {
                response.setStatus("ai_not_ready");
                response.setMessage("Parking AI system is still initializing. Please try again shortly.");
                logger.warn("AI not ready for {} (tracking_id={})", plate, trackingId);
            }
            default -> {
                response.setStatus("error");
                response.setMessage("Unexpected error from parking AI. HTTP status: " + aiResp.statusCode());
                logger.error("Unexpected AI status {} for {}", aiResp.statusCode(), plate);
            }
        }

        return response;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private AssignedSlotDto parseSlot(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return null;
        try {
            AssignedSlotDto dto = new AssignedSlotDto();
            dto.setSlotId(node.path("slot_id").asText(null));
            dto.setCx(node.path("cx").asInt());
            dto.setCy(node.path("cy").asInt());
            dto.setZone(node.path("zone").asText(null));
            dto.setDistanceFromExitM(node.path("distance_from_exit_m").asDouble());

            JsonNode polygonNode = node.path("polygon");
            if (!polygonNode.isMissingNode() && !polygonNode.isNull()) {
                List<List<Integer>> polygon = objectMapper.convertValue(
                        polygonNode, new TypeReference<>() {});
                dto.setPolygon(polygon);
            }
            return dto;
        } catch (Exception ex) {
            logger.warn("Failed to parse assigned_slot from AI response: {}", ex.getMessage());
            return null;
        }
    }

    private List<RouteWaypointDto> parseRoute(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return null;
        try {
            return objectMapper.convertValue(node, new TypeReference<>() {});
        } catch (Exception ex) {
            logger.warn("Failed to parse route from AI response: {}", ex.getMessage());
            return null;
        }
    }
}

