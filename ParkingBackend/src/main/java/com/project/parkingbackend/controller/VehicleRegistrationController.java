package com.project.parkingbackend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.parkingbackend.model.dto.MapResponseDto;
import com.project.parkingbackend.model.PinSession;
import com.project.parkingbackend.model.dto.PinVerifyRequest;
import com.project.parkingbackend.model.dto.PinVerifyResponse;
import com.project.parkingbackend.model.dto.VehicleRegisterRequest;
import com.project.parkingbackend.model.dto.VehicleRegisterResponse;
import com.project.parkingbackend.service.AiHttpClient;
import com.project.parkingbackend.service.AiMapCacheService;
import com.project.parkingbackend.service.PinSessionStore;
import com.project.parkingbackend.service.VehicleRegistrationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
@RequestMapping("/api/parking")
public class VehicleRegistrationController {

    private static final Logger logger = LoggerFactory.getLogger(VehicleRegistrationController.class);

    private final VehicleRegistrationService registrationService;
    private final AiMapCacheService mapCacheService;
    private final AiHttpClient aiHttpClient;
    private final PinSessionStore pinSessionStore;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final JsonNode defaultMap;

    public VehicleRegistrationController(VehicleRegistrationService registrationService,
                                         AiMapCacheService mapCacheService,
                                         AiHttpClient aiHttpClient,
                                         PinSessionStore pinSessionStore) {
        this.registrationService = registrationService;
        this.mapCacheService = mapCacheService;
        this.aiHttpClient = aiHttpClient;
        this.pinSessionStore = pinSessionStore;
        this.defaultMap = loadDefaultMap();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // POST /api/parking/register
    //
    // Gate entry point.  Send the vehicle registration plate and get back the
    // AI-assigned slot + navigation route.
    // ─────────────────────────────────────────────────────────────────────────
    @PostMapping("/register")
    public ResponseEntity<VehicleRegisterResponse> register(@RequestBody VehicleRegisterRequest request) {
        if (request.getPlateNumber() == null || request.getPlateNumber().isBlank()) {
            VehicleRegisterResponse err = new VehicleRegisterResponse();
            err.setStatus("error");
            err.setMessage("plate_number is required.");
            return ResponseEntity.badRequest().body(err);
        }

        VehicleRegisterResponse response = registrationService.register(request.getPlateNumber());

        HttpStatus httpStatus = switch (response.getStatus()) {
            case "assigned"           -> HttpStatus.OK;
            case "vehicle_not_found"  -> HttpStatus.NOT_FOUND;
            case "no_slots_available" -> HttpStatus.CONFLICT;
            case "ai_not_ready"       -> HttpStatus.SERVICE_UNAVAILABLE;
            default                   -> HttpStatus.INTERNAL_SERVER_ERROR;
        };

        return ResponseEntity.status(httpStatus).body(response);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GET /api/parking/map
    //
    // Returns the cached parking lot map (fetched once from AI on startup).
    // If AI map is not ready yet, returns the configured default map payload.
    // ─────────────────────────────────────────────────────────────────────────
    @GetMapping("/map")
    public ResponseEntity<MapResponseDto> getMap() {
        JsonNode map = mapCacheService.getCachedMap();
        if (map != null) {
            return ResponseEntity.ok(new MapResponseDto(map.deepCopy()));
        }
        // Fallback to the exact frontend contract when AI map is not ready yet.
        return ResponseEntity.ok(new MapResponseDto(defaultMap.deepCopy()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // POST /api/parking/pin/verify
    //
    // Driver's device enters the 4-digit PIN shown after gate registration.
    // Returns the map, assigned slot, route + WebSocket subscription instructions.
    // ─────────────────────────────────────────────────────────────────────────
    @PostMapping("/pin/verify")
    public ResponseEntity<PinVerifyResponse> verifyPin(@RequestBody PinVerifyRequest request) {
        if (request.getPin() == null || request.getPin().isBlank()) {
            PinVerifyResponse err = new PinVerifyResponse();
            err.setError("MISSING_PIN");
            err.setMessage("pin is required.");
            return ResponseEntity.badRequest().body(err);
        }

        Optional<PinSession> opt = pinSessionStore.findByPin(request.getPin().trim());
        if (opt.isEmpty()) {
            PinVerifyResponse err = new PinVerifyResponse();
            err.setError("INVALID_PIN");
            err.setMessage("Invalid or expired PIN. Please request a new one at the gate.");
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(err);
        }

        PinSession ps = opt.get();
        PinVerifyResponse response = new PinVerifyResponse();
        response.setPin(ps.getPin());
        response.setTrackingId(ps.getTrackingId());
        response.setPlateNumber(ps.getPlate());
        response.setExpiresAt(ps.getExpiresAt().toString());
        response.setVehicle(ps.getVehicle());
        response.setAssignedSlot(ps.getAssignedSlot());
        response.setRoute(ps.getRoute());
        response.setMap(mapCacheService.getCachedMap());
        response.setWebsocket(new PinVerifyResponse.WebSocketInfo(
                "ws://localhost:8080/ws/parking",
                "{\"action\":\"subscribe\",\"pin\":\"" + ps.getPin() + "\"}"));

        return ResponseEntity.ok(response);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GET /api/parking/slots
    //
    // Live slot statuses, proxied from the AI in real time.
    // ─────────────────────────────────────────────────────────────────────────
    @GetMapping("/slots")
    public ResponseEntity<JsonNode> getSlots() {
        AiHttpClient.AiHttpResponse response = aiHttpClient.fetchSlotsResponse();
        return ResponseEntity.status(response.statusCode()).body(response.body());
    }

    private JsonNode loadDefaultMap() {
        try {
            JsonNode node = objectMapper.readTree(new ClassPathResource("default-map-response.json").getInputStream());
            if (node != null && node.isObject() && node.has("map")) {
                return node.get("map");
            }
            return objectMapper.nullNode();
        } catch (Exception ex) {
            logger.warn("Could not load default-map-response.json: {}", ex.getMessage());
            return objectMapper.nullNode();
        }
    }
}

