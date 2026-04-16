package com.project.parkingbackend.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.project.parkingbackend.service.AiMapCacheService;
import com.project.parkingbackend.websocket.ParkingWebSocketHandler;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/parking")
public class ParkingController {

    private final AiMapCacheService aiMapCacheService;

    public ParkingController(AiMapCacheService aiMapCacheService) {
        this.aiMapCacheService = aiMapCacheService;
    }

    /**
     * Get status of WebSocket connection and streaming
     */
    @GetMapping("/status")
    public Map<String, Object> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("connectedClients", ParkingWebSocketHandler.getConnectedClientsCount());
        status.put("isStreaming", ParkingWebSocketHandler.isStreamingActive());
        status.put("mapReady", aiMapCacheService.isMapReady());
        status.put("websocketUrl", "ws://localhost:8080/ws/parking");
        return status;
    }

    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public Map<String, String> health() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "UP");
        response.put("message", "Smart Parking System Backend is running");
        return response;
    }
}


